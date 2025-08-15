from __future__ import annotations

import glob
import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .config import load_config
from web3 import Web3
from web3.middleware import geth_poa_middleware

# Minimal ABI for pool token addresses
POOL_ABI = [
	{
		"inputs": [],
		"name": "token0",
		"outputs": [{"internalType": "address", "name": "", "type": "address"}],
		"stateMutability": "view",
		"type": "function",
	},
	{
		"inputs": [],
		"name": "token1",
		"outputs": [{"internalType": "address", "name": "", "type": "address"}],
		"stateMutability": "view",
		"type": "function",
	},
]

UINT128_MAX = (1 << 128) - 1


@dataclass
class Proposal:
	pool_id: str
	snapped_at: str
	center_tick: int
	lower_tick: int
	upper_tick: int
	lower_price: float
	upper_price: float
	center_price: float
	decimals_token0: int
	decimals_token1: int

	@staticmethod
	def from_json(pool_id: str, data: Dict[str, Any]) -> Optional["Proposal"]:
		try:
			return Proposal(
				pool_id=pool_id,
				snapped_at=str(data.get("snapped_at")),
				center_tick=int(data.get("center_tick")),
				lower_tick=int(data.get("lower_tick")),
				upper_tick=int(data.get("upper_tick")),
				lower_price=float(data.get("lower_price")),
				upper_price=float(data.get("upper_price")),
				center_price=float(data.get("center_price")),
				decimals_token0=int((data.get("decimals") or {}).get("token0", 18)),
				decimals_token1=int((data.get("decimals") or {}).get("token1", 18)),
			)
		except Exception:
			return None


def list_latest_proposal_files(model_dir: str, pool_id: str, max_files: int = 1) -> List[str]:
	glob_pat = os.path.join(model_dir, pool_id, "proposal_*.json")
	files = sorted(glob.glob(glob_pat))
	return files[-max_files:]


def load_proposal_from_file(path: str) -> Optional[Proposal]:
	try:
		with open(path, "r") as f:
			data = json.load(f)
		pool_id = data.get("pool_id") or ""
		return Proposal.from_json(pool_id, data)
	except Exception:
		return None


def build_mint_params(cfg, prop: Proposal, perL: Dict[str, Any]) -> Dict[str, Any]:
	desired0 = float(cfg.playerliquidity) * float(perL.get("token0", 0.0))
	desired1 = float(cfg.playerliquidity) * float(perL.get("token1", 0.0))
	min0 = float(cfg.slippage) * desired0
	min1 = float(cfg.slippage) * desired1
	deadline = int((datetime.utcnow() + timedelta(minutes=30)).timestamp())
	return {
		"tickLower": int(prop.lower_tick),
		"tickUpper": int(prop.upper_tick),
		"amount0Desired": int(desired0),
		"amount1Desired": int(desired1),
		"amount0Min": int(min0),
		"amount1Min": int(min1),
		# recipient is set to account.address at submission
		"deadline": deadline,
	}


def main():
	cfg = load_config()
	model_dir = os.path.join(cfg.data_dir, "model")
	print(f"Starting Web3 executor (watching {model_dir}) on {cfg.chain_name}")
	processed: Dict[str, float] = {}
	run = True

	# Setup w3
	w3 = Web3(Web3.HTTPProvider(cfg.rpc_url, request_kwargs={"timeout": 30}))
	# HyperEVM may require POA middleware; include for safety
	w3.middleware_onion.inject(geth_poa_middleware, layer=0)
	acct_pk = os.getenv("PRIVATE_KEY")
	if not acct_pk:
		print("PRIVATE_KEY env var not set; cannot send transactions. Running in dry-run mode.")
	account = w3.eth.account.from_key(acct_pk) if acct_pk else None
	# Load ABI and contract
	with open(cfg.position_manager_abi_path, "r") as f:
		abi = json.load(f)
	pm = w3.eth.contract(address=Web3.to_checksum_address(cfg.position_manager_address), abi=abi)

	# Fetch deployer once
	try:
		deployer_addr = pm.functions.poolDeployer().call()
		deployer_addr = Web3.to_checksum_address(deployer_addr)
		print("PositionManager poolDeployer:", deployer_addr)
	except Exception as exc:
		print("Failed to fetch poolDeployer:", exc)
		deployer_addr = "0x0000000000000000000000000000000000000000"

	def handle_sigterm(signum, frame):  # noqa: ARG001
		nonlocal run
		run = False
		print("Shutdown signal received.")

	signal.signal(signal.SIGINT, handle_sigterm)
	signal.signal(signal.SIGTERM, handle_sigterm)

	while run:
		start = time.time()
		# Scan per pool
		if not os.path.isdir(model_dir):
			time.sleep(10.0)
			continue
		for pool_id in os.listdir(model_dir):
			pool_path = os.path.join(model_dir, pool_id)
			if not os.path.isdir(pool_path):
				continue
			latest_files = list_latest_proposal_files(model_dir, pool_id, max_files=1)
			if not latest_files:
				continue
			latest = latest_files[-1]
			mtime = os.path.getmtime(latest)
			key = f"{pool_id}:{latest}"
			if processed.get(key) == mtime:
				continue
			# Load proposal and per-unit amounts
			try:
				with open(latest, "r") as f:
					raw = json.load(f)
			except Exception:
				continue
			prop = Proposal.from_json(pool_id, raw)
			if not prop:
				continue
			perL = raw.get("amounts_per_unit_L", {})
			mint_args = build_mint_params(cfg, prop, perL)

			# Fetch token addresses from pool
			try:
				pool_addr = Web3.to_checksum_address(prop.pool_id)
				pool = w3.eth.contract(address=pool_addr, abi=POOL_ABI)
				token0_addr = Web3.to_checksum_address(pool.functions.token0().call())
				token1_addr = Web3.to_checksum_address(pool.functions.token1().call())
			except Exception as exc:
				print(f"[{pool_id}] Failed to fetch token addresses:", exc)
				token0_addr = "0x0000000000000000000000000000000000000000"
				token1_addr = "0x0000000000000000000000000000000000000000"

			# Before mint: for existing matching positions, run collect and decreaseLiquidity(half)
			if account is not None:
				try:
					bal = pm.functions.balanceOf(account.address).call()
					for idx in range(int(bal)):
						token_id = pm.functions.tokenOfOwnerByIndex(account.address, idx).call()
						pos = pm.functions.positions(token_id).call()
						pos_token0 = Web3.to_checksum_address(pos[2])  # token0
						pos_token1 = Web3.to_checksum_address(pos[3])  # token1
						pos_deployer = Web3.to_checksum_address(pos[4])  # deployer
						liquidity = int(pos[7])  # uint128
						# Match by deployer and token pair
						if pos_deployer == deployer_addr and ((pos_token0 == token0_addr and pos_token1 == token1_addr) or (pos_token0 == token1_addr and pos_token1 == token0_addr)):
							# Collect
							collect_params = (
								int(token_id),
								Web3.to_checksum_address(account.address),
								int(UINT128_MAX),
								int(UINT128_MAX),
							)
							try:
								tx = pm.functions.collect(collect_params).build_transaction({
									"from": account.address,
									"nonce": w3.eth.get_transaction_count(account.address),
									"chainId": w3.eth.chain_id,
									"maxFeePerGas": w3.to_wei("50", "gwei"),
									"maxPriorityFeePerGas": w3.to_wei("2", "gwei"),
								})
								signed = w3.eth.account.sign_transaction(tx, private_key=acct_pk)
								tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
								print(f"[COLLECT] tokenId={token_id} tx={tx_hash.hex()}")
							except Exception as exc:
								print(f"[COLLECT] error tokenId={token_id}:", exc)
							# Decrease full liquidity if present
							if liquidity > 0:
								liq_full = int(liquidity)
								min_out = int(float(cfg.slippage) * 0.5 * float(liquidity))
								dec_params = (
									int(token_id),
									int(liq_full),
									int(min_out),
									int(min_out),
									int((datetime.utcnow() + timedelta(minutes=30)).timestamp()),
								)
								try:
									tx = pm.functions.decreaseLiquidity(dec_params).build_transaction({
										"from": account.address,
										"nonce": w3.eth.get_transaction_count(account.address),
										"chainId": w3.eth.chain_id,
										"maxFeePerGas": w3.to_wei("50", "gwei"),
										"maxPriorityFeePerGas": w3.to_wei("2", "gwei"),
									})
									signed = w3.eth.account.sign_transaction(tx, private_key=acct_pk)
									tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
									print(f"[DECREASE] tokenId={token_id} full={liq_full} tx={tx_hash.hex()}")
								except Exception as exc:
									print(f"[DECREASE] error tokenId={token_id}:", exc)
				except Exception as exc:
					print("[MAINTENANCE] error:", exc)

			# Build and (optionally) send tx
			print(f"[MINT] pool={pool_id} args={mint_args} token0={token0_addr} token1={token1_addr} deployer={deployer_addr}")
			if account is not None:
				try:
					# The ABI shows mint takes a single struct tuple named params; build accordingly
					tx = pm.functions.mint((
						token0_addr,
						token1_addr,
						deployer_addr,
						int(mint_args["tickLower"]),
						int(mint_args["tickUpper"]),
						int(mint_args["amount0Desired"]),
						int(mint_args["amount1Desired"]),
						int(mint_args["amount0Min"]),
						int(mint_args["amount1Min"]),
						Web3.to_checksum_address(account.address),
						int(mint_args["deadline"]),
					)).build_transaction({
						"from": account.address,
						"nonce": w3.eth.get_transaction_count(account.address),
						"chainId": w3.eth.chain_id,
						"maxFeePerGas": w3.to_wei("50", "gwei"),
						"maxPriorityFeePerGas": w3.to_wei("2", "gwei"),
					})
					signed = w3.eth.account.sign_transaction(tx, private_key=acct_pk)
					tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
					print("[MINT] submitted:", tx_hash.hex())
				except Exception as exc:
					print("[MINT] error:", exc)
			processed[key] = mtime
		elapsed = time.time() - start
		# Poll every minute
		time.sleep(max(5.0, 60.0 - elapsed))


if __name__ == "__main__":
	main() 