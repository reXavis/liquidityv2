from __future__ import annotations

import glob
import json
import os
import signal
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import load_config

# Optional: when ready to integrate web3, uncomment and add to requirements
# from web3 import Web3


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


def execute_proposal_stub(prop: Proposal) -> None:
	# Placeholder: wire Web3 here later
	print(
		f"[EXECUTE] pool={prop.pool_id} at={prop.snapped_at} ticks=({prop.lower_tick},{prop.upper_tick}) prices=({prop.lower_price},{prop.upper_price})"
	)
	# Example when ready:
	# w3 = Web3(Web3.HTTPProvider(os.environ["RPC_URL"]))
	# contract = w3.eth.contract(address=..., abi=...)
	# tx = contract.functions.someMethod(...).build_transaction({...})
	# signed = w3.eth.account.sign_transaction(tx, private_key=os.environ["PRIVATE_KEY"]) 
	# tx_hash = w3.eth.send_raw_transaction(signed.rawTransaction)
	# print("Submitted tx", tx_hash.hex())


def main():
	cfg = load_config()
	model_dir = os.path.join(cfg.data_dir, "model")
	print(f"Starting Web3 executor (watching {model_dir})")
	processed: Dict[str, float] = {}
	run = True

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
			prop = load_proposal_from_file(latest)
			if not prop:
				continue
			# TODO: add gating by time (e.g., only within N minutes of snapped_at) if desired
			execute_proposal_stub(prop)
			processed[key] = mtime
		elapsed = time.time() - start
		# Poll every minute
		time.sleep(max(5.0, 60.0 - elapsed))


if __name__ == "__main__":
	main() 