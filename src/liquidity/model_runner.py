from __future__ import annotations

import json
import math
import os
import signal
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import load_config
from .graphql_client import GraphQLClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# These will be set in main() from cfg.data_dir to ensure alignment with the ingestor
DATA_DIR = os.path.join(BASE_DIR, "data", "json")
MODEL_DIR = os.path.join(DATA_DIR, "model")

LN_1_0001 = float(np.log(1.0001))


def list_pool_dirs(base_dir: str) -> List[str]:
	try:
		return sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
	except FileNotFoundError:
		return []


def load_latest_json(pool_id: str) -> Dict[str, Any] | None:
	fp = os.path.join(DATA_DIR, pool_id, "latest.json")
	if not os.path.exists(fp):
		return None
	try:
		with open(fp, "r") as f:
			return json.load(f)
	except Exception:
		return None


def list_snapshots(pool_id: str) -> pd.DataFrame:
	snaps_dir = os.path.join(DATA_DIR, pool_id, "snapshots")
	rows: List[Tuple[str, str, int]] = []
	if not os.path.isdir(snaps_dir):
		return pd.DataFrame(columns=["snapshot_id", "snapped_at", "block_number"]).head(0)
	for name in os.listdir(snaps_dir):
		if not name.endswith(".json"):
			continue
		fp = os.path.join(snaps_dir, name)
		try:
			with open(fp, "r") as f:
				data = json.load(f)
			rows.append((name[:-5], data.get("snapped_at"), int(data.get("block_number", 0))))
		except Exception:
			continue
	df = pd.DataFrame(rows, columns=["snapshot_id", "snapped_at", "block_number"]).sort_values(by=["snapped_at", "block_number"], ascending=True)
	return df


def build_price_series_df(pool_id: str, max_snaps: int) -> pd.DataFrame:
	snaps = list_snapshots(pool_id)
	if snaps.empty:
		return pd.DataFrame(columns=["snapped_at", "price"])  # minimal
	snaps = snaps.sort_values(["snapped_at", "block_number"]).tail(max_snaps)
	rows: List[Dict[str, Any]] = []
	for _, r in snaps.iterrows():
		snap_id = str(r.snapshot_id)
		try:
			with open(os.path.join(DATA_DIR, pool_id, "snapshots", f"{snap_id}.json"), "r") as f:
				data = json.load(f)
			when = data.get("snapped_at") or r.snapped_at or ""
			try:
				ts = datetime.fromisoformat(when.replace("Z", "+00:00")) if when else None
			except Exception:
				ts = None
			if ts is None:
				continue
			sqrt_price_str = data.get("sqrt_price")
			price = None
			if sqrt_price_str is not None:
				try:
					sp = int(str(sqrt_price_str))
					# price1/0_raw = (sqrtPriceX96^2) / 2^192
					raw = (sp / (2**96)) ** 2
					# Need decimals: try latest.json
					latest = load_latest_json(pool_id) or {}
					d0 = int((latest.get("token0") or {}).get("decimals") or latest.get("dec0") or 18)
					d1 = int((latest.get("token1") or {}).get("decimals") or latest.get("dec1") or 18)
					price = float(raw * (10 ** (d0 - d1)))
				except Exception:
					price = None
			if price is None:
				tick = (data.get("current_tick") if data.get("current_tick") is not None else (load_latest_json(pool_id) or {}).get("current_tick"))
				if tick is not None:
					try:
						latest = load_latest_json(pool_id) or {}
						d0 = int((latest.get("token0") or {}).get("decimals") or latest.get("dec0") or 18)
						d1 = int((latest.get("token1") or {}).get("decimals") or latest.get("dec1") or 18)
						price = float(((1.0001) ** int(tick)) * (10 ** (d0 - d1)))
					except Exception:
						price = None
			if price is None:
				continue
			rows.append({"snapped_at": ts, "price": price})
		except Exception:
			continue
	if not rows:
		return pd.DataFrame(columns=["snapped_at", "price"])
	df = pd.DataFrame(rows).sort_values("snapped_at").reset_index(drop=True)
	return df


def infer_tick_spacing_from_distribution(pool_id: str) -> int:
	# Infer from latest distribution diffs
	fp = os.path.join(DATA_DIR, pool_id, "latest.json")
	try:
		with open(fp, "r") as f:
			data = json.load(f)
		active = data.get("active", [])
		ticks = sorted(set([int(a[0]) for a in active if isinstance(a, list) and len(a) >= 2]))
		if len(ticks) < 2:
			return 1
		diffs = [abs(ticks[i] - ticks[i - 1]) for i in range(1, len(ticks))]
		g = diffs[0]
		for d in diffs[1:]:
			g = math.gcd(g, d)
		return max(1, int(g))
	except Exception:
		return 1


def compute_model_effective_ticks(pool_id: str, df_series: pd.DataFrame, t_hours: float, lookback_periods: int, z_score: float, k_vol_to_tvl: float, min_ticks: int, max_ticks: int, alpha: float) -> Dict[str, Any]:
	# Fixed cadence: 5-minute snapshots
	ppy = (365.0 * 24.0 * 60.0) / 5.0
	logs = np.log(pd.to_numeric(df_series["price"], errors="coerce").astype(float)).diff().dropna().to_numpy(dtype=float)
	if logs.size < 2:
		return {"error": "not_enough_points"}
	sigma = float(np.std(logs, ddof=1))
	sigma_annual = sigma * float(np.sqrt(ppy))
	T_years = float(t_hours) / (24.0 * 365.0)
	width_factor = float(np.exp(float(z_score) * sigma_annual * float(np.sqrt(T_years)))) if T_years > 0 else 1.0
	base_ticks = int(np.ceil(np.log(max(1.0, width_factor)) / LN_1_0001))
	# vol_to_tvl from latest.json if present
	vol_to_tvl = 0.0
	latest = load_latest_json(pool_id) or {}
	try:
		vol24 = latest.get("volume24h_usd")
		tvl = latest.get("tvl_usd")
		if vol24 is not None and tvl is not None and float(tvl) > 0:
			vol_to_tvl = float(vol24) / float(tvl)
	except Exception:
		vol_to_tvl = 0.0
	# Match Streamlit: damp vol_to_tvl by alpha power before scaling
	adj = float(vol_to_tvl) ** float(alpha)
	multiplier = 1.0 / (1.0 + float(k_vol_to_tvl) * adj)
	effective_ticks = int(max(int(min_ticks), min(int(max_ticks), int(round(base_ticks * multiplier)))))
	return {
		"sigma_annual": sigma_annual,
		"periods_per_year": ppy,
		"T_years": T_years,
		"width_factor": width_factor,
		"base_ticks": base_ticks,
		"vol_to_tvl": vol_to_tvl,
		"alpha": alpha,
		"multiplier": multiplier,
		"effective_ticks": effective_ticks,
	}


def save_proposal(pool_id: str, payload: Dict[str, Any]):
	out_dir = os.path.join(MODEL_DIR, pool_id)
	os.makedirs(out_dir, exist_ok=True)
	ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
	fp = os.path.join(out_dir, f"proposal_{ts}.json")
	with open(fp, "w") as f:
		json.dump(payload, f, indent=2, sort_keys=False, default=str)
	print(f"Saved model proposal: {fp}")


def run_once_for_pool(pool_id: str, cfg) -> None:
	latest = load_latest_json(pool_id) or {}
	current_tick = latest.get("current_tick")
	if current_tick is None:
		fp = os.path.join(DATA_DIR, pool_id, "latest.json")
		if not os.path.exists(fp):
			print(f"[{pool_id}] latest.json not found at {fp}; skipping")
		else:
			print(f"[{pool_id}] No current_tick in latest.json; skipping")
		return
	current_tick = int(current_tick)
	# decimals: fetch from subgraph to ensure correctness
	try:
		client = GraphQLClient(cfg.subgraph_url, request_rps=cfg.request_rps)
		state, _ = client.fetch_pool_state(pool_id)
		t0 = state.get("token0", {}) if state else {}
		t1 = state.get("token1", {}) if state else {}
		d0 = int(t0.get("decimals") or 18)
		d1 = int(t1.get("decimals") or 18)
	except Exception:
		# fallback to latest.json if present
		t0j = latest.get("token0") or {}
		t1j = latest.get("token1") or {}
		d0 = int(t0j.get("decimals") or latest.get("dec0") or 18)
		d1 = int(t1j.get("decimals") or latest.get("dec1") or 18)
	# Build price series (use a reasonable cap)
	df_series = build_price_series_df(pool_id, max_snaps=max(500, int(cfg.lookback_periods)))
	if df_series.empty or len(df_series) < 3:
		print(f"[{pool_id}] Not enough series data; skipping")
		return
	# Use last lookback_periods
	df_series = df_series.tail(int(cfg.lookback_periods))
	m = compute_model_effective_ticks(
		pool_id,
		df_series,
		cfg.t_hours,
		cfg.lookback_periods,
		cfg.z_score,
		cfg.k_vol_to_tvl,
		cfg.min_ticks,
		cfg.max_ticks,
		cfg.alpha,
	)
	if m.get("error"):
		print(f"[{pool_id}] Model error: {m['error']}")
		return
	eff = int(m["effective_ticks"])
	# Align to inferred tick spacing
	tick_spacing = infer_tick_spacing_from_distribution(pool_id)
	aligned_half = int(math.ceil(max(0, eff) / max(1, tick_spacing)) * max(1, tick_spacing))
	lower_tick = int(math.floor((current_tick - aligned_half) / tick_spacing) * tick_spacing)
	upper_tick = int(math.ceil((current_tick + aligned_half) / tick_spacing) * tick_spacing)
	# Prices
	scale10 = float(10 ** (int(d0) - int(d1)))
	lower_price = float((1.0001 ** float(lower_tick)) * scale10)
	upper_price = float((1.0001 ** float(upper_tick)) * scale10)
	center_price = float((1.0001 ** float(current_tick)) * scale10)
	# Simple allocation weights: inside → 0.5/0.5; below → 1/0; above → 0/1
	if current_tick <= lower_tick:
		coef0_units_per_usd = 1.0 / max(1e-18, center_price)
		coef1_units_per_usd = 0.0
	elif current_tick >= upper_tick:
		coef0_units_per_usd = 0.0
		coef1_units_per_usd = 1.0
	else:
		coef0_units_per_usd = 0.5 / max(1e-18, center_price)
		coef1_units_per_usd = 0.5
	payload = {
		"snapped_at": datetime.utcnow().isoformat() + "Z",
		"pool_id": pool_id,
		"tick_spacing": tick_spacing,
		"center_tick": current_tick,
		"lower_tick": lower_tick,
		"upper_tick": upper_tick,
		"lower_price": lower_price,
		"upper_price": upper_price,
		"center_price": center_price,
		"decimals": {"token0": d0, "token1": d1},
		"model": {
			"t_hours": cfg.t_hours,
			"lookback_periods": cfg.lookback_periods,
			"z_score": cfg.z_score,
			"k_vol_to_tvl": cfg.k_vol_to_tvl,
			"min_ticks": cfg.min_ticks,
			"max_ticks": cfg.max_ticks,
			"alpha": cfg.alpha,
			"effective_ticks": eff,
			"half_width_aligned": aligned_half,
		},
		"amounts_per_unit_L": {"token0": coef0_units_per_usd, "token1": coef1_units_per_usd},
	}
	save_proposal(pool_id, payload)


def main():
	cfg = load_config()
	# Align IO paths with ingestor-configured data_dir
	global DATA_DIR, MODEL_DIR
	DATA_DIR = cfg.data_dir
	MODEL_DIR = os.path.join(DATA_DIR, "model")
	os.makedirs(MODEL_DIR, exist_ok=True)
	print(f"Starting model runner (every 2 hours); data_dir={DATA_DIR} model_dir={MODEL_DIR}")
	last_saved_at: Dict[str, float] = {}
	run = True

	def handle_sigterm(signum, frame):  # noqa: ARG001
		nonlocal run
		run = False
		print("Shutdown signal received.")

	signal.signal(signal.SIGINT, handle_sigterm)
	signal.signal(signal.SIGTERM, handle_sigterm)

	while run:
		start = time.time()
		pools = list_pool_dirs(DATA_DIR)
		# Apply whitelist if set
		if cfg.model_pools_whitelist:
			pools = [p for p in pools if p in cfg.model_pools_whitelist]
		for pool_id in pools:
			prev = last_saved_at.get(pool_id, 0.0)
			if time.time() - prev < 2 * 3600:
				continue
			try:
				run_once_for_pool(pool_id, cfg)
				last_saved_at[pool_id] = time.time()
			except Exception as exc:  # noqa: BLE001
				print(f"[{pool_id}] Error: {exc}")
		elapsed = time.time() - start
		# Wake up every 5 minutes to check
		time.sleep(max(60.0, 300.0 - elapsed))


if __name__ == "__main__":
	main() 