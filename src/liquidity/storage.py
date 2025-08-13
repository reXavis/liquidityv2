from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any


def pool_dir(base_dir: str, pool_id: str) -> str:
    d = os.path.join(base_dir, pool_id.lower())
    os.makedirs(d, exist_ok=True)
    return d


def snapshots_dir(base_dir: str, pool_id: str) -> str:
    d = os.path.join(pool_dir(base_dir, pool_id), "snapshots")
    os.makedirs(d, exist_ok=True)
    return d


def latest_path(base_dir: str, pool_id: str) -> str:
    return os.path.join(pool_dir(base_dir, pool_id), "latest.json")


def write_snapshot(base_dir: str, pool_id: str, meta: Dict[str, Any], ticks: List[Dict[str, Any]]):
    # meta includes snapped_at (ISO), block_number, is_full, current_tick, sqrt_price, fee_bps, tvl_usd, volume24h_usd
    snap_name = f"{meta['block_number']}.json"
    snap_file = os.path.join(snapshots_dir(base_dir, pool_id), snap_name)
    payload = {
        "pool_id": pool_id,
        **meta,
        "ticks": ticks,
    }
    with open(snap_file, "w") as f:
        json.dump(payload, f)

    # Update latest with compact active vector and state
    active = [(int(t["tick_index"]), str(t["active_liquidity"])) for t in ticks]
    latest = {
        "pool_id": pool_id,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "block_number": meta["block_number"],
        "current_tick": meta["current_tick"],
        "sqrt_price": meta["sqrt_price"],
        "fee_bps": meta["fee_bps"],
        "tvl_usd": meta.get("tvl_usd"),
        "volume24h_usd": meta.get("volume24h_usd"),
        "active": active,
    }
    with open(latest_path(base_dir, pool_id), "w") as f:
        json.dump(latest, f)


def list_pools(base_dir: str) -> List[str]:
    try:
        items = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        return sorted(items)
    except FileNotFoundError:
        return []


def read_latest(base_dir: str, pool_id: str) -> Dict[str, Any] | None:
    try:
        with open(latest_path(base_dir, pool_id), "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def list_snapshots(base_dir: str, pool_id: str, limit: int = 200) -> List[Dict[str, Any]]:
    dir_path = snapshots_dir(base_dir, pool_id)
    snaps = []
    try:
        for name in os.listdir(dir_path):
            if not name.endswith(".json"):
                continue
            fp = os.path.join(dir_path, name)
            try:
                with open(fp, "r") as f:
                    data = json.load(f)
                    snaps.append({
                        "snapshot_id": name[:-5],
                        "snapped_at": data.get("snapped_at"),
                        "block_number": data.get("block_number"),
                        "is_full": data.get("is_full", False),
                        "current_tick": data.get("current_tick"),
                        "tvl_usd": data.get("tvl_usd"),
                        "volume24h_usd": data.get("volume24h_usd"),
                        "file": fp,
                    })
            except Exception:
                continue
    except FileNotFoundError:
        return []
    snaps.sort(key=lambda x: (x.get("snapped_at") or "", int(x.get("block_number") or 0)), reverse=True)
    return snaps[:limit]


def read_snapshot_ticks(base_dir: str, pool_id: str, snapshot_id: str) -> List[Dict[str, Any]]:
    fp = os.path.join(snapshots_dir(base_dir, pool_id), f"{snapshot_id}.json")
    with open(fp, "r") as f:
        data = json.load(f)
    return data.get("ticks", []) 