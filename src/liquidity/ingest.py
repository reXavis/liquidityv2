from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Tuple

from .graphql_client import GraphQLClient
from .storage import write_snapshot


@dataclass
class PoolMeta:
    id: str
    address: str
    token0: str
    token1: str
    fee: int
    tick_spacing: int


def fetch_pool_meta(client: GraphQLClient, address: str) -> PoolMeta:
    state, _ = client.fetch_pool_state(address)
    if not state:
        raise RuntimeError(f"Pool not found in subgraph: {address}")
    return PoolMeta(
        id=state["id"],
        address=address,
        token0=state.get("token0", {}).get("id", ""),
        token1=state.get("token1", {}).get("id", ""),
        fee=int(state.get("fee", 0) or 0),
        tick_spacing=int(state.get("tickSpacing", 0) or 0),
    )


def compute_active_from_net(net_by_tick: Dict[int, int]) -> List[Tuple[int, int]]:
    active: List[Tuple[int, int]] = []
    running = 0
    for tick in sorted(net_by_tick.keys()):
        running += net_by_tick[tick]
        active.append((tick, running))
    return active


def _build_snapshot_payload(state: Dict, ticks: Iterable[Dict]) -> List[Dict[str, str | int]]:
    net_by_tick: Dict[int, int] = {}
    for t in ticks:
        idx = int(t.get("tickIdx") or t.get("tickIndex") or 0)
        ln = int(str(t.get("liquidityNet") or "0"))
        net_by_tick[idx] = ln
    active = compute_active_from_net(net_by_tick)
    payload_ticks: List[Dict[str, str | int]] = []
    for idx, active_liq in active:
        payload_ticks.append({
            "tick_index": idx,
            "liquidity_net": str(net_by_tick.get(idx, 0)),
            "active_liquidity": str(active_liq),
        })
    return payload_ticks


def _fetch_metrics(client: GraphQLClient, pool_address: str, block_number: int | None = None) -> Tuple[float | None, float]:
    tvl = client.fetch_pool_tvl_usd(pool_address, block=block_number)
    vol24 = client.fetch_pool_volume24h_usd(pool_address)
    return tvl, float(vol24)


def ingest_full_for_pool(base_dir: str, client: GraphQLClient, pool_address: str):
    state, meta_block = client.fetch_pool_state(pool_address)
    if not state:
        raise RuntimeError(f"pool not found: {pool_address}")
    ticks = list(client.fetch_pool_ticks(pool_address, block=meta_block))
    payload_ticks = _build_snapshot_payload(state, ticks)
    tvl, vol24 = _fetch_metrics(client, pool_address, block_number=meta_block)
    meta = {
        "snapped_at": datetime.now(timezone.utc).isoformat(),
        "block_number": meta_block,
        "is_full": True,
        "current_tick": int(state.get("tick") or 0),
        "sqrt_price": str(state.get("sqrtPrice") or "0"),
        "fee_bps": int(state.get("fee") or 0),
        "tvl_usd": tvl,
        "volume24h_usd": vol24,
    }
    write_snapshot(base_dir, pool_address, meta, payload_ticks)


def ingest_delta_for_pool(base_dir: str, client: GraphQLClient, pool_address: str):
    state, meta_block = client.fetch_pool_state(pool_address)
    if not state:
        return
    ticks = list(client.fetch_pool_ticks(pool_address, block=meta_block))
    payload_ticks = _build_snapshot_payload(state, ticks)
    tvl, vol24 = _fetch_metrics(client, pool_address, block_number=meta_block)
    meta = {
        "snapped_at": datetime.now(timezone.utc).isoformat(),
        "block_number": meta_block,
        "is_full": False,
        "current_tick": int(state.get("tick") or 0),
        "sqrt_price": str(state.get("sqrtPrice") or "0"),
        "fee_bps": int(state.get("fee") or 0),
        "tvl_usd": tvl,
        "volume24h_usd": vol24,
    }
    write_snapshot(base_dir, pool_address, meta, payload_ticks) 