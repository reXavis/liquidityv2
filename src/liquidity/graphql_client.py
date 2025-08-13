from __future__ import annotations

import time
from typing import Any, Dict, Iterable, Optional, Tuple

import httpx


class GraphQLClient:
    def __init__(
        self,
        endpoint_url: str,
        max_retries: int = 5,
        backoff_seconds: float = 0.8,
        request_rps: float = 3.0,
    ):
        self._url = endpoint_url
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._client = httpx.Client(timeout=30)
        # Simple rate limiter based on min interval between calls
        self._min_interval = 1.0 / max(0.1, float(request_rps))
        self._last_request_ts = 0.0

    def _post(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries):
            try:
                # Rate limiting: ensure minimum interval between requests
                now = time.monotonic()
                sleep_for = (self._last_request_ts + self._min_interval) - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                resp = self._client.post(self._url, json={"query": query, "variables": variables or {}})
                self._last_request_ts = time.monotonic()
                resp.raise_for_status()
                data = resp.json()
                if "errors" in data and data["errors"]:
                    # Treat schema parse/shape errors as non-retryable to avoid long waits
                    msg = str(data["errors"])  # usually a list of dicts
                    non_retryable_markers = (
                        "Cannot query field",
                        "Unknown argument",
                        "Unknown type",
                        "does not exist",
                        "Undefined field",
                        "Unexpected",
                    )
                    if any(marker in msg for marker in non_retryable_markers):
                        raise RuntimeError(f"Non-retryable GraphQL schema error: {msg}")
                    raise RuntimeError(f"GraphQL errors: {data['errors']}")
                return data["data"]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # If schema/parse error, do not retry
                if isinstance(exc, RuntimeError) and "Non-retryable GraphQL schema error" in str(exc):
                    break
                time.sleep(self._backoff * (2 ** attempt))
        raise RuntimeError(f"GraphQL request failed after retries: {last_exc}")

    def fetch_pools(self, page_size: int = 1000) -> Iterable[Dict[str, Any]]:
        query = (
            "query($first:Int!,$skip:Int!){\n"
            "  pools(first:$first, skip:$skip, orderBy:id, orderDirection:asc){\n"
            "    id token0{ id symbol decimals } token1{ id symbol decimals } fee tickSpacing createdAtBlock:createdAtBlockNumber\n"
            "  }\n"
            "}"
        )
        skip = 0
        while True:
            data = self._post(query, {"first": page_size, "skip": skip})
            items = data.get("pools", [])
            if not items:
                break
            for it in items:
                yield it
            skip += page_size

    def fetch_pool_state(self, pool_id: str, block: Optional[int] = None) -> Tuple[Dict[str, Any], int]:
        query = (
            "query($id:ID!,$block:Block_height){\n"
            "  pool(id:$id, block:$block){ id tick sqrtPrice fee tickSpacing token0{ id symbol decimals } token1{ id symbol decimals } }\n"
            "  _meta{ block{ number } }\n"
            "}"
        )
        variables: Dict[str, Any] = {"id": pool_id}
        if block is not None:
            variables["block"] = {"number": block}
        data = self._post(query, variables)
        meta_block = data.get("_meta", {}).get("block", {}).get("number", block or 0)
        return data.get("pool"), int(meta_block)

    def fetch_pool_ticks(self, pool_id: str, block: Optional[int] = None, page_size: int = 1000) -> Iterable[Dict[str, Any]]:
        query = (
            "query($pool:String!,$first:Int!,$skip:Int!,$block:Block_height){\n"
            "  ticks(first:$first, skip:$skip, where:{ poolAddress:$pool }, block:$block, orderBy:tickIdx, orderDirection:asc){\n"
            "    tickIdx liquidityNet liquidityGross\n"
            "  }\n"
            "}"
        )
        skip = 0
        while True:
            variables: Dict[str, Any] = {"pool": pool_id, "first": page_size, "skip": skip}
            if block is not None:
                variables["block"] = {"number": block}
            data = self._post(query, variables)
            items = data.get("ticks", [])
            if not items:
                break
            for it in items:
                yield it
            skip += page_size

    # --- Helper methods for TVL ---
    def fetch_pool_tvl_usd(self, pool_id: str, block: Optional[int] = None) -> Optional[float]:
        # Keep the query minimal and target the canonical field
        query = (
            "query($id:ID!,$block:Block_height){\n"
            "  pool(id:$id, block:$block){ id totalValueLockedUSD }\n"
            "  _meta{ block{ number } }\n"
            "}"
        )
        variables: Dict[str, Any] = {"id": pool_id}
        if block is not None:
            variables["block"] = {"number": block}
        try:
            data = self._post(query, variables)
            pool = data.get("pool") or {}
            val = pool.get("totalValueLockedUSD")
            return float(val) if val is not None else None
        except Exception:
            return None

    def fetch_pool_volume24h_usd(self, pool_id: str) -> float:
        # Mirror bot.py exactly: use poolHourDatas, periodStartUnix, volumeUSD; filter by pool
        query = (
            "query($pool:String!){\n"
            "  poolHourDatas(first:1000, orderBy:periodStartUnix, orderDirection:desc, where:{ pool:$pool }){\n"
            "    periodStartUnix\n"
            "    volumeUSD\n"
            "  }\n"
            "}"
        )
        try:
            data = self._post(query, {"pool": pool_id})
            items = data.get("poolHourDatas", [])
            if not items:
                return 0.0
            try:
                latest_hour = int(items[0].get("periodStartUnix") or 0)
            except Exception:
                latest_hour = 0
            if latest_hour <= 0:
                return 0.0
            cutoff = latest_hour - 24 * 3600
            total = 0.0
            for it in items:
                try:
                    ts = int(it.get("periodStartUnix") or 0)
                except Exception:
                    continue
                if ts < cutoff:
                    break
                try:
                    v = float(it.get("volumeUSD") or 0.0)
                except Exception:
                    v = 0.0
                total += v
            return float(total)
        except Exception:
            return 0.0 