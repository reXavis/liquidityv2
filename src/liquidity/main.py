from __future__ import annotations

import signal
import time
from datetime import datetime
from typing import List

from .config import load_config
from .graphql_client import GraphQLClient
from .ingest import ingest_delta_for_pool, ingest_full_for_pool


def main():
    cfg = load_config()
    client = GraphQLClient(cfg.subgraph_url, request_rps=cfg.request_rps)

    last_full_at: dict[str, datetime] = {}
    pools_cache: List[str] = []
    pools_cache_ts: float = 0.0

    print("Starting ingestion loop (JSON storage). Discovering pools from subgraph...")

    run = True

    def handle_sigterm(signum, frame):  # noqa: ARG001
        nonlocal run
        run = False
        print("Shutdown signal received.")

    signal.signal(signal.SIGINT, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)

    def refresh_pools_cache():
        nonlocal pools_cache, pools_cache_ts
        now_ts = time.time()
        if now_ts - pools_cache_ts < 300 and pools_cache:
            return
        try:
            pools_cache = [p.get("id") for p in client.fetch_pools(page_size=1000)]
            pools_cache_ts = now_ts
            print(f"Discovered {len(pools_cache)} pools")
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to refresh pools list: {exc}")

    while run:
        start = time.time()
        refresh_pools_cache()
        for pool_addr in pools_cache:
            now = datetime.utcnow()
            full_due = (pool_addr not in last_full_at) or (
                (now - last_full_at.get(pool_addr, now)).total_seconds() >= cfg.full_snapshot_interval_minutes * 60
            )
            try:
                if full_due:
                    print(f"[full] {pool_addr}")
                    ingest_full_for_pool(cfg.data_dir, client, pool_addr)
                    last_full_at[pool_addr] = now
                else:
                    print(f"[delta] {pool_addr}")
                    ingest_delta_for_pool(cfg.data_dir, client, pool_addr)
            except Exception as exc:  # noqa: BLE001
                print(f"Error ingesting {pool_addr}: {exc}")
        elapsed = time.time() - start
        sleep_for = max(0.0, cfg.delta_interval_seconds - elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)


if __name__ == "__main__":
    main() 