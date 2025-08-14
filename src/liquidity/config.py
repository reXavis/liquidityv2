import os
from dataclasses import dataclass
from typing import List


@dataclass
class AppConfig:
    subgraph_url: str
    data_dir: str
    full_snapshot_interval_minutes: int
    delta_interval_seconds: int
    request_rps: float
    # Model parameters (adjustable)
    t_hours: int
    lookback_periods: int
    z_score: float
    k_vol_to_tvl: float
    min_ticks: int
    max_ticks: int
    alpha: float
    model_pools_whitelist: List[str]
    # Chain / contract config
    chain_name: str
    rpc_url: str
    position_manager_address: str
    position_manager_abi_path: str


def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return int(float(val))
    except Exception:
        return default


def _get_env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None or val == "":
        return default
    try:
        return float(val)
    except Exception:
        return default


def load_config() -> AppConfig:
    subgraph_url = "https://api.goldsky.com/api/public/project_cmb20ryy424yb01wy7zwd7xd1/subgraphs/analytics/1.2.3/gn"
    data_dir = os.path.abspath(os.path.join(os.getcwd(), "data", "json"))

    full_snapshot_interval_minutes = 60
    delta_interval_seconds = 300
    request_rps = 3.0

    # Adjustable model parameters (hardcoded defaults)
    t_hours = 6
    lookback_periods = 72
    z_score = 1.75
    k_vol_to_tvl = 0.6
    min_ticks = 50
    max_ticks = 2000
    alpha = 0.6
    model_pools_whitelist: List[str] = ['0xd391259888fe4599e8011eea5e27b93a9dc74920']  # add pool IDs here to restrict the model runner

    # Chain / contract config
    chain_name = "HyperEVM"
    rpc_url = os.getenv("RPC_URL", "")
    position_manager_address = "0x69D57B9D705eaD73a5d2f2476C30c55bD755cc2F"
    position_manager_abi_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "abi", "position_manager.json"))

    os.makedirs(data_dir, exist_ok=True)

    return AppConfig(
        subgraph_url=subgraph_url,
        data_dir=data_dir,
        full_snapshot_interval_minutes=full_snapshot_interval_minutes,
        delta_interval_seconds=delta_interval_seconds,
        request_rps=request_rps,
        t_hours=t_hours,
        lookback_periods=lookback_periods,
        z_score=z_score,
        k_vol_to_tvl=k_vol_to_tvl,
        min_ticks=min_ticks,
        max_ticks=max_ticks,
        alpha=alpha,
        model_pools_whitelist=model_pools_whitelist,
        chain_name=chain_name,
        rpc_url=rpc_url,
        position_manager_address=position_manager_address,
        position_manager_abi_path=position_manager_abi_path,
    ) 