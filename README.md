# LiquidityV2 — JSON ingestion + Plotly UI

## Prerequisites
- Python 3.13+
- Linux/macOS shell

## Install
```bash
git clone https://github.com/<your-org>/liquidityv2.git
cd liquidityv2
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Ingestor (auto-discovers pools; writes JSON under `data/json`)
```bash
PYTHONPATH=./src python -m liquidity.main
```
- Source: Goldsky Algebra Integral subgraph (hardcoded in `liquidity/config.py`).
- Storage: flat JSON files per pool under `data/json/<pool>/`.
- Snapshots: `data/json/<pool>/snapshots/<block>.json`; latest pointer: `data/json/<pool>/latest.json`.
- Active liquidity is computed from cumulative sum of `liquidityNet` across ticks.

## UI (Streamlit + Plotly)
```bash
. ./.venv/bin/activate
streamlit run ui/streamlit_app.py
```
- Select a pool (shown as `TOKEN0/TOKEN1`).
- Reads JSON files directly; no DB required.

### Tabs and Controls
- Latest
  - Distribution plotted vs price on x-axis (`token1/token0`).
  - KDE overlay with controls:
    - KDE bandwidth (ticks)
    - KDE kernel: Gaussian, Epanechnikov, Triangular, Uniform, Biweight/Quartic, Cosine
  - Quartile clipping: enable/disable and percentage per side (0–25%).
  - Core-band trimming removes remote limit ticks automatically.
  - Current price shown as a vertical line.
  - Summary chart below: Violin or Box of price positions, weighted by active liquidity.
- Snapshots
  - Same rendering/controls as Latest for a selected historical snapshot.
- Price
  - Price series (`token1/token0`) computed from `sqrtPriceX96` or `tick` with decimals: `(sqrtPriceX96/2^96)^2 * 10^(dec0-dec1)` or `(1.0001^tick) * 10^(dec0-dec1)`.
  - Missing data handling: line split on gaps; gaps shaded; no line drawn across gaps.
  - Optional bands: Classic (N), Time window (minutes), EWMA (span N), Robust quantile; window and multiplier controls.

- Models
  - Volatility-adaptive sizing model with parameters from `src/liquidity/config.py`.
  - Uses log-returns over the last `lookback_periods`; annualizes with fixed 5-minute cadence (periods_per_year = 105120).
  - Half-width ticks (one-sided) computed from `width_factor = exp(z_score * sigma_annual * sqrt(T_years))`, with `T_years = t_hours / (24*365)`.
  - Applies volume-to-TVL sensitivity: `multiplier = 1 / (1 + k_vol_to_tvl * (vol_to_tvl^alpha))` where `vol_to_tvl` is taken from latest metrics when available.
  - Clamps to `[min_ticks, max_ticks]`, aligns to pool `tickSpacing`, centers around `current_tick`.
  - Renders the current proposed band as a shaded horizontal band on the price chart.
  - Historical proposals are computed at a 2-hour cadence and shown as piecewise-constant bands over time (constant within each interval; shift only at proposal times).

## Notes
- Data directory `data/json` is created automatically if missing.
- Subgraph URL and data dir live in code (`liquidity/config.py`); no external config needed.
- If the UI shows “No pools ingested yet”, ensure the ingestor is running and JSON files appear under `data/json/`.

## Troubleshooting
- Import errors when running the ingestor: ensure `PYTHONPATH=./src` is set as shown.
- GraphQL schema errors: the ingestor targets Algebra Integral pools; ensure the Goldsky subgraph is reachable from your network. 

## Model Runner (continuous proposals every 2 hours)

```bash
PYTHONPATH=./src python -m liquidity.model_runner
```

- Reads pool JSON under `data/json/<pool>/` (written by the ingestor), builds the same price series as the UI, and computes the model.
- Produces proposals at most once every 2 hours per pool and saves them as JSON under:
  - `data/json/model/<pool>/proposal_YYYYMMDDTHHMMSSZ.json`
- Parameters come from `src/liquidity/config.py` (same as UI): `t_hours`, `lookback_periods`, `z_score`, `k_vol_to_tvl`, `alpha`, `min_ticks`, `max_ticks`.
- Aligns the proposed half-width to pool `tickSpacing` and centers around `current_tick`.
- Includes per-unit-liquidity token amounts for the range in the saved payload.
- Whitelist support: set `model_pools_whitelist = ["<pool_id>", ...]` in `src/liquidity/config.py` to restrict which pools are processed. Otherwise, all discovered pools in `data/json/` are considered.

Tip: Run the ingestor and the model runner concurrently so snapshots and latest pointers stay fresh for the model. 

## End-to-end flow (Ingestor → Model Runner)

1) Ingestor runs continuously and writes JSON under `data/json/<pool>/`.
2) Every 2 hours, the Model Runner computes a proposal per whitelisted pool and writes:
   - `data/json/model/<pool>/proposal_YYYYMMDDTHHMMSSZ.json`

Note: On-chain execution via a Web3 executor has been removed from this repository.