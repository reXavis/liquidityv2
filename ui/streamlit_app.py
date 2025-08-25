import os
import json
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
import numpy as np
import plotly.graph_objs as go
from datetime import datetime
import sys
import math

# Ensure local src/ is on PYTHONPATH for `liquidity` imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from liquidity.graphql_client import GraphQLClient
from liquidity.config import load_config

DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, "data", "json"))
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="Liquidity Distributions", layout="wide")
st.title("Liquidity Distributions — Algebra Integral (JSON)")
st.caption("Note: On-chain execution via a Web3 executor has been removed from this repository.")

@st.cache_data(ttl=120)
def list_pools_with_labels() -> List[Tuple[str, str]]:
    cfg = load_config()
    cli = GraphQLClient(cfg.subgraph_url)
    items: List[Tuple[str, str]] = []
    for p in cli.fetch_pools(page_size=500):
        pid = p.get("id")
        t0 = p.get("token0", {})
        t1 = p.get("token1", {})
        sym0 = (t0.get("symbol") or t0.get("id") or "?").upper()
        sym1 = (t1.get("symbol") or t1.get("id") or "?").upper()
        label = f"{sym0}/{sym1} ({pid[:6]}…{pid[-4:]})"
        items.append((pid, label))
    return items

@st.cache_data(ttl=300)
def get_pool_metadata(pool_id: str) -> Tuple[int, int, str, str]:
    cfg = load_config()
    cli = GraphQLClient(cfg.subgraph_url)
    state, _ = cli.fetch_pool_state(pool_id)
    t0 = state.get("token0", {}) if state else {}
    t1 = state.get("token1", {}) if state else {}
    d0 = int(t0.get("decimals") or 18)
    d1 = int(t1.get("decimals") or 18)
    s0 = (t0.get("symbol") or t0.get("id") or "?").upper()
    s1 = (t1.get("symbol") or t1.get("id") or "?").upper()
    return d0, d1, s0, s1

@st.cache_data(ttl=60)
def list_pools() -> List[str]:
    try:
        return sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    except FileNotFoundError:
        return []

@st.cache_data(ttl=30)
def load_latest(pool_id: str) -> pd.DataFrame:
    fp = os.path.join(DATA_DIR, pool_id, "latest.json")
    if not os.path.exists(fp):
        return pd.DataFrame(columns=["tick_index", "active_liquidity"])
    try:
        with open(fp, "r") as f:
            data = json.load(f)
        active = data.get("active", [])
        df = pd.DataFrame(active, columns=["tick_index", "active_liquidity"])  # type: ignore
        return df
    except Exception:
        return pd.DataFrame(columns=["tick_index", "active_liquidity"])

@st.cache_data(ttl=30)
def load_latest_meta(pool_id: str) -> Dict[str, Any] | None:
    fp = os.path.join(DATA_DIR, pool_id, "latest.json")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=30)
def list_snapshots(pool_id: str) -> pd.DataFrame:
    snaps_dir = os.path.join(DATA_DIR, pool_id, "snapshots")
    rows: List[Tuple[str, str, int, bool, float | None, float | None]] = []
    if not os.path.isdir(snaps_dir):
        return pd.DataFrame(columns=["snapshot_id", "snapped_at", "block_number", "is_full", "tvl_usd", "volume24h_usd"])
    for name in os.listdir(snaps_dir):
        if not name.endswith(".json"):
            continue
        fp = os.path.join(snaps_dir, name)
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            rows.append((name[:-5], data.get("snapped_at"), int(data.get("block_number", 0)), bool(data.get("is_full", False)), data.get("tvl_usd"), data.get("volume24h_usd")))
        except Exception:
            continue
    df = pd.DataFrame(rows, columns=["snapshot_id", "snapped_at", "block_number", "is_full", "tvl_usd", "volume24h_usd"]).sort_values(by=["snapped_at", "block_number"], ascending=False)
    return df  # Removed .head(200) to show all snapshots

@st.cache_data(ttl=30)
def load_snapshot(pool_id: str, snapshot_id: str) -> pd.DataFrame:
    fp = os.path.join(DATA_DIR, pool_id, "snapshots", f"{snapshot_id}.json")
    if not os.path.exists(fp):
        return pd.DataFrame(columns=["tick_index", "active_liquidity"])
    try:
        with open(fp, "r") as f:
            data = json.load(f)
        ticks = data.get("ticks", [])
        df = pd.DataFrame([{ "tick_index": t.get("tick_index"), "active_liquidity": t.get("active_liquidity") } for t in ticks])
        return df
    except Exception:
        return pd.DataFrame(columns=["tick_index", "active_liquidity"])

@st.cache_data(ttl=30)
def load_snapshot_meta(pool_id: str, snapshot_id: str) -> Dict[str, Any] | None:
    fp = os.path.join(DATA_DIR, pool_id, "snapshots", f"{snapshot_id}.json")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _trim_to_core_region(df_plot: pd.DataFrame, value_field: str, current_tick: int | None) -> pd.DataFrame:
    if df_plot.empty:
        return df_plot
    peak = float(df_plot[value_field].max())
    if peak <= 0:
        return df_plot
    thr = peak * 1e-3  # 0.1% of peak
    mask = (df_plot[value_field] >= thr).values
    ticks = df_plot["tick_index"].values
    segments: List[Tuple[int, int]] = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if (not m or i == len(mask) - 1) and start is not None:
            end = i if not m else i
            segments.append((start, end))
            start = None
    if not segments:
        return df_plot
    def seg_range(seg: Tuple[int, int]) -> Tuple[int, int]:
        lo = int(ticks[seg[0]])
        hi = int(ticks[seg[1]])
        return lo, hi
    chosen = None
    if current_tick is not None:
        for seg in segments:
            lo, hi = seg_range(seg)
            if lo <= current_tick <= hi:
                chosen = seg
                break
    if chosen is None:
        chosen = max(segments, key=lambda s: s[1] - s[0])
    lo, hi = seg_range(chosen)
    span = max(1, hi - lo)
    margin = int(min(10000, max(50, span * 0.02)))
    lo -= margin
    hi += margin
    return df_plot[(df_plot["tick_index"] >= lo) & (df_plot["tick_index"] <= hi)]


def _weighted_quantile(xs: pd.Series, ws: pd.Series, q: float) -> float:
    c = (ws.cumsum() / ws.sum()).values
    xvals = xs.values
    idx = (c >= q).argmax()
    return float(xvals[idx])


def _clip_by_quartiles(df_plot: pd.DataFrame, value_field: str, current_tick: int | None, q_each_side: float) -> pd.DataFrame:
    if df_plot.empty:
        return df_plot
    df_sorted = df_plot.sort_values("tick_index").reset_index(drop=True)
    xs = pd.to_numeric(df_sorted["tick_index"], errors="coerce").fillna(0)
    ws = pd.to_numeric(df_sorted[value_field], errors="coerce").clip(lower=0).fillna(0)
    if float(ws.sum()) <= 0:
        return df_plot
    q_low = max(0.0, min(0.25, q_each_side))
    q_high = 1.0 - q_low
    q1 = _weighted_quantile(xs, ws, q_low)
    q3 = _weighted_quantile(xs, ws, q_high)
    iqr = max(1.0, q3 - q1)
    if current_tick is not None:
        # Center window on the current tick with symmetric half-width equal to IQR
        center = float(current_tick)
        half_width = float(iqr)
        lo = center - half_width
        hi = center + half_width
    else:
        # Fallback: expand around quartiles when no current tick is available
        lo = q1 - 0.5 * iqr
        hi = q3 + 0.5 * iqr
    return df_sorted[(df_sorted["tick_index"] >= lo) & (df_sorted["tick_index"] <= hi)]


def compute_kde_over_ticks(df_plot: pd.DataFrame, bandwidth_ticks: int, kernel: str = "Gaussian") -> Tuple[np.ndarray, np.ndarray]:
    ticks = df_plot["tick_index"].to_numpy(dtype=float)
    weights = df_plot["active_liquidity"].to_numpy(dtype=float)
    if len(ticks) == 0:
        return np.array([]), np.array([])
    ticks_sorted = np.sort(ticks)
    diffs = np.diff(ticks_sorted)
    step = int(np.median(diffs)) if diffs.size > 0 else 1
    step = max(step, 1)
    tmin = int(ticks_sorted.min())
    tmax = int(ticks_sorted.max())
    grid = np.arange(tmin, tmax + step, step, dtype=int)
    signal = np.zeros_like(grid, dtype=float)
    # accumulate weights into grid bins
    idx = np.clip(((ticks - tmin) / step).round().astype(int), 0, len(grid) - 1)
    for i, w in zip(idx, weights):
        signal[i] += float(w)

    k = (kernel or "Gaussian").lower()
    if k == "gaussian":
        sigma_pts = max(1.0, bandwidth_ticks / step)
        half = int(3 * sigma_pts)
        kx = np.arange(-half, half + 1, dtype=float)
        kern = np.exp(-0.5 * (kx / sigma_pts) ** 2)
    else:
        h = max(1.0, bandwidth_ticks / step)
        half = int(h)
        kx = np.arange(-half, half + 1, dtype=float)
        u = np.clip(kx / h, -1.0, 1.0)
        if k in ("epanechnikov", "epan"):
            kern = 0.75 * (1.0 - u**2)
            kern[np.abs(u) > 1.0] = 0.0
        elif k in ("triangular", "triangle"):
            kern = 1.0 - np.abs(u)
            kern[np.abs(u) > 1.0] = 0.0
        elif k in ("uniform", "tophat", "rect"):
            kern = np.where(np.abs(u) <= 1.0, 1.0, 0.0)
        elif k in ("biweight", "quartic"):
            kern = (15.0 / 16.0) * (1.0 - u**2) ** 2
            kern[np.abs(u) > 1.0] = 0.0
        elif k in ("cosine", "cos"):
            # normalized cosine kernel
            kern = (np.pi / 4.0) * np.cos((np.pi / 2.0) * u)
            kern[np.abs(u) > 1.0] = 0.0
        else:
            # fallback to gaussian
            sigma_pts = max(1.0, bandwidth_ticks / step)
            half = int(3 * sigma_pts)
            kx = np.arange(-half, half + 1, dtype=float)
            kern = np.exp(-0.5 * (kx / sigma_pts) ** 2)

    s = kern.sum()
    if s <= 0 or not np.isfinite(s):
        s = 1.0
    kern = kern / s

    density = np.convolve(signal, kern, mode="same")
    # scale density to match max of active for overlay
    scale = signal.max() if signal.max() > 0 else 1.0
    if density.max() > 0:
        density = density * (scale / density.max())
    return grid.astype(float), density


def render_tick_distribution_plotly(
    df: pd.DataFrame,
    current_tick: int | None,
    enable_quartile_clip: bool,
    clip_percent_each_side: int,
    kde_bandwidth: int,
    d0: int,
    d1: int,
    sym0: str,
    sym1: str,
    kde_kernel: str = "Gaussian",
    y_title: str = "Active Liquidity",
):
    if df.empty:
        st.info("No data to display.")
        return
    df_plot = df.copy()
    df_plot["tick_index"] = pd.to_numeric(df_plot["tick_index"], errors="coerce")
    df_plot["active_liquidity"] = pd.to_numeric(df_plot["active_liquidity"], errors="coerce")
    df_plot = df_plot.dropna().sort_values("tick_index").reset_index(drop=True)

    value_field = "active_liquidity"
    df_plot = _trim_to_core_region(df_plot, value_field, current_tick)
    if enable_quartile_clip and clip_percent_each_side > 0:
        df_plot = _clip_by_quartiles(df_plot, value_field, current_tick, q_each_side=clip_percent_each_side / 100.0)
    if df_plot.empty:
        st.info("No data in view after trimming.")
        return

    # Convert tick to price for x-axis
    scale10 = float(10 ** (d0 - d1))
    x_prices = (np.power(1.0001, df_plot["tick_index"].to_numpy(dtype=float)) * scale10)

    fig = go.Figure()
    # Step area for distribution
    fig.add_trace(
        go.Scatter(
            x=x_prices,
            y=df_plot[value_field],
            mode="lines",
            line=dict(color="#4c78a8"),
            name="Distribution",
            fill="tozeroy",
            line_shape="hv",
            customdata=df_plot["tick_index"].to_numpy(),
            hovertemplate="Price=%{x}<br>Tick=%{customdata}<br>Active=%{y}<extra></extra>",
        )
    )
    # KDE overlay computed on ticks, mapped to price grid
    grid, density = compute_kde_over_ticks(df_plot, kde_bandwidth, kernel=kde_kernel)
    if grid.size > 0:
        grid_prices = np.power(1.0001, grid.astype(float)) * scale10
        fig.add_trace(
            go.Scatter(
                x=grid_prices,
                y=density,
                mode="lines",
                line=dict(color="#e45756", width=1.5),
                name="KDE",
                hoverinfo="skip",
            )
        )
    if current_tick is not None:
        current_price = float((1.0001 ** float(current_tick)) * scale10)
        fig.add_vline(x=current_price, line_color="#e45756", opacity=0.8, line_width=1)

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        xaxis=dict(title=f"Price {sym1}/{sym0}"),
        yaxis=dict(title=None, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def _sample_ticks_weighted(df_plot: pd.DataFrame, max_samples: int) -> np.ndarray:
    values = pd.to_numeric(df_plot["tick_index"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(df_plot["active_liquidity"], errors="coerce").clip(lower=0).to_numpy(dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]
    if values.size == 0:
        return np.array([])
    total = float(weights.sum())
    if total <= 0:
        # fall back to uniform sample of ticks
        take = min(max_samples, values.size)
        rng = np.random.default_rng()
        idx = rng.choice(values.size, size=take, replace=(values.size < take))
        return values[idx]
    probs = weights / total
    take = int(min(max_samples, max(100, values.size)))
    rng = np.random.default_rng()
    idx = rng.choice(values.size, size=take, replace=True, p=probs)
    return values[idx]


def render_distribution_summary_plotly(
    df: pd.DataFrame,
    current_tick: int | None,
    enable_quartile_clip: bool,
    clip_percent_each_side: int,
    kind: str = "Violin",
    max_samples: int = 5000,
    d0: int | None = None,
    d1: int | None = None,
    sym0: str | None = None,
    sym1: str | None = None,
):
    if df.empty:
        return
    df_plot = df.copy()
    df_plot["tick_index"] = pd.to_numeric(df_plot["tick_index"], errors="coerce")
    df_plot["active_liquidity"] = pd.to_numeric(df_plot["active_liquidity"], errors="coerce")
    df_plot = df_plot.dropna().sort_values("tick_index").reset_index(drop=True)

    # Apply same visibility trimming
    df_plot = _trim_to_core_region(df_plot, "active_liquidity", current_tick)
    if enable_quartile_clip and clip_percent_each_side > 0:
        df_plot = _clip_by_quartiles(df_plot, "active_liquidity", current_tick, q_each_side=clip_percent_each_side / 100.0)
    if df_plot.empty:
        return

    samples_ticks = _sample_ticks_weighted(df_plot, max_samples=max_samples)
    if samples_ticks.size == 0:
        return

    # Convert sampled ticks to price if metadata provided
    if d0 is not None and d1 is not None:
        scale10 = float(10 ** (int(d0) - int(d1)))
        samples = np.power(1.0001, samples_ticks.astype(float)) * scale10
        x_title = f"Price {sym1}/{sym0}" if sym0 and sym1 else "Price token1/token0"
    else:
        samples = samples_ticks
        x_title = "Tick"

    fig = go.Figure()
    if kind == "Box":
        fig.add_trace(
            go.Box(
                x=samples,
                orientation="h",
                boxpoints="outliers",
                marker=dict(color="#4c78a8", size=2),
                line=dict(color="#4c78a8"),
                fillcolor="rgba(76,120,168,0.15)",
                name="Box",
                hoverinfo="skip",
            )
        )
    else:
        fig.add_trace(
            go.Violin(
                x=samples,
                orientation="h",
                line=dict(color="#e45756", width=1),
                fillcolor="rgba(228,87,86,0.25)",
                name="Violin",
                meanline_visible=False,
                points=False,
                spanmode="hard",
                hoverinfo="skip",
            )
        )
    fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=0, b=20),
        showlegend=False,
        xaxis=dict(title=x_title),
        yaxis=dict(visible=False, showticklabels=False),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


@st.cache_data(ttl=30)
def build_price_series_df(pool_id: str, max_snaps: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    snaps = list_snapshots(pool_id)
    if snaps.empty:
        return pd.DataFrame(columns=["snapped_at", "price", "tvl_usd", "volume24h_usd"]), pd.DataFrame(columns=["start", "end"])
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
                    # adjust by decimals: human price1/0 = raw * 10^(dec0 - dec1)
                    d0, d1, _, _ = get_pool_metadata(pool_id)
                    price = float(raw * (10 ** (d0 - d1)))
                except Exception:
                    price = None
            if price is None:
                tick = data.get("current_tick")
                if tick is not None:
                    try:
                        d0, d1, _, _ = get_pool_metadata(pool_id)
                        price = float(((1.0001) ** int(tick)) * (10 ** (d0 - d1)))
                    except Exception:
                        price = None
            if price is None:
                continue
            # Coerce metrics to float or None
            def _to_float(x):
                try:
                    return float(x)
                except Exception:
                    return None
            rows.append({
                "snapped_at": ts,
                "price": price,
                "tvl_usd": _to_float(data.get("tvl_usd")),
                "volume24h_usd": _to_float(data.get("volume24h_usd")),
            })
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["snapped_at", "price", "tvl_usd", "volume24h_usd"]), pd.DataFrame(columns=["start", "end"])
    df = pd.DataFrame(rows).sort_values("snapped_at").reset_index(drop=True)
    # Detect missing gaps based on median interval
    if len(df) >= 3:
        diffs = df["snapped_at"].diff().dt.total_seconds().dropna()
        med = max(1.0, float(diffs.median()))
        thr = max(med * 3.0, 3600.0)  # at least 1 hour to flag a gap
        gaps = []
        for i in range(1, len(df)):
            gap_s = (df.loc[i, "snapped_at"] - df.loc[i - 1, "snapped_at"]).total_seconds()
            if gap_s > thr:
                gaps.append({"start": df.loc[i - 1, "snapped_at"], "end": df.loc[i, "snapped_at"]})
        gaps_df = pd.DataFrame(gaps)
    else:
        gaps_df = pd.DataFrame(columns=["start", "end"])
    return df, gaps_df


def render_price_series_plotly(
    df: pd.DataFrame,
    gaps_df: pd.DataFrame,
    y_title: str,
    bands: pd.DataFrame | None = None,
    hover_extra: Dict[str, Any] | None = None,
    static_band: tuple[float, float] | None = None,
):
    if df.empty:
        st.info("No snapshot data to render price series.")
        return

    # Build segments from sorted data
    d_sorted = df.sort_values("snapped_at").reset_index(drop=True)
    if len(d_sorted) >= 2:
        diffs = d_sorted["snapped_at"].diff().dt.total_seconds().fillna(0)
        med = max(1.0, float(diffs[diffs > 0].median() if (diffs > 0).any() else 1.0))
        thr = max(med * 3.0, 3600.0)
    else:
        thr = 3600.0
    seg_x: list[list] = [[]]
    seg_y: list[list] = [[]]
    seg_cd: list[list[tuple]] = [[]]
    for i, row in d_sorted.iterrows():
        if i > 0:
            gap_s = (d_sorted.loc[i, "snapped_at"] - d_sorted.loc[i - 1, "snapped_at"]).total_seconds()
            if gap_s > thr:
                seg_x.append([])
                seg_y.append([])
                seg_cd.append([])
        seg_x[-1].append(row["snapped_at"])
        seg_y[-1].append(row["price"])
        seg_cd[-1].append((row.get("tvl_usd"), row.get("volume24h_usd")))

    # Build a fresh figure
    fig = go.Figure()
    # Price line as segments split on large time gaps
    for xs, ys, cds in zip(seg_x, seg_y, seg_cd):
        if len(xs) >= 2:
            hovertemplate = "%{x|%b %d %H:%M}<br>Price=%{y}"
            # If per-point data is present, reference via customdata
            use_cd = cds and any(cd is not None for cd in cds)
            if use_cd:
                hovertemplate += "<br>TVL(USD)=%{customdata[0]:.2f}<br>Vol 24h(USD)=%{customdata[1]:.2f}"
            elif hover_extra is not None:
                tvl = hover_extra.get("tvl_usd")
                vol = hover_extra.get("volume24h_usd")
                if tvl is not None:
                    hovertemplate += f"<br>TVL(USD)={tvl:,.2f}"
                if vol is not None:
                    hovertemplate += f"<br>Vol 24h(USD)={vol:,.2f}"
            hovertemplate += "<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=y_title,
                    line=dict(color="#e45756"),
                    hovertemplate=hovertemplate,
                    connectgaps=False,
                    customdata=cds if use_cd else None,
                )
            )

    # Bands: create traces per segment so bands do not span gaps
    if bands is not None and not bands.empty:
        try:
            b = pd.merge(d_sorted[["snapped_at"]], bands, on="snapped_at", how="left").set_index("snapped_at")
        except Exception:
            b = None
        if b is not None:
            for xs in seg_x:
                if len(xs) < 2:
                    continue
                try:
                    seg_b = b.loc[xs]
                except Exception:
                    continue
                # Upper
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=seg_b["upper"],
                        mode="lines",
                        line=dict(color="#4c78a8", width=1),
                        name="Upper band",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                # Lower (filled to upper)
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=seg_b["lower"],
                        mode="lines",
                        line=dict(color="#4c78a8", width=1),
                        name="Lower band",
                        fill="tonexty",
                        fillcolor="rgba(76,120,168,0.12)",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                # Mean
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=seg_b["mean"],
                        mode="lines",
                        line=dict(color="#4c78a8", width=1, dash="dash"),
                        name="Mean",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

    # Gap shading via shapes
    if not gaps_df.empty:
        for _, g in gaps_df.iterrows():
            fig.add_vrect(
                x0=g["start"],
                x1=g["end"],
                fillcolor="#999999",
                opacity=0.2,
                layer="below",
                line_width=0,
            )

    # Static horizontal band (e.g., proposed ticks converted to price)
    if static_band is not None:
        try:
            y0, y1 = float(static_band[0]), float(static_band[1])
            if y0 > y1:
                y0, y1 = y1, y0
            fig.add_hrect(y0=y0, y1=y1, fillcolor="rgba(255,193,7,0.15)", line_width=0, layer="below")
            # Upper and lower boundary lines
            fig.add_hline(y=y0, line_color="#ffc107", line_width=1)
            fig.add_hline(y=y1, line_color="#ffc107", line_width=1)
        except Exception:
            pass

    fig.update_layout(
        height=360,
        margin=dict(l=10, r=10, t=10, b=30),
        showlegend=False,
        xaxis=dict(title="Time"),
        yaxis=dict(title=y_title),
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})


def compute_bands(
    df: pd.DataFrame,
    method: str,
    window: int,
    multiplier: float,
    min_points: int = 5,
) -> pd.DataFrame:
    if df.empty or "price" not in df.columns:
        return pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"])
    s = pd.to_numeric(df["price"], errors="coerce")
    t = pd.to_datetime(df["snapped_at"], errors="coerce")
    base = pd.DataFrame({"snapped_at": t, "price": s}).dropna().sort_values("snapped_at")
    if base.empty:
        return pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"])
    if method == "Classic (N points)":
        mean = base["price"].rolling(window=window, min_periods=min_points).mean()
        std = base["price"].rolling(window=window, min_periods=min_points).std()
        upper = mean + multiplier * std
        lower = mean - multiplier * std
    elif method == "Time window (minutes)":
        ts_idx = base.set_index("snapped_at")["price"]
        roll = ts_idx.rolling(f"{window}T", min_periods=min_points)
        mean = roll.mean().reindex(ts_idx.index)
        std = roll.std().reindex(ts_idx.index)
        upper = mean + multiplier * std
        lower = mean - multiplier * std
        # Restore to base order
        mean = mean.reset_index(drop=True)
        upper = upper.reset_index(drop=True)
        lower = lower.reset_index(drop=True)
    elif method == "EWMA (span N)":
        # Use ewm; guard with min_points via expanding count
        count = base["price"].expanding().count()
        mean = base["price"].ewm(span=window, adjust=False).mean()
        try:
            std = base["price"].ewm(span=window, adjust=False).std()
        except Exception:
            var = base["price"].ewm(span=window, adjust=False).var()
            std = var.pow(0.5)
        mean[count < min_points] = float("nan")
        std[count < min_points] = float("nan")
        upper = mean + multiplier * std
        lower = mean - multiplier * std
    elif method == "Robust quantile (N points)":
        qh = min(0.49, multiplier)  # multiplier interpreted as half-width quantile (0..0.49)
        ql = 1.0 - qh
        mean = base["price"].rolling(window=window, min_periods=min_points).median()
        upper = base["price"].rolling(window=window, min_periods=min_points).quantile(qh)
        lower = base["price"].rolling(window=window, min_periods=min_points).quantile(1 - qh)
    else:
        return pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"])
    out = pd.DataFrame({
        "snapped_at": base["snapped_at"],
        "mean": mean,
        "upper": upper,
        "lower": lower,
    })
    return out


def compute_bands_segmented(
    df: pd.DataFrame,
    method: str,
    window: int,
    multiplier: float,
    min_points: int = 5,
    gaps_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if df.empty or "price" not in df.columns or "snapped_at" not in df.columns:
        return pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"])
    d_sorted = df.dropna(subset=["snapped_at", "price"]).sort_values("snapped_at").reset_index(drop=True)
    if d_sorted.empty:
        return pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"])
    # Always detect gaps using the same threshold heuristic as the renderer
    diffs = d_sorted["snapped_at"].diff().dt.total_seconds().fillna(0)
    med = max(1.0, float(diffs[diffs > 0].median() if (diffs > 0).any() else 1.0))
    thr = max(med * 3.0, 3600.0)
    split_indices: list[int] = []
    for i in range(1, len(d_sorted)):
        gap_s = (d_sorted.loc[i, "snapped_at"] - d_sorted.loc[i - 1, "snapped_at"]).total_seconds()
        if gap_s > thr:
            split_indices.append(i)

    # Build segments
    segments: list[pd.DataFrame] = []
    last = 0
    for idx in split_indices + [len(d_sorted)]:
        seg = d_sorted.iloc[last:idx]
        if len(seg) >= 2:
            segments.append(seg)
        last = idx

    if not segments:
        return compute_bands(d_sorted, method, window, multiplier, min_points=min_points)

    # Compute per-segment bands and concatenate
    out_parts: list[pd.DataFrame] = []
    for seg in segments:
        part = compute_bands(seg, method, window, multiplier, min_points=min_points)
        out_parts.append(part)
    out = pd.concat(out_parts, ignore_index=True).sort_values("snapped_at")
    return out


# ===== Models helpers (from example.py math) =====
LN_1_0001 = float(np.log(1.0001))


def _realized_vol_annualized(log_returns: np.ndarray, periods_per_year: float) -> float:
    if log_returns.size < 2 or periods_per_year <= 0:
        return 0.0
    sigma = float(np.std(log_returns, ddof=1))
    return sigma * float(np.sqrt(periods_per_year))


def _ticks_from_width_factor(width_factor: float) -> int:
    if not np.isfinite(width_factor) or width_factor <= 1.0:
        return 0
    return int(np.ceil(np.log(float(width_factor)) / LN_1_0001))

def compute_model_effective_ticks(
    df_series: pd.DataFrame,
    cfg,
) -> dict:
    d = df_series.dropna(subset=["snapped_at", "price"]).sort_values("snapped_at").tail(int(max(1, cfg.lookback_periods)))
    if d.empty or len(d) < 2:
        return {"error": "not_enough_points"}
    # Fixed cadence: snapshots every 5 minutes → periods_per_year = 365*24*60/5
    ppy = (365.0 * 24.0 * 60.0) / 5.0
    med_s = 300.0
    logs = np.log(pd.to_numeric(d["price"], errors="coerce").astype(float)).diff().dropna().to_numpy(dtype=float)
    sigma_annual = _realized_vol_annualized(logs, periods_per_year=ppy)
    T_years = float(cfg.t_hours) / (24.0 * 365.0)
    width_factor = float(np.exp(float(cfg.z_score) * sigma_annual * float(np.sqrt(T_years)))) if T_years > 0 else 1.0
    base_ticks = _ticks_from_width_factor(width_factor)
    # vol_to_tvl: use most recent row with both present, else 0
    vol_to_tvl = 0.0
    d_v = d.dropna(subset=["tvl_usd", "volume24h_usd"]) if ("tvl_usd" in d.columns and "volume24h_usd" in d.columns) else pd.DataFrame()
    if not d_v.empty:
        tvl = float(d_v["tvl_usd"].iloc[-1])
        vol = float(d_v["volume24h_usd"].iloc[-1])
        if tvl > 0:
            vol_to_tvl = vol / tvl

    adj = vol_to_tvl ** float(cfg.alpha)
    multiplier = 1.0 / (1.0 + float(cfg.k_vol_to_tvl) * adj)
    effective_ticks = int(max(int(cfg.min_ticks), min(int(cfg.max_ticks), int(round(base_ticks * multiplier)))))
    return {
        "sigma_annual": sigma_annual,
        "periods_per_year": ppy,
        "median_dt_seconds": med_s,
        "T_years": T_years,
        "width_factor": width_factor,
        "base_ticks": base_ticks,
        "vol_to_tvl": vol_to_tvl,
        "multiplier": multiplier,
        "effective_ticks": effective_ticks,
    }


def compute_model_historical_bands(
    df_series: pd.DataFrame,
    cfg,
    tick_spacing: int,
    d0: int,
    d1: int,
    min_interval_seconds: int = 7200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_series.empty:
        return (
            pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"]),
            pd.DataFrame(columns=[]),
        )
    d = df_series.dropna(subset=["snapped_at", "price"]).sort_values("snapped_at").reset_index(drop=True)
    if d.empty:
        return (
            pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"]),
            pd.DataFrame(columns=[]),
        )
    scale10 = float(10 ** (int(d0) - int(d1)))
    proposals: list[dict] = []
    last_prop_ts = None
    for i in range(len(d)):
        ts = d.loc[i, "snapped_at"]
        if last_prop_ts is not None:
            delta = (ts - last_prop_ts).total_seconds()
            if delta < float(min_interval_seconds):
                continue
        # subset up to current ts
        up_to = d.iloc[: i + 1]
        m = compute_model_effective_ticks(up_to, cfg)
        eff = int(m.get("effective_ticks") or 0)
        if eff <= 0:
            continue
        half_aligned = int(math.ceil(eff / max(1, int(tick_spacing))) * max(1, int(tick_spacing)))
        # center tick from latest price
        price_t = float(up_to.loc[up_to.index[-1], "price"])
        # convert price to tick
        try:
            center_tick = int(round(np.log(price_t / scale10) / LN_1_0001))
        except Exception:
            continue
        lower_raw = center_tick - half_aligned
        upper_raw = center_tick + half_aligned
        lower_tick = int(math.floor(lower_raw / tick_spacing) * tick_spacing)
        upper_tick = int(math.ceil(upper_raw / tick_spacing) * tick_spacing)
        lower_price = float((1.0001 ** float(lower_tick)) * scale10)
        upper_price = float((1.0001 ** float(upper_tick)) * scale10)
        mean_price = float((1.0001 ** float(center_tick)) * scale10)
        proposals.append({
            "snapped_at": ts,
            "center_tick": center_tick,
            "tick_spacing": int(tick_spacing),
            "base_ticks": int(m.get("base_ticks") or 0),
            "effective_ticks": eff,
            "half_width_aligned": half_aligned,
            "lower_tick": lower_tick,
            "upper_tick": upper_tick,
            "lower": lower_price,
            "upper": upper_price,
            "mean": mean_price,
            # diagnostics
            "sigma_annual": m.get("sigma_annual"),
            "periods_per_year": m.get("periods_per_year"),
            "median_dt_seconds": m.get("median_dt_seconds"),
            "T_years": m.get("T_years"),
            "width_factor": m.get("width_factor"),
            "vol_to_tvl": m.get("vol_to_tvl"),
            "multiplier": m.get("multiplier"),
        })
        last_prop_ts = ts
    if not proposals:
        return (
            pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"]),
            pd.DataFrame(columns=[]),
        )
    props_df = pd.DataFrame(proposals).dropna().sort_values("snapped_at").reset_index(drop=True)
    # Assign each price timestamp the last-known proposal (constant within interval)
    base_times = d[["snapped_at"]].sort_values("snapped_at").reset_index(drop=True)
    full = pd.merge_asof(base_times, props_df, on="snapped_at", direction="backward")
    # If early timestamps precede first proposal, they remain NaN; renderer will skip
    return full[["snapped_at", "mean", "upper", "lower"]], props_df


def _parse_any_timestamp(v: Any) -> datetime | None:
    try:
        if isinstance(v, str) and v:
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None
    return None


def load_external_testmodel_bands(pool_id: str, df_series: pd.DataFrame, d0: int, d1: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_dir = os.path.join(DATA_DIR, "testmodel", pool_id)
    if not os.path.isdir(base_dir):
        return pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"]), pd.DataFrame(columns=[])
    rows: list[dict] = []
    props: list[dict] = []
    scale10 = float(10 ** (int(d0) - int(d1)))
    display_scale = 10e11
    for name in sorted(os.listdir(base_dir)):
        if not name.endswith(".json"):
            continue
        fp = os.path.join(base_dir, name)
        try:
            with open(fp, "r") as f:
                data = json.load(f)
            ts = _parse_any_timestamp(data.get("snapped_at") or data.get("timestamp") or data.get("time"))
            if ts is None:
                continue
            lower_p = data.get("lower_price")
            upper_p = data.get("upper_price")
            mean_p = data.get("center_price") or data.get("mean_price")
            if lower_p is None or upper_p is None:
                # Try ticks + decimals
                lt = data.get("lower_tick")
                ut = data.get("upper_tick")
                ct = data.get("center_tick")
                if lt is None or ut is None:
                    continue
                lower_p = float((1.0001 ** float(int(lt))) * scale10)
                upper_p = float((1.0001 ** float(int(ut))) * scale10)
                if mean_p is None:
                    if ct is not None:
                        mean_p = float((1.0001 ** float(int(ct))) * scale10)
                    else:
                        mean_p = (float(lower_p) + float(upper_p)) / 2.0
            props.append({
                "snapped_at": ts,
                "lower": float(lower_p) * display_scale,
                "upper": float(upper_p) * display_scale,
                "mean": float(mean_p if mean_p is not None else ((float(lower_p) + float(upper_p)) / 2.0)) * display_scale,
            })
        except Exception:
            continue
    if not props:
        return pd.DataFrame(columns=["snapped_at", "mean", "upper", "lower"]), pd.DataFrame(columns=[])
    props_df = pd.DataFrame(props).sort_values("snapped_at").reset_index(drop=True)
    # Map proposals to piecewise-constant bands over our price series timeline
    if df_series.empty or "snapped_at" not in df_series.columns:
        return props_df[["snapped_at", "mean", "upper", "lower"]], props_df
    base_times = df_series.dropna(subset=["snapped_at"]).sort_values("snapped_at")[["snapped_at"]].reset_index(drop=True)
    full = pd.merge_asof(base_times, props_df, on="snapped_at", direction="backward")
    return full[["snapped_at", "mean", "upper", "lower"]], props_df


@st.cache_data(ttl=300)
def get_pool_tick_spacing(pool_id: str) -> int:
    cfg = load_config()
    cli = GraphQLClient(cfg.subgraph_url)
    state, _ = cli.fetch_pool_state(pool_id)
    try:
        return int(state.get("tickSpacing") or 1)
    except Exception:
        return 1


pools = list_pools_with_labels()
if not pools:
    st.info("No pools ingested yet. Start the ingestor to populate JSON data.")
else:
    labels = [label for (_id, label) in pools]
    sel_label = st.selectbox("Select pool", labels)
    label_to_id = {label: pid for (pid, label) in pools}
    sel_addr = label_to_id[sel_label]

    tab_latest, tab_history, tab_price, tab_models, tab_model_test = st.tabs(["Latest", "Snapshots", "Price", "Models", "Model Test"])

    with tab_latest:
        meta = load_latest_meta(sel_addr) or {}
        current_tick = int(meta.get("current_tick")) if "current_tick" in meta else None
        df_full = load_latest(sel_addr)
        # Legend for TVL/24h Vol
        tvl = meta.get("tvl_usd")
        vol = meta.get("volume24h_usd")
        legend_parts = []
        if tvl is not None:
            legend_parts.append(f"TVL: ${tvl:,.2f}")
        if vol is not None:
            legend_parts.append(f"24h Vol: ${vol:,.2f}")
        if legend_parts:
            st.caption(" | ".join(legend_parts))
        toggle = st.checkbox("Enable quartile clipping", value=True)
        clip_pct = st.slider("Clip percentile each side (%)", min_value=0, max_value=25, value=25, step=1, help="0 disables clipping; 25 matches classic Q1/Q3")
        bw = st.slider("KDE bandwidth (ticks)", min_value=10, max_value=500, value=50, step=10)
        kde_kernel = st.selectbox("KDE kernel", ["Gaussian", "Epanechnikov", "Triangular", "Uniform", "Biweight", "Quartic", "Cosine"])
        d0, d1, sym0, sym1 = get_pool_metadata(sel_addr)
        render_tick_distribution_plotly(df_full, current_tick, enable_quartile_clip=toggle, clip_percent_each_side=clip_pct, kde_bandwidth=bw, d0=d0, d1=d1, sym0=sym0, sym1=sym1, kde_kernel=kde_kernel)
        # Summary chart controls and render
        st.caption("Summary of tick locations weighted by active liquidity")
        kind = st.selectbox("Summary chart type", ["Violin", "Box"], index=0)
        render_distribution_summary_plotly(df_full, current_tick, enable_quartile_clip=toggle, clip_percent_each_side=clip_pct, kind=kind, d0=d0, d1=d1, sym0=sym0, sym1=sym1)

    with tab_history:
        snaps = list_snapshots(sel_addr)
        if snaps.empty:
            st.info("No snapshots yet.")
        else:
            sel = st.selectbox(
                "Select snapshot",
                options=[f"{r.snapshot_id} | block {int(r.block_number)} | {'full' if r.is_full else 'delta'} | {r.snapped_at}" for _, r in snaps.iterrows()],
            )
            snap_id = sel.split("|")[0].strip()
            meta = load_snapshot_meta(sel_addr, snap_id) or {}
            # Display TVL/Vol for snapshot
            tvl = meta.get("tvl_usd")
            vol = meta.get("volume24h_usd")
            legend_parts = []
            if tvl is not None:
                legend_parts.append(f"TVL: ${tvl:,.2f}")
            if vol is not None:
                legend_parts.append(f"24h Vol: ${vol:,.2f}")
            if legend_parts:
                st.caption(" | ".join(legend_parts))
            current_tick = int(meta.get("current_tick")) if "current_tick" in meta else None
            df_full = load_snapshot(sel_addr, snap_id)
            toggle = st.checkbox("Enable quartile clipping (snap)", value=True)
            clip_pct = st.slider("Clip percentile each side (snap) (%)", min_value=0, max_value=25, value=25, step=1)
            bw = st.slider("KDE bandwidth (snap) (ticks)", min_value=50, max_value=50000, value=1000, step=50)
            kde_kernel = st.selectbox("KDE kernel (snap)", ["Gaussian", "Epanechnikov", "Triangular", "Uniform", "Biweight", "Quartic", "Cosine"])
            d0, d1, sym0, sym1 = get_pool_metadata(sel_addr)
            render_tick_distribution_plotly(df_full, current_tick, enable_quartile_clip=toggle, clip_percent_each_side=clip_pct, kde_bandwidth=bw, d0=d0, d1=d1, sym0=sym0, sym1=sym1, kde_kernel=kde_kernel)

    with tab_price:
        st.caption("Price over time; shaded regions indicate missing data intervals")
        max_snaps = st.slider("Snapshots to include", min_value=10, max_value=5000, value=1000, step=10)
        df_series, df_gaps = build_price_series_df(sel_addr, max_snaps=max_snaps)
        # Derive axis label from selector label
        try:
            pair = sel_label.split(" ")[0]  # "SYM0/SYM1 ("...
            sym0, sym1 = pair.split("/")
            y_title = f"Price {sym1}/{sym0}"
        except Exception:
            y_title = "Price token1/token0"
        # Bands controls (immediate apply)
        bands_df = pd.DataFrame()
        bands_enable = st.checkbox("Show bands", value=False)
        if bands_enable and not df_series.empty:
            method = st.selectbox(
                "Bands method",
                ["Classic (N points)", "Time window (minutes)", "EWMA (span N)", "Robust quantile (N points)"],
                index=0,
            )
            if method == "Time window (minutes)":
                window = st.slider("Window (minutes)", min_value=1, max_value=240, value=30, step=1)
                mult_label = "Std multiplier (k)"
                mult_min, mult_max, mult_step = 0.5, 3.0, 0.1
                mult_val = 2.0
            elif method == "Robust quantile (N points)":
                window = st.slider("Window (points)", min_value=5, max_value=200, value=50, step=1)
                mult_label = "Half-quantile width (0..0.49)"
                mult_min, mult_max, mult_step = 0.05, 0.45, 0.01
                mult_val = 0.25
            else:
                window = st.slider("Window (points)", min_value=5, max_value=200, value=50, step=1)
                mult_label = "Std multiplier (k)"
                mult_min, mult_max, mult_step = 0.5, 3.0, 0.1
                mult_val = 2.0
            multiplier = st.slider(mult_label, min_value=mult_min, max_value=mult_max, value=mult_val, step=mult_step)
            min_points = 5
            bands_df = compute_bands_segmented(df_series, method, window, multiplier, min_points=min_points, gaps_df=df_gaps)
        # Get latest meta for hover extras
        latest_meta = load_latest_meta(sel_addr) or {}
        hover_extra = {
            "tvl_usd": latest_meta.get("tvl_usd"),
            "volume24h_usd": latest_meta.get("volume24h_usd"),
        }
        render_price_series_plotly(df_series, df_gaps, y_title=y_title, bands=bands_df, hover_extra=hover_extra)

    with tab_models:
        st.caption("Models: volatility-adaptive sizing prototype; shaded regions indicate missing data intervals")
        # Same price chart configuration as the Price tab (with proposed band overlay)
        max_snaps = st.slider("Snapshots to include (models)", min_value=10, max_value=5000, value=1000, step=10)
        df_series, df_gaps = build_price_series_df(sel_addr, max_snaps=max_snaps)
        try:
            pair = sel_label.split(" ")[0]
            sym0, sym1 = pair.split("/")
            y_title = f"Price {sym1}/{sym0}"
        except Exception:
            y_title = "Price token1/token0"
        latest_meta = load_latest_meta(sel_addr) or {}
        cfg = load_config()
        model_out = compute_model_effective_ticks(df_series, cfg)
        # Align effective_ticks to tick spacing and center around current tick
        tick_spacing = get_pool_tick_spacing(sel_addr)
        effective_ticks = int(model_out.get("effective_ticks") or 0)
        # Align the half-width to spacing by rounding up to the next multiple
        aligned_half = int(math.ceil(max(0, effective_ticks) / max(1, tick_spacing)) * max(1, tick_spacing))
        current_tick = latest_meta.get("current_tick")
        lower_price = upper_price = None
        lower_aligned = upper_aligned = None
        d0, d1, _, _ = get_pool_metadata(sel_addr)
        if current_tick is not None:
            c = int(current_tick)
            lower_raw = c - aligned_half
            upper_raw = c + aligned_half
            # Align bounds to spacing: lower down to multiple, upper up to multiple
            lower_aligned = int(math.floor(lower_raw / tick_spacing) * tick_spacing)
            upper_aligned = int(math.ceil(upper_raw / tick_spacing) * tick_spacing)
            # Convert aligned ticks to price using token decimals
            scale10 = float(10 ** (int(d0) - int(d1)))
            lower_price = float((1.0001 ** float(lower_aligned)) * scale10)
            upper_price = float((1.0001 ** float(upper_aligned)) * scale10)

        # Historical bands at 2-hour cadence
        hist_bands, hist_props = compute_model_historical_bands(df_series, cfg, tick_spacing=tick_spacing, d0=d0, d1=d1, min_interval_seconds=7200)

        # Render chart with band overlay if available
        hover_extra = {
            "tvl_usd": latest_meta.get("tvl_usd"),
            "volume24h_usd": latest_meta.get("volume24h_usd"),
        }
        static_band = (lower_price, upper_price) if (lower_price is not None and upper_price is not None) else None
        render_price_series_plotly(df_series, df_gaps, y_title=y_title, bands=hist_bands, hover_extra=hover_extra, static_band=static_band)

        # Diagnostics
        with st.expander("Model outputs"):
            st.write(model_out)
            if lower_aligned is not None and upper_aligned is not None:
                st.write({
                    "tick_spacing": tick_spacing,
                    "center_tick": int(current_tick) if current_tick is not None else None,
                    "half_width_effective": effective_ticks,
                    "half_width_aligned": aligned_half,
                    "lower_tick": lower_aligned,
                    "upper_tick": upper_aligned,
                    "lower_price": lower_price,
                    "upper_price": upper_price,
                })
            if not hist_props.empty:
                st.write({"historical_proposals": int(len(hist_props))})
                st.dataframe(hist_props)

    with tab_model_test:
        st.caption("Model Test: overlay external proposals from data/json/testmodel/<pool>/ on the price chart")
        max_snaps = st.slider("Snapshots to include (test)", min_value=10, max_value=5000, value=2000, step=10)
        df_series, df_gaps = build_price_series_df(sel_addr, max_snaps=max_snaps)
        # y-axis title
        try:
            pair = sel_label.split(" ")[0]
            sym0, sym1 = pair.split("/")
            y_title = f"Price {sym1}/{sym0}"
        except Exception:
            y_title = "Price token1/token0"
        d0, d1, _, _ = get_pool_metadata(sel_addr)
        ext_bands, ext_props = load_external_testmodel_bands(sel_addr, df_series, d0=d0, d1=d1)
        latest_meta = load_latest_meta(sel_addr) or {}
        hover_extra = {
            "tvl_usd": latest_meta.get("tvl_usd"),
            "volume24h_usd": latest_meta.get("volume24h_usd"),
        }
        render_price_series_plotly(df_series, df_gaps, y_title=y_title, bands=ext_bands, hover_extra=hover_extra)
        if ext_props.empty:
            st.info("No external proposals found under data/json/testmodel/<pool>/.")
        else:
            st.write({"external_proposals": int(len(ext_props))})
            st.dataframe(ext_props) 