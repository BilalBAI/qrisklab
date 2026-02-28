"""
Plot regime signals (RV_1h, RV_24h, RV_ratio, Trend Score, IV_30d) for ETH.

Data sources (same as regime_backtest.py):
    - ETH/USDT hourly OHLCV: Binance public API
    - IV_30d: Deribit ETH DVOL index

Usage:
    python research/regime/plot_signals.py
    python research/regime/plot_signals.py --start 2025-12-01 --end 2026-02-28
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Data Fetching (mirrors regime_backtest.py)
# ---------------------------------------------------------------------------

def fetch_binance_ohlcv(
    symbol: str = "ETHUSDT",
    interval: str = "1h",
    start_dt: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc),
    end_dt: datetime = datetime(2026, 2, 28, 23, tzinfo=timezone.utc),
) -> pd.DataFrame:
    url = "https://api.binance.us/api/v3/klines"
    all_rows: list = []
    current = start_dt
    limit = 1000

    print(f"Fetching Binance {symbol} {interval} from {start_dt.date()} to {end_dt.date()}...")

    while current < end_dt:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": int(current.timestamp() * 1000),
            "endTime": int(end_dt.timestamp() * 1000),
            "limit": limit,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            break
        all_rows.extend(rows)
        last_ts = rows[-1][0]
        current = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc) + timedelta(hours=1)
        time.sleep(0.2)

    if not all_rows:
        raise ValueError("No data returned from Binance")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"  → {len(df):,} bars fetched ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
    return df


def fetch_deribit_dvol(
    start_dt: datetime,
    end_dt: datetime,
) -> pd.Series:
    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    all_rows: list = []
    window = timedelta(days=10)

    print(f"Fetching Deribit ETH DVOL from {start_dt.date()} to {end_dt.date()}...")
    current = start_dt
    while current < end_dt:
        batch_end = min(current + window, end_dt)
        params = {
            "currency": "ETH",
            "resolution": "3600",
            "start_timestamp": int(current.timestamp() * 1000),
            "end_timestamp": int(batch_end.timestamp() * 1000),
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("result", {})
            rows = data.get("data", [])
            for row in rows:
                all_rows.append({
                    "timestamp": pd.Timestamp(row[0], unit="ms", tz="UTC"),
                    "dvol": float(row[4]) / 100.0,
                })
        except Exception as exc:
            print(f"  WARNING: DVOL fetch failed for {current.date()}–{batch_end.date()} ({exc})")
        current = batch_end + timedelta(hours=1)
        time.sleep(0.5)

    if not all_rows:
        raise ValueError("No DVOL data returned from Deribit")

    dvol_df = pd.DataFrame(all_rows).sort_values("timestamp").drop_duplicates("timestamp")
    dvol_series = dvol_df.set_index("timestamp")["dvol"]
    print(f"  → {len(dvol_series):,} DVOL bars")
    return dvol_series


# ---------------------------------------------------------------------------
# Signal Computation (mirrors regime_backtest.py)
# ---------------------------------------------------------------------------

def compute_parkinson_rv(df: pd.DataFrame) -> pd.DataFrame:
    ln_hl = np.log(df["high"] / df["low"])
    parkinson_var = (ln_hl ** 2) / (4 * np.log(2))

    df = df.copy()
    df["parkinson_var"] = parkinson_var
    ann_factor = 365 * 24
    df["rv_1h"] = np.sqrt(parkinson_var * ann_factor)
    df["rv_24h"] = np.sqrt(parkinson_var.rolling(24, min_periods=12).mean() * ann_factor)
    df["rv_ratio"] = (df["rv_1h"] / df["rv_24h"]).clip(0, 10)
    return df


def compute_trend_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=6, adjust=False).mean()
    ema_slow = df["close"].ewm(span=24, adjust=False).mean()
    ema_drift = (ema_fast - ema_slow) / df["close"]

    sma_20d = df["close"].rolling(480, min_periods=240).mean()
    price_vs_ma = (df["close"] - sma_20d) / df["close"]

    ema_component = ema_drift.clip(-0.02, 0.02) / 0.02
    ma_component = price_vs_ma.clip(-0.05, 0.05) / 0.05

    df["trend_score"] = (0.6 * ema_component + 0.4 * ma_component).clip(-1.0, 1.0)
    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_signals(df: pd.DataFrame, display_start: datetime) -> go.Figure:
    """Build a 6-panel Plotly chart: Price, RV_1h, RV_24h, RV_ratio, Trend, IV_30d."""
    plot_df = df[df["timestamp"] >= display_start].copy()

    fig = make_subplots(
        rows=6, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "ETH/USDT Price",
            "RV_1h (annualized)",
            "RV_24h (annualized)",
            "RV Ratio (RV_1h / RV_24h)",
            "Trend Score",
            "IV_30d (DVOL)",
        ),
        row_heights=[0.22, 0.16, 0.16, 0.16, 0.16, 0.16],
    )

    # 1. Price
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=plot_df["close"],
            name="ETH Price", line=dict(color="#636EFA", width=1),
        ),
        row=1, col=1,
    )

    # 2. RV_1h
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=plot_df["rv_1h"],
            name="RV_1h", line=dict(color="#EF553B", width=1),
        ),
        row=2, col=1,
    )

    # 3. RV_24h
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=plot_df["rv_24h"],
            name="RV_24h", line=dict(color="#00CC96", width=1),
        ),
        row=3, col=1,
    )

    # 4. RV Ratio
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=plot_df["rv_ratio"],
            name="RV Ratio", line=dict(color="#AB63FA", width=1),
        ),
        row=4, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5, row=4, col=1)

    # 5. Trend Score
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=plot_df["trend_score"],
            name="Trend Score", line=dict(color="#FFA15A", width=1),
            fill="tozeroy", fillcolor="rgba(255,161,90,0.15)",
        ),
        row=5, col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=5, col=1)

    # 6. IV_30d
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=plot_df["iv_30d"],
            name="IV_30d (DVOL)", line=dict(color="#19D3F3", width=1),
        ),
        row=6, col=1,
    )

    # Y-axis labels
    fig.update_yaxes(title_text="USD", row=1, col=1)
    fig.update_yaxes(title_text="Ann. Vol", row=2, col=1)
    fig.update_yaxes(title_text="Ann. Vol", row=3, col=1)
    fig.update_yaxes(title_text="Ratio", row=4, col=1)
    fig.update_yaxes(title_text="Score", row=5, col=1)
    fig.update_yaxes(title_text="Ann. Vol", row=6, col=1)

    fig.update_layout(
        title="ETH Regime Signals  |  2026-01-01 → 2026-02-28",
        height=1100,
        showlegend=False,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=40),
    )
    fig.update_xaxes(title_text="Time (UTC)", row=6, col=1)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ETH regime signals")
    parser.add_argument("--start", default="2026-01-01", help="Display start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-02-28", help="End date (YYYY-MM-DD)")
    parser.add_argument("--no-dvol", action="store_true", help="Use RV proxy instead of Deribit DVOL")
    parser.add_argument("--save-html", type=str, default=None, help="Save chart to HTML file")
    args = parser.parse_args()

    display_start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(hour=23, tzinfo=timezone.utc)

    # Fetch extra 30 days before display_start for signal warm-up
    # (trend_score needs 480 bars = 20 days of SMA, plus rolling windows)
    fetch_start = display_start - timedelta(days=30)

    # 1. Fetch OHLCV
    price_df = fetch_binance_ohlcv(start_dt=fetch_start, end_dt=end_dt)

    # 2. Compute RV signals
    price_df = compute_parkinson_rv(price_df)
    price_df = compute_trend_score(price_df)

    # 3. Fetch IV_30d
    if args.no_dvol:
        print("Using IV proxy: RV_24h × 1.20")
        price_df["iv_30d"] = price_df["rv_24h"] * 1.20
    else:
        try:
            dvol = fetch_deribit_dvol(start_dt=fetch_start, end_dt=end_dt)
            price_df = price_df.set_index("timestamp")
            price_df["iv_30d"] = dvol.reindex(price_df.index).ffill().bfill()
            price_df = price_df.reset_index()
        except Exception as e:
            print(f"  WARNING: DVOL fetch failed ({e}); falling back to RV proxy")
            price_df["iv_30d"] = price_df["rv_24h"] * 1.20

    # 4. Plot (only the display window, warm-up bars are trimmed)
    fig = plot_signals(price_df, display_start)

    if args.save_html:
        fig.write_html(args.save_html)
        print(f"Chart saved to {args.save_html}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
