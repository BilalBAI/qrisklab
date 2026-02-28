"""
Regime Signal Backtesting
=========================
Replays the Crypto VolAlpha regime state machine against ETH 2021-2024
historical data to calibrate thresholds and validate event detection.

Data sources:
    - ETH/USDT hourly OHLCV: Binance public API (no auth required)
    - IV_30d: Deribit ETH DVOL index (free, no auth required)
    - Gas score: synthetic (real data requires Etherscan API key)

Usage:
    python qrisklab/research/regime/regime_backtest.py
    python qrisklab/research/regime/regime_backtest.py --start 2022-01-01 --end 2022-12-31
    python qrisklab/research/regime/regime_backtest.py --rv-threshold-elevated 0.50 --iv-threshold-elevated 0.65

See docs/regime_backtest.md for full methodology.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeThresholds:
    """
    Regime transition thresholds. All RV/IV values are annualized decimals
    (e.g., 0.55 = 55%).
    """
    # CALM → ELEVATED (ANY trigger)
    # Calibrated 2026-02-27 against ETH 2021-2024 (regime_states.md §3)
    calm_to_elevated_rv24h: float = 0.80       # ETH p75 RV; genuine vol breakout
    calm_to_elevated_rv_ratio: float = 1.5
    calm_to_elevated_trend: float = 0.45       # |trend_score| threshold
    calm_to_elevated_iv30d: float = 1.10       # IV pricing real stress

    # ELEVATED → CALM (ALL must revert, sustained ≥ hysteresis_elevated_calm_h hours)
    elevated_to_calm_rv24h: float = 0.65       # ETH p50 RV; genuine calm
    elevated_to_calm_rv_ratio: float = 1.2
    elevated_to_calm_trend: float = 0.30
    elevated_to_calm_iv30d: float = 0.85
    hysteresis_elevated_calm_h: int = 4

    # ELEVATED → STRESS (ANY trigger)
    elevated_to_stress_rv24h: float = 1.26     # ETH p90 RV; high stress
    elevated_to_stress_rv1h: float = 1.90      # acute vol spike
    elevated_to_stress_trend: float = 0.65
    elevated_to_stress_iv30d: float = 1.40     # IV pricing tail risk
    elevated_to_stress_gas: float = 3.0

    # STRESS → ELEVATED (ALL must revert, sustained ≥ hysteresis_stress_elevated_h hours)
    stress_to_elevated_rv24h: float = 1.00     # below ETH p90
    stress_to_elevated_rv1h: float = 1.30      # acute spike resolved
    stress_to_elevated_trend: float = 0.50
    stress_to_elevated_iv30d: float = 1.20     # IV cooling
    stress_to_elevated_gas: float = 2.0
    hysteresis_stress_elevated_h: int = 8

    # STRESS / ELEVATED → GAP (ANY trigger, uses raw unsmoothed signals)
    to_gap_rv1h_raw: float = 3.00              # ≈5.5% hourly H/L; genuine flash event
    to_gap_iv30d_raw: float = 1.75
    to_gap_price_move_1h: float = 0.10         # |1h return| ≥ 10%
    gap_min_hold_h: int = 4


KNOWN_EVENTS: list[dict] = [
    {"name": "May 2021 crash",         "date": "2021-05-19", "expect_elevated_before_h": 12},
    {"name": "Sep 2021 China ban",     "date": "2021-09-07", "expect_elevated_before_h": 4},
    {"name": "Nov 2021 peak",          "date": "2021-11-10", "expect_elevated_before_h": 6},
    {"name": "LUNA collapse",          "date": "2022-05-12", "expect_elevated_before_h": 24},
    {"name": "FTX collapse",           "date": "2022-11-08", "expect_elevated_before_h": 6},
    {"name": "2023 banking crisis",    "date": "2023-03-10", "expect_elevated_before_h": 4},
    {"name": "2024 Q1 bull run",       "date": "2024-03-12", "expect_elevated_before_h": None},  # no crash expected
    {"name": "Aug 2024 Japan unwind",  "date": "2024-08-05", "expect_elevated_before_h": 4},
]


# ---------------------------------------------------------------------------
# Data Fetching
# ---------------------------------------------------------------------------

def fetch_binance_ohlcv(
    symbol: str = "ETHUSDT",
    interval: str = "1h",
    start_dt: datetime = datetime(2021, 1, 1, tzinfo=timezone.utc),
    end_dt: datetime = datetime(2024, 12, 31, 23, tzinfo=timezone.utc),
) -> pd.DataFrame:
    """
    Fetch hourly OHLCV from Binance public API.
    Paginates automatically; respects rate limits with 0.2s sleeps.
    """
    url = "https://api.binance.us/api/v3/klines"
    all_rows = []

    current = start_dt
    limit = 1000  # Binance max per request

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
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"  → {len(df):,} bars fetched ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
    return df


def fetch_deribit_dvol(
    start_dt: datetime = datetime(2021, 1, 1, tzinfo=timezone.utc),
    end_dt: datetime = datetime(2024, 12, 31, 23, tzinfo=timezone.utc),
) -> pd.Series:
    """
    Fetch Deribit ETH DVOL index (30D implied vol, annualized %) hourly.
    Returns a Series indexed by UTC timestamp, values in decimal (e.g., 0.75 = 75%).

    Uses get_volatility_index_data (ETH DVOL available from ~Apr 2021).
    Response format: [[timestamp_ms, open, high, low, close], ...]
    """
    # DVOL data only available from April 2021
    dvol_start = datetime(2021, 4, 1, tzinfo=timezone.utc)
    effective_start = max(start_dt, dvol_start)

    url = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
    all_rows = []
    window = timedelta(days=10)  # conservative window to avoid rate limits

    print(f"Fetching Deribit DVOL from {effective_start.date()} to {end_dt.date()}...")
    current = effective_start
    while current < end_dt:
        batch_end = min(current + window, end_dt)
        params = {
            "currency": "ETH",
            "resolution": "3600",  # 1 hour in seconds
            "start_timestamp": int(current.timestamp() * 1000),
            "end_timestamp": int(batch_end.timestamp() * 1000),
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json().get("result", {})
            rows = data.get("data", [])  # [[ts_ms, open, high, low, close], ...]
            for row in rows:
                all_rows.append({
                    "timestamp": pd.Timestamp(row[0], unit="ms", tz="UTC"),
                    "dvol": float(row[4]) / 100.0,  # close price, convert % → decimal
                })
        except Exception as exc:
            print(f"  WARNING: DVOL fetch failed for {current.date()}–{batch_end.date()} ({exc}); skipping window")
        current = batch_end + timedelta(hours=1)
        time.sleep(0.5)

    if not all_rows:
        raise ValueError("No DVOL data returned from Deribit")

    dvol_df = pd.DataFrame(all_rows).sort_values("timestamp").drop_duplicates("timestamp")
    dvol_series = dvol_df.set_index("timestamp")["dvol"]
    print(f"  → {len(dvol_series):,} DVOL bars ({dvol_series.index.min().date()} to {dvol_series.index.max().date()})")
    return dvol_series


def build_synthetic_gas_score(rv_24h: pd.Series) -> pd.Series:
    """
    Synthetic gas_score when Etherscan data is unavailable.
    Approximates the historical correlation between gas spikes and vol events.
    """
    z = (rv_24h - 0.60) / 0.20
    sigmoid = 1.0 / (1.0 + np.exp(-z))
    return 1.0 + 2.0 * sigmoid


# ---------------------------------------------------------------------------
# Signal Computation
# ---------------------------------------------------------------------------

def compute_parkinson_rv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Parkinson RV estimator from OHLC data.
    Returns df with columns: parkinson_var, rv_1h, rv_24h, rv_7d, rv_ratio.
    """
    ln_hl = np.log(df["high"] / df["low"])
    parkinson_var = (ln_hl ** 2) / (4 * np.log(2))

    df = df.copy()
    df["parkinson_var"] = parkinson_var

    # RV windows (annualize: × 365×24 since each bar = 1 hour)
    ann_factor = 365 * 24
    df["rv_1h"] = np.sqrt(parkinson_var * ann_factor)
    df["rv_24h"] = np.sqrt(parkinson_var.rolling(24, min_periods=12).mean() * ann_factor)
    df["rv_7d"] = np.sqrt(parkinson_var.rolling(168, min_periods=84).mean() * ann_factor)

    df["rv_ratio"] = (df["rv_1h"] / df["rv_24h"]).clip(0, 10)
    return df


def compute_trend_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite trend score in [-1, +1] from EMA crossover and price vs MA.
    """
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


def apply_smoothing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply exponential smoothing with time constants from regime_states.md §7.2.
    Adds smoothed columns: rv_1h_s, rv_24h_s, iv_30d_s, trend_s, gas_score_s.
    dt = 1 hour.
    """
    df = df.copy()

    def ema_tau(series: pd.Series, tau_hours: float) -> pd.Series:
        alpha = 1 - np.exp(-1.0 / tau_hours)
        return series.ewm(alpha=alpha, adjust=False).mean()

    df["rv_1h_s"] = ema_tau(df["rv_1h"], tau_hours=0.25)
    df["rv_24h_s"] = ema_tau(df["rv_24h"], tau_hours=2.0)
    df["iv_30d_s"] = ema_tau(df["iv_30d"], tau_hours=0.5)
    df["trend_s"] = ema_tau(df["trend_score"], tau_hours=1.0)
    df["gas_score_s"] = ema_tau(df["gas_score"], tau_hours=0.167)
    return df


# ---------------------------------------------------------------------------
# State Machine
# ---------------------------------------------------------------------------

def run_state_machine(
    df: pd.DataFrame,
    thresholds: RegimeThresholds,
    burnin_bars: int = 7 * 24,
) -> pd.DataFrame:
    """
    Apply regime state machine with full hysteresis logic.
    Returns df with columns: regime_state, in_normalization.

    Parameters
    ----------
    df : DataFrame with smoothed signal columns (rv_1h_s, rv_24h_s, rv_1h [raw],
         iv_30d_s, iv_30d [raw], trend_s, gas_score_s) plus 'close' for
         1h return computation.
    thresholds : RegimeThresholds dataclass.
    burnin_bars : Number of initial bars to hold as CALM for signal warm-up.
    """
    n = len(df)
    states = ["CALM"] * n
    in_norm = [False] * n
    triggers = [""] * n  # signal(s) that fired at each transition bar

    state = "CALM"
    normalization_start_idx: Optional[int] = None  # for downward hysteresis timer

    # Pre-compute 1h returns for GAP detection
    returns_1h = df["close"].pct_change().abs().fillna(0).values

    rv_1h_s = df["rv_1h_s"].values
    rv_24h_s = df["rv_24h_s"].values
    rv_ratio_s = df["rv_ratio"].ewm(alpha=1 - np.exp(-1.0 / 0.5), adjust=False).mean().values
    iv_30d_s = df["iv_30d_s"].values
    trend_s = df["trend_s"].values
    gas_s = df["gas_score_s"].values
    rv_1h_raw = df["rv_1h"].values
    iv_30d_raw = df["iv_30d"].values

    t = thresholds

    for i in range(n):
        if i < burnin_bars:
            states[i] = "CALM"
            continue

        _state_at_start = state  # snapshot for hysteresis trigger detection at end

        # ---- GAP/CRASH entry (immediate, raw signals) ----
        if state != "GAP":
            _gap = []
            if rv_1h_raw[i] > t.to_gap_rv1h_raw: _gap.append("rv_1h_raw")
            if iv_30d_raw[i] > t.to_gap_iv30d_raw: _gap.append("iv_30d_raw")
            if returns_1h[i] > t.to_gap_price_move_1h: _gap.append("price_1h")
            if _gap:
                state = "GAP"
                normalization_start_idx = None
                states[i] = state
                triggers[i] = "+".join(_gap)
                in_norm[i] = False
                continue

        # ---- GAP → STRESS (auto after min hold; in live this is manual) ----
        if state == "GAP":
            # find gap entry index
            gap_start = _last_transition_to(states, "GAP", i)
            if gap_start is not None and (i - gap_start) >= t.gap_min_hold_h:
                state = "STRESS"
                normalization_start_idx = None
                triggers[i] = "gap_hold"
            states[i] = state
            in_norm[i] = False
            continue

        # ---- STRESS → GAP (upward from STRESS) ----
        if state == "STRESS":
            _gap = []
            if rv_1h_raw[i] > t.to_gap_rv1h_raw: _gap.append("rv_1h_raw")
            if iv_30d_raw[i] > t.to_gap_iv30d_raw: _gap.append("iv_30d_raw")
            if returns_1h[i] > t.to_gap_price_move_1h: _gap.append("price_1h")
            if _gap:
                state = "GAP"
                normalization_start_idx = None
                states[i] = state
                triggers[i] = "+".join(_gap)
                in_norm[i] = False
                continue

        # ---- ELEVATED → STRESS (upward from ELEVATED) ----
        if state == "ELEVATED":
            _stress = []
            if rv_24h_s[i] > t.elevated_to_stress_rv24h: _stress.append("rv_24h")
            if rv_1h_s[i] > t.elevated_to_stress_rv1h: _stress.append("rv_1h")
            if abs(trend_s[i]) > t.elevated_to_stress_trend: _stress.append("trend")
            if iv_30d_s[i] > t.elevated_to_stress_iv30d: _stress.append("iv_30d")
            if gas_s[i] > t.elevated_to_stress_gas: _stress.append("gas")
            if _stress:
                state = "STRESS"
                normalization_start_idx = None
                states[i] = state
                triggers[i] = "+".join(_stress)
                in_norm[i] = False
                continue

        # ---- CALM → ELEVATED (upward from CALM) ----
        if state == "CALM":
            _elev = []
            if rv_24h_s[i] > t.calm_to_elevated_rv24h: _elev.append("rv_24h")
            if rv_ratio_s[i] > t.calm_to_elevated_rv_ratio: _elev.append("rv_ratio")
            if abs(trend_s[i]) > t.calm_to_elevated_trend: _elev.append("trend")
            if iv_30d_s[i] > t.calm_to_elevated_iv30d: _elev.append("iv_30d")
            if _elev:
                state = "ELEVATED"
                normalization_start_idx = None
                states[i] = state
                triggers[i] = "+".join(_elev)
                in_norm[i] = False
                continue

        # ---- Downward transitions with hysteresis ----

        if state == "STRESS":
            all_reverted = (
                rv_24h_s[i] < t.stress_to_elevated_rv24h
                and rv_1h_s[i] < t.stress_to_elevated_rv1h
                and abs(trend_s[i]) < t.stress_to_elevated_trend
                and iv_30d_s[i] < t.stress_to_elevated_iv30d
                and gas_s[i] < t.stress_to_elevated_gas
            )
            if all_reverted:
                if normalization_start_idx is None:
                    normalization_start_idx = i
                elif (i - normalization_start_idx) >= t.hysteresis_stress_elevated_h:
                    state = "ELEVATED"
                    normalization_start_idx = None
            else:
                normalization_start_idx = None  # reset if any signal re-elevates

        elif state == "ELEVATED":
            all_reverted = (
                rv_24h_s[i] < t.elevated_to_calm_rv24h
                and rv_ratio_s[i] < t.elevated_to_calm_rv_ratio
                and abs(trend_s[i]) < t.elevated_to_calm_trend
                and iv_30d_s[i] < t.elevated_to_calm_iv30d
            )
            if all_reverted:
                if normalization_start_idx is None:
                    normalization_start_idx = i
                elif (i - normalization_start_idx) >= t.hysteresis_elevated_calm_h:
                    state = "CALM"
                    normalization_start_idx = None
            else:
                normalization_start_idx = None

        if state != _state_at_start and not triggers[i]:
            triggers[i] = "hysteresis"
        states[i] = state
        in_norm[i] = normalization_start_idx is not None

    df = df.copy()
    df["regime_state"] = states
    df["in_normalization"] = in_norm
    df["transition_trigger"] = triggers
    return df


def _last_transition_to(states: list[str], target: str, current_idx: int) -> Optional[int]:
    """Return the index of the most recent entry into target state."""
    for i in range(current_idx - 1, -1, -1):
        if states[i] == target and (i == 0 or states[i - 1] != target):
            return i
    return None


# ---------------------------------------------------------------------------
# Analysis and Reporting
# ---------------------------------------------------------------------------

def compute_state_distribution(df: pd.DataFrame) -> dict:
    """Compute % of hours in each regime state (excluding burn-in)."""
    counts = df["regime_state"].value_counts()
    total = len(df)
    targets = {"CALM": (0.50, 0.65), "ELEVATED": (0.22, 0.38), "STRESS": (0.08, 0.18), "GAP": (0.01, 0.05)}
    result = {}
    for state in ["CALM", "ELEVATED", "STRESS", "GAP"]:
        pct = counts.get(state, 0) / total
        lo, hi = targets[state]
        result[state] = {
            "pct": round(pct * 100, 1),
            "hours": int(counts.get(state, 0)),
            "target": f"{lo*100:.0f}–{hi*100:.0f}%",
            "in_range": lo <= pct <= hi,
        }
    return result


def compute_transition_stats(df: pd.DataFrame) -> dict:
    """Count state transitions and compute annualized rates."""
    states = df["regime_state"].values
    transitions: dict[str, int] = {}
    for i in range(1, len(states)):
        if states[i] != states[i - 1]:
            key = f"{states[i-1]}→{states[i]}"
            transitions[key] = transitions.get(key, 0) + 1

    # Annualize: total hours / 8760
    years = len(df) / 8760
    return {k: {"count": v, "per_year": round(v / years, 1)} for k, v in transitions.items()}


def compute_event_detection(df: pd.DataFrame) -> list[dict]:
    """
    For each known event, report the regime state at the event peak and
    how many hours before the peak the machine first reached ELEVATED+.
    """
    results = []
    df = df.set_index("timestamp") if "timestamp" in df.columns else df

    for event in KNOWN_EVENTS:
        event_dt = pd.Timestamp(event["date"], tz="UTC") + pd.Timedelta(hours=12)
        window_start = event_dt - pd.Timedelta(hours=72)

        try:
            window = df.loc[window_start:event_dt]
        except Exception:
            results.append({"event": event["name"], "error": "date out of range"})
            continue

        if len(window) == 0:
            results.append({"event": event["name"], "error": "no data in window"})
            continue

        state_at_peak = df.loc[event_dt:event_dt + pd.Timedelta(hours=1)]["regime_state"].iloc[0] if event_dt in df.index or True else "UNKNOWN"
        # Find first ELEVATED or above in window
        elevated_times = window[window["regime_state"].isin(["ELEVATED", "STRESS", "GAP"])].index
        first_elevated_time = elevated_times[0] if len(elevated_times) > 0 else None
        lead_hours = ((event_dt - first_elevated_time).total_seconds() / 3600) if first_elevated_time else None

        required = event.get("expect_elevated_before_h")
        if required is None:
            detected_ok = True  # No crash expected; just report
        elif lead_hours is not None and lead_hours >= required:
            detected_ok = True
        else:
            detected_ok = False

        try:
            state_at_peak = df["regime_state"].asof(event_dt)
        except Exception:
            state_at_peak = "N/A"

        results.append({
            "event": event["name"],
            "date": event["date"],
            "state_at_peak": state_at_peak,
            "first_elevated_hours_before": round(lead_hours, 1) if lead_hours is not None else None,
            "required_lead_h": required,
            "detected_ok": detected_ok,
        })
    return results


def compute_avg_state_duration(df: pd.DataFrame) -> dict:
    """Compute average and median duration of each state in hours."""
    durations: dict[str, list[int]] = {"CALM": [], "ELEVATED": [], "STRESS": [], "GAP": []}
    states = df["regime_state"].values
    current_state = states[0]
    run_len = 1
    for i in range(1, len(states)):
        if states[i] == current_state:
            run_len += 1
        else:
            if current_state in durations:
                durations[current_state].append(run_len)
            current_state = states[i]
            run_len = 1
    if current_state in durations:
        durations[current_state].append(run_len)

    result = {}
    for state, runs in durations.items():
        if runs:
            result[state] = {
                "mean_h": round(float(np.mean(runs)), 1),
                "median_h": round(float(np.median(runs)), 1),
                "n_episodes": len(runs),
            }
    return result


def compute_signal_attribution(df: pd.DataFrame) -> dict:
    """
    For each transition type, count how often each signal was among the triggers.
    A single transition with trigger "rv_24h+rv_1h" increments both counters.
    Returns: {transition_key: {"n_events": int, "signals": {signal: count}}}
    """
    from collections import defaultdict
    states = df["regime_state"].values
    trigs = df["transition_trigger"].values
    n_events: dict[str, int] = defaultdict(int)
    sig_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i in range(1, len(states)):
        if states[i] != states[i - 1] and trigs[i]:
            key = f"{states[i-1]}→{states[i]}"
            n_events[key] += 1
            for sig in trigs[i].split("+"):
                if sig:
                    sig_counts[key][sig] += 1
    return {
        k: {
            "n_events": n_events[k],
            "signals": dict(sorted(sig_counts[k].items(), key=lambda x: -x[1])),
        }
        for k in sorted(n_events.keys())
    }


def print_signal_attribution(attribution: dict) -> None:
    print("\n--- Signal Attribution (which signal(s) drove each transition) ---")
    for trans, info in attribution.items():
        n = info["n_events"]
        sigs = info["signals"]
        print(f"  {trans}  ({n} events):")
        for sig, count in sigs.items():
            pct = 100.0 * count / n
            bar = "█" * int(pct / 5 + 0.5)
            print(f"    {sig:<18} {count:>4}/{n:<4}  {pct:5.1f}%  {bar}")
    print()


def print_report(dist: dict, transitions: dict, events: list[dict], durations: dict) -> None:
    print("\n" + "=" * 65)
    print("REGIME BACKTEST REPORT")
    print("=" * 65)

    print("\n--- State Distribution ---")
    print(f"{'State':<12} {'Hours':>7} {'%':>7} {'Target':>12} {'OK?':>5}")
    for state in ["CALM", "ELEVATED", "STRESS", "GAP"]:
        d = dist.get(state, {})
        ok_str = "✓" if d.get("in_range") else "✗"
        print(f"{state:<12} {d.get('hours',0):>7,} {d.get('pct',0):>6.1f}% {d.get('target',''):>12} {ok_str:>5}")

    print("\n--- Average Episode Duration ---")
    for state in ["CALM", "ELEVATED", "STRESS", "GAP"]:
        d = durations.get(state, {})
        print(f"  {state:<12} mean={d.get('mean_h','N/A'):>6}h  median={d.get('median_h','N/A'):>6}h  episodes={d.get('n_episodes','N/A')}")

    print("\n--- Transition Counts ---")
    for k, v in sorted(transitions.items()):
        print(f"  {k:<28} {v['count']:>4} total  ({v['per_year']:>5.1f}/year)")

    print("\n--- Event Detection ---")
    passed = sum(1 for e in events if e.get("detected_ok") and e.get("required_lead_h") is not None)
    testable = sum(1 for e in events if e.get("required_lead_h") is not None)
    for e in events:
        label = e.get("event", e.get("name", "?"))
        if "error" in e:
            print(f"  {label:<30} ERROR: {e['error']}")
            continue
        lead = e.get("first_elevated_hours_before")
        lead_str = f"{lead:.1f}h before" if lead is not None else "not detected"
        ok_str = "✓" if e.get("detected_ok") else "✗"
        print(f"  {ok_str} {label:<30} state={e.get('state_at_peak','?'):<12} elevated={lead_str}")
    print(f"\n  Result: {passed}/{testable} events detected with sufficient lead time")

    print("\n" + "=" * 65)


# ---------------------------------------------------------------------------
# VRP Distribution by Regime (optional output for alpha_overlay_signals.md)
# ---------------------------------------------------------------------------

def compute_vrp_by_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary of VRP (IV_30d − RV_24h) by regime state.
    """
    df = df.copy()
    df["vrp"] = (df["iv_30d"] - df["rv_24h"]) * 100  # in vol points
    summary = df.groupby("regime_state")["vrp"].agg(
        mean="mean", median="median", p25=lambda x: x.quantile(0.25),
        p75=lambda x: x.quantile(0.75), p5=lambda x: x.quantile(0.05),
        p95=lambda x: x.quantile(0.95),
    ).round(1)
    return summary


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regime signal backtest for Crypto VolAlpha")
    p.add_argument("--start", default="2021-01-01", help="Start date (YYYY-MM-DD)")
    p.add_argument("--end",   default="2024-12-31", help="End date (YYYY-MM-DD)")

    # Threshold overrides — entry
    p.add_argument("--rv-threshold-elevated", type=float, default=None,
                   help="Override CALM→ELEVATED RV_24h entry threshold (default: 0.55)")
    p.add_argument("--iv-threshold-elevated", type=float, default=None,
                   help="Override CALM→ELEVATED IV_30d entry threshold (default: 0.70)")
    p.add_argument("--rv-threshold-stress", type=float, default=None,
                   help="Override ELEVATED→STRESS RV_24h entry threshold (default: 0.90)")
    p.add_argument("--rv1h-threshold-stress", type=float, default=None,
                   help="Override ELEVATED→STRESS RV_1h entry threshold (default: 1.20)")
    # Threshold overrides — exit
    p.add_argument("--rv-exit-elevated", type=float, default=None,
                   help="Override ELEVATED→CALM RV_24h exit threshold (default: 0.45)")
    p.add_argument("--iv-exit-elevated", type=float, default=None,
                   help="Override ELEVATED→CALM IV_30d exit threshold (default: 0.60)")
    p.add_argument("--rv-exit-stress", type=float, default=None,
                   help="Override STRESS→ELEVATED RV_24h exit threshold (default: 0.60)")
    # GAP overrides
    p.add_argument("--gap-rv1h", type=float, default=None,
                   help="Override GAP entry RV_1h threshold (default: 2.00 = 200%% ann)")
    p.add_argument("--gap-price-move", type=float, default=None,
                   help="Override GAP entry |1h return| threshold (default: 0.10 = 10%%)")
    # Hysteresis overrides
    p.add_argument("--hysteresis-ec", type=int, default=None,
                   help="Override ELEVATED→CALM hysteresis hours (default: 4)")
    p.add_argument("--hysteresis-se", type=int, default=None,
                   help="Override STRESS→ELEVATED hysteresis hours (default: 8)")

    p.add_argument("--output", default=None,
                   help="Save signal DataFrame to CSV at this path")
    p.add_argument("--no-dvol", action="store_true",
                   help="Skip Deribit DVOL fetch; use RV_24h * 1.20 as IV proxy")
    p.add_argument("--vrp-analysis", action="store_true",
                   help="Print VRP distribution by regime (supplement to alpha_overlay_signals.md)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt   = datetime.fromisoformat(args.end).replace(hour=23, tzinfo=timezone.utc)

    # 1. Build thresholds (apply any CLI overrides)
    thresholds = RegimeThresholds()
    # Entry overrides
    if args.rv_threshold_elevated is not None:
        thresholds.calm_to_elevated_rv24h = args.rv_threshold_elevated
    if args.iv_threshold_elevated is not None:
        thresholds.calm_to_elevated_iv30d = args.iv_threshold_elevated
    if args.rv_threshold_stress is not None:
        thresholds.elevated_to_stress_rv24h = args.rv_threshold_stress
    if args.rv1h_threshold_stress is not None:
        thresholds.elevated_to_stress_rv1h = args.rv1h_threshold_stress
    # Exit overrides
    if args.rv_exit_elevated is not None:
        thresholds.elevated_to_calm_rv24h = args.rv_exit_elevated
    if args.iv_exit_elevated is not None:
        thresholds.elevated_to_calm_iv30d = args.iv_exit_elevated
    if args.rv_exit_stress is not None:
        thresholds.stress_to_elevated_rv24h = args.rv_exit_stress
    # GAP overrides
    if args.gap_rv1h is not None:
        thresholds.to_gap_rv1h_raw = args.gap_rv1h
    if args.gap_price_move is not None:
        thresholds.to_gap_price_move_1h = args.gap_price_move
    # Hysteresis overrides
    if args.hysteresis_ec is not None:
        thresholds.hysteresis_elevated_calm_h = args.hysteresis_ec
    if args.hysteresis_se is not None:
        thresholds.hysteresis_stress_elevated_h = args.hysteresis_se

    # 2. Fetch price data
    price_df = fetch_binance_ohlcv(start_dt=start_dt, end_dt=end_dt)

    # 3. Compute RV signals
    price_df = compute_parkinson_rv(price_df)
    price_df = compute_trend_score(price_df)

    # 4. Fetch / approximate IV
    if args.no_dvol:
        print("Using IV proxy: RV_24h × 1.20 (DVOL fetch skipped)")
        price_df["iv_30d"] = price_df["rv_24h"] * 1.20
        iv_source = "proxy"
    else:
        try:
            dvol = fetch_deribit_dvol(start_dt=start_dt, end_dt=end_dt)
            price_df = price_df.set_index("timestamp")
            price_df["iv_30d"] = dvol.reindex(price_df.index).ffill().bfill()
            price_df = price_df.reset_index()
            iv_source = "deribit_dvol"
        except Exception as e:
            print(f"  WARNING: DVOL fetch failed ({e}); falling back to RV proxy")
            price_df["iv_30d"] = price_df["rv_24h"] * 1.20
            iv_source = "proxy_fallback"

    # 5. Gas score (synthetic)
    price_df["gas_score"] = build_synthetic_gas_score(price_df["rv_24h"])
    print(f"Gas signal: synthetic (no Etherscan API key used)")
    print(f"IV source: {iv_source}")

    # 6. Apply smoothing
    price_df = apply_smoothing(price_df)

    # 7. Run state machine
    burnin = 7 * 24  # 7-day burn-in
    print(f"\nRunning state machine (burn-in: {burnin} bars)...")
    result_df = run_state_machine(price_df, thresholds, burnin_bars=burnin)
    result_df_valid = result_df.iloc[burnin:].copy()

    # 8. Compute statistics
    dist = compute_state_distribution(result_df_valid)
    transitions = compute_transition_stats(result_df_valid)
    events = compute_event_detection(result_df_valid)
    durations = compute_avg_state_duration(result_df_valid)
    attribution = compute_signal_attribution(result_df_valid)

    # 9. Print report
    print_report(dist, transitions, events, durations)
    print_signal_attribution(attribution)

    # 10. Optional: VRP by regime
    if args.vrp_analysis:
        print("\n--- VRP (vol points) by Regime State ---")
        print("(Supplement for alpha_overlay_signals.md §4.1)")
        vrp_table = compute_vrp_by_regime(result_df_valid)
        print(vrp_table.to_string())
        print()

    # 11. Optional: save to CSV
    if args.output:
        cols = ["timestamp", "close", "rv_1h", "rv_24h", "rv_ratio",
                "iv_30d", "trend_score", "gas_score", "regime_state", "in_normalization"]
        result_df[cols].to_csv(args.output, index=False)
        print(f"\nSignal DataFrame saved to: {args.output}")

    # 12. Check if calibration passed
    all_in_range = all(d.get("in_range", False) for d in dist.values())
    testable_events = [e for e in events if e.get("required_lead_h") is not None]
    events_passed = sum(1 for e in testable_events if e.get("detected_ok"))
    events_total = len(testable_events)

    print(f"\nCalibration status:")
    print(f"  Distribution: {'PASS' if all_in_range else 'FAIL — adjust thresholds per docs/regime_backtest.md §6'}")
    print(f"  Event detection: {events_passed}/{events_total} {'PASS' if events_passed >= events_total * 0.75 else 'FAIL'}")


if __name__ == "__main__":
    main()
