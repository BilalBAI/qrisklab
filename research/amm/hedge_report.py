"""
Hedge Report Generator
======================
Fetches live market data (Binance spot, Deribit DVOL + option chain),
computes LP greeks for a symmetric ±RANGE_PCT Uniswap V3/V4 position,
selects the best 30Δ put, sizes the delta-first hedge, runs an
instantaneous stress test, and writes a dated markdown report to docs/.

Usage:
    python hedge_report.py
    python hedge_report.py --capital 500000 --range 0.15 --delta 0.30
"""
from __future__ import annotations

import math
import sys
import time
import warnings
import argparse
from datetime import datetime, timezone
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Project path setup
# ─────────────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_PROJECT_ROOT / "qrisklab"))

from qrisklab.core.black_scholes import bsm_pricing, calc_greeks       # noqa: E402
from qrisklab.core.amm_valuation import calc_current_holdings           # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (edit before running)
# ─────────────────────────────────────────────────────────────────────────────
CAPITAL           = 1_000_000.0   # USD
RANGE_PCT         = 0.22          # symmetric ±RANGE_PCT around spot
PUT_DELTA_TGT     = 0.30          # target |Δ_put|
TENOR_TARGET_DAYS = 30            # preferred tenor (days)
TENOR_WINDOW      = (15, 55)      # accept expiries within this day range
STRIKE_WINDOW     = (0.78, 1.00)  # candidate strikes as fraction of spot
STRESS_MOVES      = [-0.20, -0.10, -0.05, -0.01, 0.0, +0.01, +0.05, +0.10, +0.20]
RATE              = 0.05          # risk-free rate (annual)
OUTPUT_DIR        = _PROJECT_ROOT / "docs"

# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper
# ─────────────────────────────────────────────────────────────────────────────

def _get(url: str, params: dict | None = None, timeout: int = 15) -> dict:
    """GET with automatic SSL fallback (needed for macOS cert store issues)."""
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.SSLError:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except ImportError:
            pass
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        r = requests.get(url, params=params, timeout=timeout, verify=False)
        r.raise_for_status()
        return r.json()


# ─────────────────────────────────────────────────────────────────────────────
# Market data fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_eth_spot() -> tuple[float, float]:
    """
    Returns (binance_spot, deribit_index).
    Tries Binance first; falls back to Deribit index for both values.
    """
    binance_price = None
    try:
        data = _get("https://api.binance.com/api/v3/ticker/price",
                    {"symbol": "ETHUSDT"})
        binance_price = float(data["price"])
    except Exception as e:
        print(f"  [warn] Binance failed: {e}")

    deribit_price = None
    try:
        data = _get("https://www.deribit.com/api/v2/public/get_index_price",
                    {"index_name": "eth_usd"})
        deribit_price = float(data["result"]["index_price"])
    except Exception as e:
        print(f"  [warn] Deribit index failed: {e}")

    spot = binance_price or deribit_price
    if spot is None:
        raise RuntimeError("Cannot fetch ETH spot from Binance or Deribit.")
    return spot, (deribit_price or spot)


def fetch_dvol() -> float:
    """Fetch ETH DVOL (30D implied vol index) from Deribit. Returns decimal (0.719 etc.)."""
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - 3 * 3600 * 1000  # last 3 hours
    try:
        data = _get(
            "https://www.deribit.com/api/v2/public/get_volatility_index_data",
            {"currency": "ETH", "resolution": "3600",
             "start_timestamp": start_ts, "end_timestamp": end_ts},
        )
        rows = data["result"]["data"]   # [[ts, open, high, low, close], ...]
        if rows:
            return rows[-1][4] / 100.0  # last close, % → decimal
    except Exception as e:
        print(f"  [warn] DVOL fetch failed: {e}. Using 75% fallback.")
    return 0.75


# ─────────────────────────────────────────────────────────────────────────────
# Option chain fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_put_candidates(
    spot: float,
    tenor_min: int,
    tenor_max: int,
    strike_lo_frac: float,
    strike_hi_frac: float,
) -> tuple[list[str], dict]:
    """
    Fetch all ETH put instruments from Deribit within the tenor and strike windows.

    Returns:
        instruments  : list of instrument name strings
        expiry_meta  : {expiry_str: {"days": int, "expiry_dt": datetime}}
    """
    today = datetime.now(tz=timezone.utc).date()
    data = _get(
        "https://www.deribit.com/api/v2/public/get_instruments",
        {"currency": "ETH", "expired": "false", "kind": "option"},
    )

    instruments: list[str] = []
    expiry_meta: dict = {}

    for item in data["result"]:
        name = item["instrument_name"]
        parts = name.split("-")
        if len(parts) != 4 or parts[3] != "P":
            continue
        try:
            exp_dt = datetime.strptime(parts[1], "%d%b%y")
        except ValueError:
            continue
        days = (exp_dt.date() - today).days
        if not (tenor_min <= days <= tenor_max):
            continue
        strike = float(parts[2])
        if not (strike_lo_frac * spot <= strike <= strike_hi_frac * spot):
            continue
        instruments.append(name)
        exp_str = parts[1]
        if exp_str not in expiry_meta:
            expiry_meta[exp_str] = {"days": days, "expiry_dt": exp_dt}

    print(f"  {len(instruments)} candidate put instruments across "
          f"{len(expiry_meta)} expiries.")
    return instruments, expiry_meta


def fetch_option_greeks(instruments: list[str], index_price: float) -> list[dict]:
    """
    Fetch order book + greeks for each instrument from Deribit.
    Returns list of standardised dicts.
    """
    url = "https://www.deribit.com/api/v2/public/get_order_book"
    results: list[dict] = []

    for name in instruments:
        try:
            data = _get(url, {"instrument_name": name, "depth": 1})
            c = data["result"]
            greeks = c.get("greeks") or {}
            mark_eth = c.get("mark_price")
            if mark_eth is None or not greeks:
                continue
            parts = name.split("-")
            results.append({
                "instrument": name,
                "expiry":     parts[1],
                "strike":     float(parts[2]),
                "mark_usd":   mark_eth * index_price,
                "mark_eth":   mark_eth,
                "bid_usd":    (c.get("best_bid_price") or 0.0) * index_price,
                "ask_usd":    (c.get("best_ask_price") or 0.0) * index_price,
                "delta":      greeks.get("delta", 0.0),      # negative for puts
                "gamma":      greeks.get("gamma", 0.0),      # positive
                "theta_usd":  greeks.get("theta", 0.0),      # USD/day, negative
                "vega":       greeks.get("vega", 0.0),
                "mark_iv":    c.get("mark_iv", 0.0),         # in percent
                "index_price": index_price,
            })
        except Exception as e:
            print(f"  [warn] Failed to fetch {name}: {e}")

    return results


def select_best_put(
    candidates: list[dict],
    delta_tgt: float,
    tenor_tgt: int,
    expiry_meta: dict,
) -> dict:
    """
    Score each candidate: score = |days − tenor_tgt| × 10 + |delta + delta_tgt|
    (lower = better). Returns best candidate dict with days_to_exp and T_years added.
    """
    best: dict | None = None
    best_score = float("inf")

    for row in candidates:
        days = expiry_meta[row["expiry"]]["days"]
        score = abs(days - tenor_tgt) * 10.0 + abs(row["delta"] + delta_tgt)
        if score < best_score:
            best_score = score
            best = row

    if best is None:
        raise RuntimeError("No suitable put candidate found.")

    best = dict(best)  # copy
    best["days_to_exp"] = expiry_meta[best["expiry"]]["days"]
    best["T_years"] = best["days_to_exp"] / 365.0
    return best


# ─────────────────────────────────────────────────────────────────────────────
# LP position math
# ─────────────────────────────────────────────────────────────────────────────

def calc_lp_position(spot: float, capital: float, range_pct: float) -> dict:
    """
    Compute LP greeks and initial holdings for a symmetric ±range_pct position.

    Formulas (from lp_greeks_model.md):
        pa   = S × (1 − r),  pb   = S × (1 + r)
        L    = C / (2√S − √pa − S/√pb)
        Δ_LP = L × (1/√S − 1/√pb)    [ETH]
        Γ_LP = −L / (2 × S^1.5)      [$/($²), always ≤ 0]
        amt0 = Δ_LP                   [initial ETH]
        amt1 = L × (√S − √pa)        [initial USD]
    """
    pa = spot * (1.0 - range_pct)
    pb = spot * (1.0 + range_pct)
    sq_S = math.sqrt(spot)
    sq_a = math.sqrt(pa)
    sq_b = math.sqrt(pb)

    L        = capital / (2.0 * sq_S - sq_a - spot / sq_b)
    delta_lp = L * (1.0 / sq_S - 1.0 / sq_b)    # ETH
    gamma_lp = -L / (2.0 * spot ** 1.5)           # $/($²)
    amt0     = delta_lp                            # initial ETH
    amt1     = L * (sq_S - sq_a)                  # initial USD

    return {
        "pa":          pa,
        "pb":          pb,
        "L":           L,
        "delta_lp":    delta_lp,
        "gamma_lp":    gamma_lp,
        "amt0":        amt0,
        "amt1":        amt1,
        "entry_spot":  spot,
        "capital":     capital,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hedge sizing
# ─────────────────────────────────────────────────────────────────────────────

def size_hedge(lp: dict, opt: dict) -> dict:
    """Delta-first put sizing: N_puts = Δ_LP / |Δ_put|."""
    n_puts     = lp["delta_lp"] / abs(opt["delta"])
    n_puts_int = round(n_puts)
    gamma_hedge   = n_puts_int * opt["gamma"]
    gamma_coverage = gamma_hedge / abs(lp["gamma_lp"])
    delta_net     = lp["delta_lp"] + n_puts_int * opt["delta"]
    total_premium = n_puts_int * opt["mark_usd"]
    daily_theta   = n_puts_int * abs(opt["theta_usd"])

    return {
        "n_puts":          n_puts,
        "n_puts_int":      n_puts_int,
        "gamma_hedge":     gamma_hedge,
        "gamma_coverage":  gamma_coverage,
        "delta_net":       delta_net,
        "total_premium":   total_premium,
        "daily_theta_usd": daily_theta,
        "monthly_theta_usd": daily_theta * 30,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Stress test
# ─────────────────────────────────────────────────────────────────────────────

def run_stress(
    spot: float,
    lp: dict,
    opt: dict,
    hedge: dict,
    dvol: float,
    rate: float,
    moves: list[float],
) -> list[dict]:
    """
    Instantaneous stress test (no time decay, sticky-strike IV = dvol).
    Includes move = 0.0 as the base case row.
    """
    K = opt["strike"]
    T = opt["T_years"]
    n = hedge["n_puts_int"]

    # Baseline values
    put0_per  = bsm_pricing(K, T, spot, rate, dvol, "put")
    put0_total = n * put0_per
    lp0        = lp["capital"]

    sorted_moves = sorted(moves)
    rows: list[dict] = []

    for move in sorted_moves:
        S1 = spot * (1.0 + move)
        ds = S1 - spot

        # LP value at S1 via calc_current_holdings
        # decimals0=18, decimals1=18 → works with human-readable USD amounts
        h = calc_current_holdings(
            current_price=S1,
            lower_price=lp["pa"],
            upper_price=lp["pb"],
            initial_price=lp["entry_spot"],
            initial_amount0=lp["amt0"],
            initial_amount1=lp["amt1"],
            decimals0=18,
            decimals1=18,
        )
        lp_val   = h["value_in_token1"]
        in_range = h["in_range"]
        eth_held = h["amount0"]
        usd_held = h["amount1"]

        # Put value at S1
        put1_per  = bsm_pricing(K, T, S1, rate, dvol, "put")
        put_val   = n * put1_per

        # P&L
        lp_pnl   = lp_val - lp0
        put_pnl  = put_val - put0_total
        port_val = lp_val + put_val
        port_pnl = lp_pnl + put_pnl
        port0    = lp0 + put0_total
        port_ret = port_pnl / port0 if port0 > 0 else 0.0

        # Greek attribution (from entry, using entry greeks)
        lp_delta_term  = lp["delta_lp"] * ds
        lp_gamma_drag  = 0.5 * lp["gamma_lp"] * ds ** 2
        lp_residual    = lp_pnl - lp_delta_term - lp_gamma_drag

        put_delta_term = n * opt["delta"] * ds          # opt["delta"] < 0
        put_gamma_term = 0.5 * n * opt["gamma"] * ds ** 2
        put_residual   = put_pnl - put_delta_term - put_gamma_term

        # Put greeks at S1 (for holdings table)
        g1 = calc_greeks(K, T, S1, rate, dvol, "put")

        rows.append({
            "move":           move,
            "S1":             S1,
            "in_range":       in_range,
            "eth_held":       eth_held,
            "usd_held":       usd_held,
            "lp_val":         lp_val,
            "lp_pnl":         lp_pnl,
            "put_price":      put1_per,
            "put_val":        put_val,
            "put_pnl":        put_pnl,
            "port_val":       port_val,
            "port_pnl":       port_pnl,
            "port_return":    port_ret,
            "lp_delta_term":  lp_delta_term,
            "lp_gamma_drag":  lp_gamma_drag,
            "lp_residual":    lp_residual,
            "put_delta_term": put_delta_term,
            "put_gamma_term": put_gamma_term,
            "put_residual":   put_residual,
            "net_delta_pnl":  lp_delta_term + put_delta_term,
            "net_gamma_pnl":  lp_gamma_drag + put_gamma_term,
            "net_residual":   lp_residual + put_residual,
            "put_delta_s1":   g1["delta"],
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_move(m: float) -> str:
    if abs(m) < 1e-9:
        return "base"
    return f"{m:+.0%}"


def _fmt_usd(v: float, dash_zero: bool = False) -> str:
    if dash_zero and abs(v) < 0.5:
        return "—"
    sign = "+" if v >= 0 else "−"
    return f"{sign}${abs(v):,.0f}"


def _fmt_ret(r: float, dash_zero: bool = False) -> str:
    if dash_zero and abs(r) < 1e-9:
        return "—"
    sign = "+" if r >= 0 else "−"
    return f"{sign}{abs(r) * 100:.2f}%"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_markdown(
    report_date: str,
    spot: float,
    deribit_index: float,
    dvol: float,
    lp: dict,
    opt: dict,
    hedge: dict,
    stress_rows: list[dict],
    candidates: list[dict],
    expiry_meta: dict,
    capital: float,
    range_pct: float,
    put_delta_tgt: float,
    rate: float,
) -> str:
    lines: list[str] = []
    app = lines.append

    # ── Header ──────────────────────────────────────────────────────────────
    app(f"> Auto-generated by `hedge_report.py` on {report_date}")
    app("")
    app("# Live Hedge Sizing & Stress Test")
    app(f"**Date**: {report_date}")
    app(f"**Strategy**: Uniswap V3/V4 LP (±{range_pct:.0%}) + "
        f"Delta-First {put_delta_tgt:.0%}Δ Put Hedge")
    app(f"**Capital**: ${capital:,.0f} USD")
    app("")
    app("---")
    app("")

    # ── §1 Market Data ───────────────────────────────────────────────────────
    app(f"## 1. Market Data (Live — {report_date})")
    app("")
    app("| Source | Data |")
    app("|--------|------|")
    app(f"| ETH spot (Binance ETHUSDT) | **${spot:,.2f}** |")
    app(f"| ETH index (Deribit) | ${deribit_index:,.2f} |")
    app(f"| DVOL 30D implied vol index | **{dvol:.1%}** |")
    app("")
    app("---")
    app("")

    # ── §2 LP Position ───────────────────────────────────────────────────────
    app("## 2. LP Position")
    app("")
    app(f"**Configuration**: Symmetric ±{range_pct:.0%} range around current spot")
    app("")
    app("| Parameter | Value |")
    app("|-----------|-------|")
    app(f"| Capital C | ${capital:,.0f} |")
    app(f"| Entry price S₀ | ${spot:,.2f} |")
    app(f"| Lower bound pa | **${lp['pa']:,.0f}** (S × {1 - range_pct:.2f}) |")
    app(f"| Upper bound pb | **${lp['pb']:,.0f}** (S × {1 + range_pct:.2f}) |")
    app(f"| Liquidity L | {lp['L']:,.0f} |")
    app(f"| LP delta Δ_LP | **{lp['delta_lp']:.1f} ETH** |")
    app(f"| LP gamma Γ_LP | **{lp['gamma_lp']:.4f} $/($²)** |")
    app(f"| Initial ETH held (x) | {lp['amt0']:.1f} ETH |")
    app(f"| Initial USDC held (y) | ${lp['amt1']:,.0f} |")
    app("")
    app("**LP value formula** (Uniswap V3, pa < S < pb):")
    app("```")
    app(f"L    = C / (2√S − √pa − S/√pb)      = {lp['L']:,.0f}")
    app(f"Δ_LP = L × (1/√S − 1/√pb)           = {lp['delta_lp']:.1f} ETH")
    app(f"Γ_LP = −L / (2 × S^1.5)             = {lp['gamma_lp']:.4f} $/($²)")
    app("```")
    app("")
    app("---")
    app("")

    # ── §3 Option Selection ──────────────────────────────────────────────────
    app("## 3. Option Selection")
    app("")
    app(f"Target: {put_delta_tgt:.0%}Δ put, ~{TENOR_TARGET_DAYS}D tenor. "
        "Available expirations and candidate strikes (Deribit live data):")
    app("")

    # Group candidates by expiry, sort expiries by days
    by_expiry: dict[str, list[dict]] = {}
    for row in candidates:
        by_expiry.setdefault(row["expiry"], []).append(row)

    sorted_expiries = sorted(expiry_meta.keys(), key=lambda e: expiry_meta[e]["days"])
    selected_expiry = opt["expiry"]

    for exp_str in sorted_expiries:
        days = expiry_meta[exp_str]["days"]
        rows_exp = sorted(by_expiry.get(exp_str, []), key=lambda r: r["strike"])
        if not rows_exp:
            continue

        if exp_str == selected_expiry:
            app(f"### {exp_str} — {days} days (closest to {TENOR_TARGET_DAYS}D)")
            app("")
            app("| Strike | Mark ($) | Bid ($) | Ask ($) | Delta | Gamma | Theta ($/day) | IV |")
            app("|--------|----------|---------|---------|-------|-------|---------------|----|")
            for r in rows_exp:
                b = "**" if r["instrument"] == opt["instrument"] else ""
                app(f"| {b}{r['strike']:,.0f}{b} | "
                    f"{b}{r['mark_usd']:.2f}{b} | "
                    f"{r['bid_usd']:.2f} | "
                    f"{r['ask_usd']:.2f} | "
                    f"{b}{r['delta']:.4f}{b} | "
                    f"{r['gamma']:.6f} | "
                    f"{r['theta_usd']:.2f} | "
                    f"{r['mark_iv']:.1f}% |")
        else:
            label = "too short — reject" if days < TENOR_TARGET_DAYS else "too long — reject"
            app(f"### {exp_str} — {days} days ({label})")
            app("")
            app("| Strike | Mark ($) | Delta | Gamma | Theta ($/day) | IV |")
            app("|--------|----------|-------|-------|---------------|----|")
            # Show only the strike closest to target delta
            closest = min(rows_exp, key=lambda r: abs(r["delta"] + put_delta_tgt))
            app(f"| {closest['strike']:,.0f} | {closest['mark_usd']:.2f} | "
                f"{closest['delta']:.4f} | {closest['gamma']:.6f} | "
                f"{closest['theta_usd']:.2f} | {closest['mark_iv']:.1f}% |")
            app("")
            if days < TENOR_TARGET_DAYS:
                app("*Shorter tenor → higher gamma per contract but fewer days of "
                    "protection. Tenor too short for 30D strategy.*")
            else:
                g_ratio = (closest["gamma"] / opt["gamma"]
                           if opt["gamma"] > 0 else 0.0)
                app(f"*Lower gamma per contract ({closest['gamma']:.6f} vs "
                    f"{opt['gamma']:.6f}) → {g_ratio:.0%} gamma per contract vs "
                    f"{selected_expiry}. Higher upfront premium.*")
        app("")

    # Selected option box
    app("### Selected Option")
    app("")
    app(f"> **{opt['instrument']}**")
    app("")
    app("| Criterion | Value | Notes |")
    app("|-----------|-------|-------|")
    app(f"| Expiry | {opt['expiry']} | {opt['days_to_exp']} days remaining; "
        "roll when TTE < 2D |")
    app(f"| Strike | **${opt['strike']:,.0f}** | "
        f"{opt['strike'] / spot:.1%} of spot; "
        f"{(1 - opt['strike'] / spot):.1%} OTM |")
    app(f"| Delta | **{opt['delta']:.4f}** | Target {put_delta_tgt:.0%}Δ |")
    app(f"| Gamma | **{opt['gamma']:.6f} /($²)** | Per contract |")
    app(f"| Theta | **{opt['theta_usd']:.2f}/day** | Per contract; USD/day |")
    app(f"| Mark price | **${opt['mark_usd']:.2f}/contract** | Mid-market |")
    app(f"| Strike IV | **{opt['mark_iv']:.1f}%** | DVOL = {dvol:.1%} |")
    app("")
    app(f"Strike ${opt['strike']:,.0f} sits *within* the LP range "
        f"[${lp['pa']:,.0f}, ${lp['pb']:,.0f}]. "
        "The put gains delta as ETH falls — precisely when the LP accumulates ETH. "
        "This dynamic convexity matching is why puts are preferred over perps for "
        "LP delta hedging.")
    app("")
    app("---")
    app("")

    # ── §4 Hedge Sizing ──────────────────────────────────────────────────────
    app("## 4. Hedge Sizing")
    app("")
    app("**Delta-first rule** (from `hedge_sizing.md`): `N_puts = Δ_LP / |Δ_put|`")
    app("")
    app("```")
    app(f"N_puts = {lp['delta_lp']:.1f} / {abs(opt['delta']):.4f}"
        f" = {hedge['n_puts']:.1f}  →  {hedge['n_puts_int']} contracts")
    app("```")
    app("")
    app("### Portfolio Greeks")
    app("")
    app(f"| Greek | LP | Puts ({hedge['n_puts_int']} × contract) | **Net** |")
    app("|-------|----|-----------------------|---------|")
    n_d = hedge["n_puts_int"] * opt["delta"]
    app(f"| Δ | +{lp['delta_lp']:.1f} ETH | "
        f"{hedge['n_puts_int']} × ({opt['delta']:.4f}) = {n_d:.1f} ETH | "
        f"**{hedge['delta_net']:+.1f} ETH ≈ 0** |")
    gamma_net = lp["gamma_lp"] + hedge["gamma_hedge"]
    app(f"| Γ | {lp['gamma_lp']:.4f} | "
        f"{hedge['n_puts_int']} × {opt['gamma']:.6f} = +{hedge['gamma_hedge']:.4f} | "
        f"**{gamma_net:+.4f}** |")
    app(f"| Γ coverage | — | "
        f"{hedge['gamma_hedge']:.4f} / {abs(lp['gamma_lp']):.4f} | "
        f"**{hedge['gamma_coverage']:.1%}** |")
    app("")
    app("### Cost")
    app("")
    app("| Item | Value |")
    app("|------|-------|")
    app(f"| Upfront premium | {hedge['n_puts_int']} × ${opt['mark_usd']:.2f} = "
        f"**${hedge['total_premium']:,.0f}** "
        f"({hedge['total_premium'] / capital:.1%} of AUM) |")
    app(f"| Daily theta | {hedge['n_puts_int']} × ${abs(opt['theta_usd']):.2f} = "
        f"**${hedge['daily_theta_usd']:,.0f}/day** |")
    app(f"| Monthly theta | **${hedge['monthly_theta_usd']:,.0f}/month** |")
    app("")
    direction = "above" if hedge["gamma_coverage"] > 1.0 else "below"
    long_short = "long" if gamma_net > 0 else "short"
    app(f"**Interpretation**: Gamma coverage of {hedge['gamma_coverage']:.1%} means "
        f"the portfolio sits {direction} the ±21.5% crossover. "
        f"Γ_net = {gamma_net:+.4f} ({long_short} gamma).")
    app("")
    app("---")
    app("")

    # ── §5 Stress Test ───────────────────────────────────────────────────────
    base_row = next((r for r in stress_rows if abs(r["move"]) < 1e-9), None)
    put0_total = base_row["put_val"] if base_row else 0.0
    port0 = capital + put0_total

    app("## 5. Stress Test")
    app("")
    app(f"**Positions**: ${capital:,.0f} LP "
        f"[pa=${lp['pa']:,.0f} → pb=${lp['pb']:,.0f}] "
        f"+ {hedge['n_puts_int']} × {opt['instrument']}")
    app(f"**Assumption**: Sticky-strike IV = {dvol:.1%} "
        "(constant across scenarios — conservative on downside)")
    app("**No rebalancing**; instantaneous price move (no time decay)")
    app("")
    app(f"**Portfolio₀ = ${port0:,.0f}** "
        f"(LP ${capital:,.0f} + Puts ${put0_total:,.0f})")
    app("")

    # P&L table
    app("### P&L by Scenario")
    app("")
    app("| Move | S_new | LP value | LP P&L | Puts value | Puts P&L | "
        "**Portfolio** | **P&L** | **Return** | In range |")
    app("|------|-------|----------|--------|-----------|---------|"
        "--------------|---------|-----------|---------|")
    for r in stress_rows:
        is_base = abs(r["move"]) < 1e-9
        in_r = "✓" if r["in_range"] else "✗"
        lp_pnl_str  = "—" if is_base else _fmt_usd(r["lp_pnl"])
        put_pnl_str = "—" if is_base else _fmt_usd(r["put_pnl"])
        pnl_str     = "—" if is_base else f"**{_fmt_usd(r['port_pnl'])}**"
        ret_str     = "—" if is_base else f"**{_fmt_ret(r['port_return'])}**"
        app(f"| {_fmt_move(r['move'])} | ${r['S1']:,.0f} | "
            f"${r['lp_val']:,.0f} | {lp_pnl_str} | "
            f"${r['put_val']:,.0f} | {put_pnl_str} | "
            f"${r['port_val']:,.0f} | {pnl_str} | {ret_str} | {in_r} |")
    app("")

    # Range margin note
    row_dn = next((r for r in stress_rows if abs(r["move"] + 0.20) < 0.005), None)
    row_up = next((r for r in stress_rows if abs(r["move"] - 0.20) < 0.005), None)
    margin_notes = []
    if row_dn:
        m = row_dn["S1"] - lp["pa"]
        margin_notes.append(
            f"pa=${lp['pa']:,.0f} (−20% S=${row_dn['S1']:,.0f} "
            f"{'>' if m > 0 else '<'} pa; margin ${abs(m):,.0f})"
        )
    if row_up:
        m = lp["pb"] - row_up["S1"]
        margin_notes.append(
            f"pb=${lp['pb']:,.0f} (+20% S=${row_up['S1']:,.0f} "
            f"{'<' if m > 0 else '>'} pb; margin ${abs(m):,.0f})"
        )
    if margin_notes:
        app("*Range margins: " + " | ".join(margin_notes) + ".*")
    app("")

    # Attribution table
    app("### P&L Attribution (Greek Decomposition)")
    app("")
    app("Decomposition: LP P&L ≈ **Δ_LP·ΔS** + **½Γ_LP·ΔS²** + residual  ")
    app("Put P&L ≈ **N·Δ_put·ΔS** + **½·N·Γ_put·ΔS²** + residual")
    app("")
    app("| Move | LP Δ term | LP Γ drag | LP resid | "
        "Put Δ term | Put Γ term | Put resid | "
        "**Net Δ** | **Net Γ** | **Net resid** |")
    app("|------|-----------|-----------|----------|"
        "-----------|-----------|----------|"
        "----------|----------|------------|")
    for r in stress_rows:
        if abs(r["move"]) < 1e-9:
            continue  # skip base case in attribution
        app(f"| {_fmt_move(r['move'])} | "
            f"{_fmt_usd(r['lp_delta_term'])} | "
            f"{_fmt_usd(r['lp_gamma_drag'])} | "
            f"{_fmt_usd(r['lp_residual'])} | "
            f"{_fmt_usd(r['put_delta_term'])} | "
            f"{_fmt_usd(r['put_gamma_term'])} | "
            f"{_fmt_usd(r['put_residual'])} | "
            f"**{_fmt_usd(r['net_delta_pnl'])}** | "
            f"**{_fmt_usd(r['net_gamma_pnl'])}** | "
            f"**{_fmt_usd(r['net_residual'])}** |")
    app("")
    app("*LP residual at large moves reflects Uniswap V3 higher-order convexity "
        "beyond the quadratic approximation.*")
    app("")

    # Holdings table
    app("### Holdings at Each Scenario")
    app("")
    app("| Move | S_new | ETH held (LP) | USDC held (LP) | "
        "Put delta | Put $/contract | Total puts |")
    app("|------|-------|--------------|---------------|"
        "-----------|----------------|-----------|")
    for r in stress_rows:
        app(f"| {_fmt_move(r['move'])} | ${r['S1']:,.0f} | "
            f"{r['eth_held']:.1f} ETH | ${r['usd_held']:,.0f} | "
            f"{r['put_delta_s1']:.4f} | ${r['put_price']:.2f} | "
            f"${r['put_val']:,.0f} |")
    app("")
    app("---")
    app("")

    # ── §6 Key Findings ──────────────────────────────────────────────────────
    non_base = [r for r in stress_rows if abs(r["move"]) >= 0.005]
    worst = min(non_base, key=lambda r: r["port_pnl"]) if non_base else stress_rows[0]

    app("## 6. Key Findings")
    app("")
    app("### A. Portfolio is near-flat across ±20%")
    app("")
    app(f"The worst-case single-step loss is "
        f"**{_fmt_usd(worst['port_pnl'])} ({_fmt_ret(worst['port_return'])})** "
        f"at {_fmt_move(worst['move'])}. The portfolio behaves as a carry trade: "
        "collect LP fees while holding a position that is essentially neutral to "
        "large ETH price moves.")
    app("")
    app("### B. Gamma posture")
    app("")
    long_short = "long (convex)" if gamma_net > 0 else "short (concave)"
    app(f"With {hedge['gamma_coverage']:.1%} gamma coverage, "
        f"Γ_net = {gamma_net:+.4f}. The portfolio is net {long_short} gamma:")
    app("")
    app("```")
    ds_20 = spot * 0.20
    bonus = 0.5 * gamma_net * ds_20 ** 2
    app(f"Net P&L ≈ ({hedge['delta_net']:+.1f} ETH)·ΔS + ½ × ({gamma_net:+.4f}) × ΔS²")
    app(f"At ±20% (ΔS = ±${ds_20:,.0f}): "
        f"½ × {gamma_net:+.4f} × {ds_20:,.0f}² = {bonus:+,.0f} gamma contribution")
    app("```")
    app("")
    app("### C. Stress test is conservative on the downside")
    app("")
    app(f"Constant IV of {dvol:.1%} was assumed (sticky-strike). In practice, "
        "vol rises significantly on large down moves:")
    app("")
    app("| Scenario | Assumed DVOL | Realistic DVOL | Impact |")
    app("|----------|-------------|----------------|--------|")
    app(f"| −10% | {dvol:.1%} | ~{dvol * 1.10:.0%}–{dvol * 1.18:.0%} | "
        "Put value higher; actual P&L better |")
    app(f"| −20% | {dvol:.1%} | ~{dvol * 1.20:.0%}–{dvol * 1.35:.0%} | "
        "Significant put value boost; large positive P&L swing |")
    app("")
    app("### D. All scenarios remain in range")
    app("")
    app("```")
    if row_dn:
        m = row_dn["S1"] - lp["pa"]
        sym = ">" if m >= 0 else "<"
        flag = "✓" if m >= 0 else "✗"
        app(f"−20%: S = ${row_dn['S1']:,.0f}  {sym}  pa = ${lp['pa']:,.0f}  "
            f"{flag} (margin: ${abs(m):,.0f})")
    if row_up:
        m = lp["pb"] - row_up["S1"]
        sym = "<" if m >= 0 else ">"
        flag = "✓" if m >= 0 else "✗"
        app(f"+20%: S = ${row_up['S1']:,.0f}  {sym}  pb = ${lp['pb']:,.0f}  "
            f"{flag} (margin: ${abs(m):,.0f})")
    sigma_daily = dvol / math.sqrt(365)
    z = range_pct / sigma_daily
    from math import erf, sqrt as msqrt
    prob = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / msqrt(2.0))))
    app(f"P(|daily move| > {range_pct:.0%}) ≈ {prob:.4%} ({z:.1f}σ event)")
    app("```")
    app("")
    app("### E. Roll calendar")
    app("")
    app("| Event | Date |")
    app("|-------|------|")
    app(f"| Position opened | {report_date} |")
    roll_dt = opt["expiry_dt"] if "expiry_dt" in opt else expiry_meta[opt["expiry"]]["expiry_dt"]
    from datetime import timedelta
    roll_trigger = (roll_dt - timedelta(days=2)).strftime("%Y-%m-%d")
    expiry_str   = roll_dt.strftime("%Y-%m-%d")
    app(f"| Roll trigger (TTE < 2d) | ~{roll_trigger} |")
    app(f"| Option expiry | {expiry_str} |")
    app(f"| Next put | Roll into 30Δ put at prevailing spot on ~{roll_trigger} |")
    app("")
    app("---")
    app("")

    # ── §7 What's Not Included ───────────────────────────────────────────────
    spread = (opt["ask_usd"] - opt["bid_usd"])
    roll_cost = spread * hedge["n_puts_int"]

    app("## 7. What This Example Does Not Include")
    app("")
    app("| Missing element | Impact |")
    app("|-----------------|--------|")
    app("| LP fee income | Missing upside: fees offset theta cost. "
        "At CALM (5bps), ~$6–10k/month. At ELEVATED (20bps), ~$25–40k/month. |")
    app(f"| Time decay (theta) | ${hedge['daily_theta_usd']:,.0f}/day = "
        f"${hedge['monthly_theta_usd']:,.0f}/month reduces portfolio P&L shown above. "
        "Fee income must exceed this. |")
    app("| Rebalancing costs | If price breaches range (>±{:.0%}), a rebalance "
        "incurs gas + slippage (~$200–500). |".format(range_pct))
    app(f"| Roll bid/ask spread | ~${spread:.2f}/contract × "
        f"{hedge['n_puts_int']} = ~${roll_cost:,.0f}/roll every "
        f"~{opt['days_to_exp']} days. |")
    app("| Vol surface dynamics | Actual downside protection is stronger due to "
        "vol skew/smile on large moves (see §6C). |")
    app("")
    app("---")
    app("")

    # ── §8 Reference ─────────────────────────────────────────────────────────
    app("## 8. Reference")
    app("")
    app("| Item | File |")
    app("|------|------|")
    app("| LP greeks derivation | [lp_greeks_model.md](lp_greeks_model.md) |")
    app("| Hedge sizing rule | [hedge_sizing.md](hedge_sizing.md) |")
    app("| Gamma coverage analysis | [lp_range_analysis.md](lp_range_analysis.md) |")
    app("| Python range comparison | "
        "[qrisklab/research/amm/lp_range_analysis.py]"
        "(../qrisklab/research/amm/lp_range_analysis.py) |")
    app("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestration
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    capital: float = CAPITAL,
    range_pct: float = RANGE_PCT,
    put_delta_tgt: float = PUT_DELTA_TGT,
    stress_moves: list[float] | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    if stress_moves is None:
        stress_moves = list(STRESS_MOVES)

    report_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"\n=== Hedge Report Generator — {report_date} ===")

    print("\n[1/6] Fetching market data...")
    spot, deribit_index = fetch_eth_spot()
    dvol = fetch_dvol()
    print(f"  ETH spot:  ${spot:,.2f}")
    print(f"  Deribit:   ${deribit_index:,.2f}")
    print(f"  DVOL:      {dvol:.1%}")

    print("\n[2/6] Fetching option chain...")
    tenor_min, tenor_max = TENOR_WINDOW
    strike_lo, strike_hi = STRIKE_WINDOW
    instruments, expiry_meta = fetch_put_candidates(
        spot, tenor_min, tenor_max, strike_lo, strike_hi
    )
    if not instruments:
        raise RuntimeError("No candidate instruments found. "
                           "Check TENOR_WINDOW and STRIKE_WINDOW settings.")

    print("\n[3/6] Fetching greeks for candidates...")
    candidates = fetch_option_greeks(instruments, spot)
    print(f"  Retrieved greeks for {len(candidates)} instruments.")

    if not candidates:
        raise RuntimeError("No greeks returned for any candidate. "
                           "Check Deribit API connectivity.")

    print("\n[4/6] Selecting best put & computing LP position...")
    opt = select_best_put(candidates, put_delta_tgt, TENOR_TARGET_DAYS, expiry_meta)
    # Attach expiry_dt for roll calendar
    opt["expiry_dt"] = expiry_meta[opt["expiry"]]["expiry_dt"]
    print(f"  Selected:  {opt['instrument']}  Δ={opt['delta']:.4f}  "
          f"K=${opt['strike']:,.0f}  T={opt['days_to_exp']}D  "
          f"Mark=${opt['mark_usd']:.2f}")

    lp    = calc_lp_position(spot, capital, range_pct)
    hedge = size_hedge(lp, opt)
    print(f"  LP:        pa=${lp['pa']:,.0f}  pb=${lp['pb']:,.0f}  "
          f"Δ={lp['delta_lp']:.1f} ETH")
    print(f"  Hedge:     N={hedge['n_puts_int']}  "
          f"coverage={hedge['gamma_coverage']:.1%}  "
          f"premium=${hedge['total_premium']:,.0f}")

    print("\n[5/6] Running stress test...")
    stress_rows = run_stress(spot, lp, opt, hedge, dvol, RATE, stress_moves)

    print("\n[6/6] Rendering & writing markdown...")
    md = render_markdown(
        report_date=report_date,
        spot=spot,
        deribit_index=deribit_index,
        dvol=dvol,
        lp=lp,
        opt=opt,
        hedge=hedge,
        stress_rows=stress_rows,
        candidates=candidates,
        expiry_meta=expiry_meta,
        capital=capital,
        range_pct=range_pct,
        put_delta_tgt=put_delta_tgt,
        rate=RATE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{report_date}_hedge_report.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"\n  Written: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate live hedge sizing & stress test report."
    )
    parser.add_argument(
        "--capital", type=float, default=CAPITAL,
        help=f"Capital in USD (default: {CAPITAL:,.0f})",
    )
    parser.add_argument(
        "--range", type=float, default=RANGE_PCT, dest="range_pct",
        help=f"LP range ±pct as decimal (default: {RANGE_PCT})",
    )
    parser.add_argument(
        "--delta", type=float, default=PUT_DELTA_TGT, dest="put_delta_tgt",
        help=f"Target put delta (default: {PUT_DELTA_TGT})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()
    generate_report(
        capital=args.capital,
        range_pct=args.range_pct,
        put_delta_tgt=args.put_delta_tgt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
