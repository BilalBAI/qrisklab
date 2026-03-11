"""
market_data.py
==============
Lightweight market data helpers for ETH spot, DVOL, and Deribit option
order books.  All functions use plain ``requests`` with an automatic SSL
fallback (needed on macOS with missing root-cert bundles).

These are intentionally thin wrappers — no pandas, no class hierarchy —
so they can be imported anywhere without pulling in heavy dependencies.
"""
from __future__ import annotations

import time
import warnings
from datetime import datetime, timezone

import requests


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
# Spot price
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


# ─────────────────────────────────────────────────────────────────────────────
# DVOL
# ─────────────────────────────────────────────────────────────────────────────

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
# Option order book
# ─────────────────────────────────────────────────────────────────────────────

def fetch_option_greeks(instruments: list[str], index_price: float) -> list[dict]:
    """
    Fetch order book + greeks for each instrument from Deribit.

    Parameters
    ----------
    instruments : list[str]
        Deribit instrument names, e.g. ["ETH-27MAR26-2000-P"].
    index_price : float
        Current ETH index price used to convert ETH-denominated mark prices to USD.

    Returns
    -------
    list[dict]
        One dict per instrument with keys:
        instrument, expiry, strike, mark_usd, mark_eth, bid_usd, ask_usd,
        delta, gamma, theta_usd, vega, mark_iv, index_price.
        Instruments with no market data (no mark price or no greeks) are skipped.
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
