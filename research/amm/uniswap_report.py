"""
Uniswap V3 & V4 Market Report

Generates a summary report covering:
  - V3 factory-level statistics
  - V3 top pools by TVL
  - V3 recent swaps and daily data for a featured pool
  - V4 pool-manager overview and top pools

Usage:
    uv run python examples/uniswap_report.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import pandas as pd

from qrisklab.clients.graph import GraphClient
from qrisklab.clients.graph_query import (
    UNISWAP_V3_ENDPOINT,
    UNISWAP_V4_ENDPOINT,
    v3_factory_query,
    v3_pools_query,
    v3_pool_query,
    v3_pool_day_data_query,
    v3_swaps_query,
    v3_token_query,
    v3_mints_query,
    v3_burns_query,
    v4_overview_query,
)

WIDTH = 88
THIN = "-" * WIDTH
THICK = "=" * WIDTH


def _fmt_usd(value) -> str:
    v = float(value)
    if v >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:,.2f}M"
    if v >= 1e3:
        return f"${v / 1e3:,.2f}K"
    return f"${v:,.2f}"


def _fmt_count(value) -> str:
    v = int(value)
    if v >= 1e6:
        return f"{v / 1e6:,.1f}M"
    if v >= 1e3:
        return f"{v / 1e3:,.1f}K"
    return f"{v:,}"


def _ts_to_date(ts) -> str:
    return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")


def _fee_label(fee_tier) -> str:
    return f"{int(fee_tier) / 10_000:.2f}%"


def _section(title: str):
    print(f"\n{THICK}")
    print(f"  {title}")
    print(THICK)


# ── V3 factory ──────────────────────────────────────────────────────────────

def report_v3_factory(client: GraphClient):
    _section("UNISWAP V3 — FACTORY OVERVIEW")
    data = client.query(v3_factory_query())
    factory = data.get("factories", [{}])[0]
    print(f"  Pool count          : {_fmt_count(factory.get('poolCount', 0))}")
    print(f"  Total transactions  : {_fmt_count(factory.get('txCount', 0))}")
    print(f"  Total volume (USD)  : {_fmt_usd(factory.get('totalVolumeUSD', 0))}")
    print(f"  Total volume (ETH)  : {float(factory.get('totalVolumeETH', 0)):,.0f} ETH")
    print(f"  Total TVL (USD)     : {_fmt_usd(factory.get('totalValueLockedUSD', 0))}")
    print(f"  Total TVL (ETH)     : {float(factory.get('totalValueLockedETH', 0)):,.0f} ETH")
    print(f"  Total fees (USD)    : {_fmt_usd(factory.get('totalFeesUSD', 0))}")
    print(f"  Total fees (ETH)    : {float(factory.get('totalFeesETH', 0)):,.0f} ETH")


# ── V3 top pools ────────────────────────────────────────────────────────────

def report_v3_top_pools(client: GraphClient, n: int = 15) -> pd.DataFrame:
    _section(f"UNISWAP V3 — TOP {n} POOLS BY TVL")
    data = client.query(v3_pools_query(first=n))
    pools = data.get("pools", [])
    rows = []
    for p in pools:
        t0 = p.get("token0", {}).get("symbol", "?")
        t1 = p.get("token1", {}).get("symbol", "?")
        rows.append({
            "pair": f"{t0}/{t1}",
            "fee": _fee_label(p.get("feeTier", 0)),
            "tvl": _fmt_usd(p.get("totalValueLockedUSD", 0)),
            "volume": _fmt_usd(p.get("volumeUSD", 0)),
            "txs": _fmt_count(p.get("txCount", 0)),
            "created": _ts_to_date(p.get("createdAtTimestamp", 0)),
            "address": p.get("id", ""),
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return df


# ── V3 featured pool deep-dive ─────────────────────────────────────────────

def report_v3_pool_detail(client: GraphClient, pool_address: str):
    _section("UNISWAP V3 — POOL DETAIL")
    data = client.query(v3_pool_query(pool_address))
    pool = data.get("pool", {})
    if not pool:
        print(f"  Pool {pool_address} not found.")
        return

    t0 = pool.get("token0", {})
    t1 = pool.get("token1", {})
    pair = f"{t0.get('symbol', '?')}/{t1.get('symbol', '?')}"
    print(f"  Pool    : {pair}  ({pool_address[:10]}…)")
    print(f"  Fee tier: {_fee_label(pool.get('feeTier', 0))}")
    print(f"  TVL     : {_fmt_usd(pool.get('totalValueLockedUSD', 0))}")
    print(f"  Volume  : {_fmt_usd(pool.get('volumeUSD', 0))}")
    print(f"  Txs     : {_fmt_count(pool.get('txCount', 0))}")
    print(f"  Tick    : {pool.get('tick')}")

    # Daily data
    print(f"\n  {THIN}")
    print(f"  Last 7 daily snapshots")
    print(f"  {THIN}")
    day_data = client.query(v3_pool_day_data_query(pool_address, first=7))
    days = day_data.get("poolDayDatas", [])
    if days:
        day_rows = []
        for d in days:
            day_rows.append({
                "date": _ts_to_date(d.get("date", 0)),
                "volume_usd": _fmt_usd(d.get("volumeUSD", 0)),
                "fees_usd": _fmt_usd(d.get("feesUSD", 0)),
                "tvl_usd": _fmt_usd(d.get("tvlUSD", 0)),
                "txs": _fmt_count(d.get("txCount", 0)),
                "close": f"{float(d.get('close', 0)):,.2f}",
            })
        print(pd.DataFrame(day_rows).to_string(index=False))
    else:
        print("  (no daily data available)")

    # Recent swaps
    print(f"\n  {THIN}")
    print(f"  Last 10 swaps")
    print(f"  {THIN}")
    swap_data = client.query(v3_swaps_query(pool_address, first=10))
    swaps = swap_data.get("swaps", [])
    if swaps:
        swap_rows = []
        for s in swaps:
            tx = s.get("transaction", {})
            swap_rows.append({
                "time": _ts_to_date(tx.get("timestamp", 0)),
                "amount0": f"{float(s.get('amount0', 0)):+.6f}",
                "amount1": f"{float(s.get('amount1', 0)):+.2f}",
                "value_usd": _fmt_usd(abs(float(s.get("amountUSD", 0)))),
            })
        print(pd.DataFrame(swap_rows).to_string(index=False))
    else:
        print("  (no swap data available)")

    # Recent mints & burns
    print(f"\n  {THIN}")
    print(f"  Last 5 mints (add-liquidity)")
    print(f"  {THIN}")
    mint_data = client.query(v3_mints_query(pool_address, first=5))
    mints = mint_data.get("mints", [])
    if mints:
        mint_rows = []
        for m in mints:
            tx = m.get("transaction", {})
            mint_rows.append({
                "time": _ts_to_date(tx.get("timestamp", 0)),
                "tick_range": f"[{m.get('tickLower')}, {m.get('tickUpper')}]",
                "amount0": f"{float(m.get('amount0', 0)):.6f}",
                "amount1": f"{float(m.get('amount1', 0)):.2f}",
                "value_usd": _fmt_usd(abs(float(m.get("amountUSD", 0)))),
            })
        print(pd.DataFrame(mint_rows).to_string(index=False))
    else:
        print("  (no mint data available)")

    print(f"\n  {THIN}")
    print(f"  Last 5 burns (remove-liquidity)")
    print(f"  {THIN}")
    burn_data = client.query(v3_burns_query(pool_address, first=5))
    burns = burn_data.get("burns", [])
    if burns:
        burn_rows = []
        for b in burns:
            tx = b.get("transaction", {})
            burn_rows.append({
                "time": _ts_to_date(tx.get("timestamp", 0)),
                "tick_range": f"[{b.get('tickLower')}, {b.get('tickUpper')}]",
                "amount0": f"{float(b.get('amount0', 0)):.6f}",
                "amount1": f"{float(b.get('amount1', 0)):.2f}",
                "value_usd": _fmt_usd(abs(float(b.get("amountUSD", 0)))),
            })
        print(pd.DataFrame(burn_rows).to_string(index=False))
    else:
        print("  (no burn data available)")


# ── V3 token spotlight ──────────────────────────────────────────────────────

def report_v3_token(client: GraphClient, token_address: str):
    data = client.query(v3_token_query(token_address))
    token = data.get("token", {})
    if not token:
        return
    print(f"\n  {token.get('symbol')} ({token.get('name')})")
    print(f"    Address     : {token.get('id')}")
    print(f"    Decimals    : {token.get('decimals')}")
    print(f"    Volume (USD): {_fmt_usd(token.get('volumeUSD', 0))}")
    print(f"    TVL (USD)   : {_fmt_usd(token.get('totalValueLockedUSD', 0))}")
    print(f"    Txs         : {_fmt_count(token.get('txCount', 0))}")
    print(f"    ETH price   : {float(token.get('derivedETH', 0)):.8f}")


def report_v3_tokens(client: GraphClient):
    _section("UNISWAP V3 — TOKEN SPOTLIGHT")
    weth = "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2"
    usdc = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48"
    usdt = "0xdac17f958d2ee523a2206206994597c13d831ec7"
    wbtc = "0x2260fac5e5542a773aa44fbcfedf7c193bc2c599"
    for addr in [weth, usdc, usdt, wbtc]:
        try:
            report_v3_token(client, addr)
        except Exception as e:
            print(f"  (could not fetch token {addr[:10]}…: {e})")


# ── V4 overview ─────────────────────────────────────────────────────────────

def report_v4_overview(client: GraphClient):
    _section("UNISWAP V4 — POOL MANAGER OVERVIEW")
    data = client.query(v4_overview_query())

    pm = data.get("poolManager", {})
    print(f"  Pool count          : {_fmt_count(pm.get('poolCount', 0))}")
    print(f"  Total transactions  : {_fmt_count(pm.get('txCount', 0))}")
    print(f"  Total volume (USD)  : {_fmt_usd(pm.get('totalVolumeUSD', 0))}")
    print(f"  Total volume (ETH)  : {float(pm.get('totalVolumeETH', 0)):,.0f} ETH")

    bundles = data.get("bundles", [])
    if bundles:
        eth_price = float(bundles[0].get("ethPriceUSD", 0))
        print(f"  ETH price (USD)     : ${eth_price:,.2f}")

    pools = data.get("pools", [])
    if pools:
        print(f"\n  {THIN}")
        print(f"  Top {len(pools)} V4 pools by TVL")
        print(f"  {THIN}")
        rows = []
        for p in pools:
            t0 = p.get("token0", {}).get("symbol", "?")
            t1 = p.get("token1", {}).get("symbol", "?")
            rows.append({
                "pair": f"{t0}/{t1}",
                "fee": _fee_label(p.get("feeTier", 0)),
                "tvl": _fmt_usd(p.get("totalValueLockedUSD", 0)),
                "volume": _fmt_usd(p.get("volumeUSD", 0)),
                "txs": _fmt_count(p.get("txCount", 0)),
            })
        print(pd.DataFrame(rows).to_string(index=False))


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    api_key = os.getenv("GRAPH_API_KEY")
    if not api_key:
        print("ERROR: GRAPH_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(THICK)
    print(f"  UNISWAP V3 & V4 MARKET REPORT")
    print(f"  Generated: {now}")
    print(THICK)

    # WETH/USDC 0.3% — the canonical deep-liquidity pool
    featured_pool = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"

    # ── V3 ──
    v3 = GraphClient(endpoint=UNISWAP_V3_ENDPOINT, api_key=api_key)
    report_v3_factory(v3)
    report_v3_top_pools(v3, n=15)
    report_v3_tokens(v3)
    report_v3_pool_detail(v3, featured_pool)

    # ── V4 ──
    v4 = GraphClient(endpoint=UNISWAP_V4_ENDPOINT, api_key=api_key)
    report_v4_overview(v4)

    print(f"\n{THICK}")
    print("  Report complete.")
    print(THICK)


if __name__ == "__main__":
    main()
