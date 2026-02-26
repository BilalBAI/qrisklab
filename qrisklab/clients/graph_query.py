"""
GraphQL query templates for The Graph Protocol subgraphs.

Endpoints and reusable query builders for Uniswap V3 / V4.
https://docs.uniswap.org/api/subgraph/overview
"""

from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

UNISWAP_V3_ENDPOINT = (
    "https://gateway.thegraph.com/api/subgraphs/id/"
    "5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"
)

UNISWAP_V4_ENDPOINT = (
    "https://gateway.thegraph.com/api/subgraphs/id/"
    "DiYPVdygkfjDWhbxGSqAQxwBKmfKnkWQojqeM2rkLb3G"
)

# ---------------------------------------------------------------------------
# Shared field fragments (kept as plain strings so they compose easily)
# ---------------------------------------------------------------------------

POOL_FIELDS = """
    id
    feeTier
    tick
    liquidity
    sqrtPrice
    totalValueLockedUSD
    volumeUSD
    txCount
    createdAtTimestamp
    token0 { id symbol name decimals }
    token1 { id symbol name decimals }
"""

POOL_FIELDS_COMPACT = """
    id
    feeTier
    tick
    liquidity
    sqrtPrice
    token0Price
    token1Price
    totalValueLockedUSD
    volumeUSD
    txCount
    createdAtTimestamp
    token0 { id symbol decimals }
    token1 { id symbol decimals }
"""

TICK_FIELDS = """
    tickIdx
    liquidityGross
    liquidityNet
"""

POSITION_FIELDS = """
    id
    owner
    pool {
      id
      token0 { symbol decimals }
      token1 { symbol decimals }
      feeTier
    }
    tickLower
    tickUpper
    liquidity
    depositedToken0
    depositedToken1
    withdrawnToken0
    withdrawnToken1
    collectedFeesToken0
    collectedFeesToken1
    transaction {
      timestamp
      blockNumber
    }
"""

SWAP_FIELDS = """
    id
    transaction {
      id
      timestamp
      blockNumber
    }
    pool {
      id
      token0 { symbol }
      token1 { symbol }
    }
    amount0
    amount1
    amountUSD
    sqrtPriceX96
    tick
"""

# ---------------------------------------------------------------------------
# Helper: build a where clause from a dict
# ---------------------------------------------------------------------------

def _where_clause(conditions: Optional[Dict[str, Any]] = None) -> str:
    if not conditions:
        return ""
    parts: List[str] = []
    for key, value in conditions.items():
        if isinstance(value, str):
            parts.append(f'{key}: "{value}"')
        elif isinstance(value, bool):
            parts.append(f"{key}: {str(value).lower()}")
        elif isinstance(value, (int, float)):
            parts.append(f"{key}: {value}")
    if not parts:
        return ""
    return f", where: {{{', '.join(parts)}}}"


# ---------------------------------------------------------------------------
# Uniswap V3 queries
# ---------------------------------------------------------------------------

def v3_pool_query(pool_address: str) -> str:
    """Single pool by address."""
    return f"""
    {{
      pool(id: "{pool_address.lower()}") {{{POOL_FIELDS}}}
    }}
    """


def v3_pools_query(
    first: int = 50,
    skip: int = 0,
    order_by: str = "totalValueLockedUSD",
    order_direction: str = "desc",
    where: Optional[Dict[str, Any]] = None,
) -> str:
    """Multiple pools with pagination and optional filter."""
    wc = _where_clause(where)
    return f"""
    {{
      pools(
        first: {first}
        skip: {skip}
        orderBy: {order_by}
        orderDirection: {order_direction}
        {wc}
      ) {{{POOL_FIELDS}}}
    }}
    """


def v3_ticks_query(
    pool_address: str,
    first: int = 1000,
    skip: int = 0,
    liquidity_net_not_zero: bool = True,
) -> str:
    """Initialized ticks for a pool."""
    where_parts = f'poolAddress: "{pool_address.lower()}"'
    if liquidity_net_not_zero:
        where_parts += ', liquidityNet_not: "0"'
    return f"""
    {{
      ticks(
        where: {{{where_parts}}}
        first: {first}
        skip: {skip}
        orderBy: tickIdx
        orderDirection: asc
      ) {{{TICK_FIELDS}}}
    }}
    """


def v3_positions_query(
    pool_address: Optional[str] = None,
    owner: Optional[str] = None,
    first: int = 100,
    skip: int = 0,
) -> str:
    """LP positions, optionally filtered by pool and/or owner."""
    conditions: List[str] = []
    if pool_address:
        conditions.append(f'pool: "{pool_address.lower()}"')
    if owner:
        conditions.append(f'owner: "{owner.lower()}"')
    wc = ""
    if conditions:
        wc = f", where: {{{', '.join(conditions)}}}"
    return f"""
    {{
      positions(
        first: {first}
        skip: {skip}
        {wc}
      ) {{{POSITION_FIELDS}}}
    }}
    """


def v3_swaps_query(
    pool_address: Optional[str] = None,
    first: int = 100,
    skip: int = 0,
    order_by: str = "timestamp",
    order_direction: str = "desc",
) -> str:
    """Swap events, optionally filtered by pool."""
    wc = ""
    if pool_address:
        wc = f', where: {{pool: "{pool_address.lower()}"}}'
    return f"""
    {{
      swaps(
        first: {first}
        skip: {skip}
        orderBy: {order_by}
        orderDirection: {order_direction}
        {wc}
      ) {{{SWAP_FIELDS}}}
    }}
    """


def v3_pool_day_data_query(
    pool_address: str,
    first: int = 30,
    skip: int = 0,
    order_by: str = "date",
    order_direction: str = "desc",
) -> str:
    """Daily OHLC / volume / TVL snapshots for a pool."""
    return f"""
    {{
      poolDayDatas(
        where: {{pool: "{pool_address.lower()}"}}
        first: {first}
        skip: {skip}
        orderBy: {order_by}
        orderDirection: {order_direction}
      ) {{
        date
        pool {{ id }}
        liquidity
        sqrtPrice
        token0Price
        token1Price
        tick
        tvlUSD
        volumeToken0
        volumeToken1
        volumeUSD
        feesUSD
        txCount
        open
        high
        low
        close
      }}
    }}
    """


def v3_pool_hour_data_query(
    pool_address: str,
    first: int = 168,
    skip: int = 0,
    order_by: str = "periodStartUnix",
    order_direction: str = "desc",
) -> str:
    """Hourly snapshots for a pool."""
    return f"""
    {{
      poolHourDatas(
        where: {{pool: "{pool_address.lower()}"}}
        first: {first}
        skip: {skip}
        orderBy: {order_by}
        orderDirection: {order_direction}
      ) {{
        periodStartUnix
        pool {{ id }}
        liquidity
        sqrtPrice
        token0Price
        token1Price
        tick
        tvlUSD
        volumeToken0
        volumeToken1
        volumeUSD
        feesUSD
        txCount
        open
        high
        low
        close
      }}
    }}
    """


def v3_token_query(token_address: str) -> str:
    """Single token metadata and stats."""
    return f"""
    {{
      token(id: "{token_address.lower()}") {{
        id
        symbol
        name
        decimals
        totalSupply
        volume
        volumeUSD
        totalValueLocked
        totalValueLockedUSD
        txCount
        derivedETH
      }}
    }}
    """


def v3_factory_query() -> str:
    """Global Uniswap V3 factory stats."""
    return """
    {
      factories(first: 1) {
        id
        poolCount
        txCount
        totalVolumeUSD
        totalVolumeETH
        totalValueLockedUSD
        totalValueLockedETH
        totalFeesUSD
        totalFeesETH
      }
    }
    """


def v3_mints_query(
    pool_address: Optional[str] = None,
    first: int = 100,
    skip: int = 0,
    order_by: str = "timestamp",
    order_direction: str = "desc",
) -> str:
    """Mint (add-liquidity) events."""
    wc = ""
    if pool_address:
        wc = f', where: {{pool: "{pool_address.lower()}"}}'
    return f"""
    {{
      mints(
        first: {first}
        skip: {skip}
        orderBy: {order_by}
        orderDirection: {order_direction}
        {wc}
      ) {{
        id
        transaction {{ id timestamp blockNumber }}
        pool {{ id token0 {{ symbol }} token1 {{ symbol }} }}
        owner
        sender
        tickLower
        tickUpper
        amount
        amount0
        amount1
        amountUSD
      }}
    }}
    """


def v3_burns_query(
    pool_address: Optional[str] = None,
    first: int = 100,
    skip: int = 0,
    order_by: str = "timestamp",
    order_direction: str = "desc",
) -> str:
    """Burn (remove-liquidity) events."""
    wc = ""
    if pool_address:
        wc = f', where: {{pool: "{pool_address.lower()}"}}'
    return f"""
    {{
      burns(
        first: {first}
        skip: {skip}
        orderBy: {order_by}
        orderDirection: {order_direction}
        {wc}
      ) {{
        id
        transaction {{ id timestamp blockNumber }}
        pool {{ id token0 {{ symbol }} token1 {{ symbol }} }}
        owner
        tickLower
        tickUpper
        amount
        amount0
        amount1
        amountUSD
      }}
    }}
    """


# ---------------------------------------------------------------------------
# Uniswap V4 queries
# ---------------------------------------------------------------------------

UNISWAP_V4_POOL_MANAGER_ID = "0x000000000004444c5dc75cb358380d2e3de08a90"


def v4_overview_query(pool_manager_id: str = UNISWAP_V4_POOL_MANAGER_ID) -> str:
    """V4 pool manager stats + ETH price + top pools."""
    return f"""
    {{
      poolManager(id: "{pool_manager_id}") {{
        poolCount
        txCount
        totalVolumeUSD
        totalVolumeETH
      }}
      bundles(first: 1) {{
        id
        ethPriceUSD
      }}
      pools(
        first: 50
        orderBy: totalValueLockedUSD
        orderDirection: desc
      ) {{{POOL_FIELDS_COMPACT}}}
    }}
    """
