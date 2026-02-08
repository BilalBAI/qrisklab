"""
Simple Graph Protocol client for querying blockchain and DeFi data.

This client provides a simple interface to The Graph Protocol subgraphs.
"""

import requests
import pandas as pd
from typing import Dict, Any, Optional, List


# https://docs.uniswap.org/api/subgraph/overview
# Uniswap V3
UNISWAP_V3_ENDPOINT = 'https://gateway.thegraph.com/api/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV'

UNISWAP_V3_QUERY_EXAMPLE = """{
  pools(
    first: 50
    skip: 0
    orderBy: totalValueLockedUSD
    orderDirection: desc
  ) {
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
  }
}"""

# Uniswap V4
UNISWAP_V4_ENDPOINT = 'https://gateway.thegraph.com/api/subgraphs/id/DiYPVdygkfjDWhbxGSqAQxwBKmfKnkWQojqeM2rkLb3G'
UNISWAP_V4_QUERY_EXAMPLE = """ 
{
  poolManager(id: "0x000000000004444c5dc75cb358380d2e3de08a90") {
    poolCount
    txCount
    totalVolumeUSD
    totalVolumeETH
  }

  bundles(first: 1) {
    id
    ethPriceUSD
  }

  pools(
    first: 50
    orderBy: totalValueLockedUSD
    orderDirection: desc
  ) {
    id
    createdAtTimestamp
    token0 { id symbol decimals }
    token1 { id symbol decimals }
    feeTier
    liquidity
    sqrtPrice
    tick
    token0Price
    token1Price
    totalValueLockedUSD
    volumeUSD
    txCount
  }
}

"""


class GraphClient:
    """
    Simple client for querying The Graph Protocol subgraphs.

    Example:
        client = GraphClient(
            endpoint='https://gateway.thegraph.com/api/subgraphs/id/YOUR_DEPLOYMENT_ID',
            api_key='your-api-key'
        )
        query = '{ pools(first: 10) { id totalValueLockedUSD } }'
        data = client.query(query)
    """

    def __init__(
        self,
        endpoint: str,
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the Graph client.

        Args:
            endpoint: Full subgraph endpoint URL (e.g., 
                     'https://gateway.thegraph.com/api/subgraphs/id/YOUR_DEPLOYMENT_ID')
            api_key: Optional API key for authenticated requests
            timeout: Request timeout in seconds
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def query(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a GraphQL query.

        Args:
            query: GraphQL query string
            variables: Optional dictionary of variables for the query

        Returns:
            Dictionary containing the query response data

        Raises:
            requests.RequestException: If the request fails
            ValueError: If the response contains GraphQL errors
        """
        # Prepare headers
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Prepare payload
        payload = {
            "query": query,
        }
        if variables:
            payload["variables"] = variables

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Check for GraphQL errors
            if "errors" in data:
                error_messages = [err.get("message", str(err)) for err in data["errors"]]
                raise ValueError(f"GraphQL errors: {', '.join(error_messages)}")

            return data.get("data", {})

        except requests.RequestException as e:
            raise requests.RequestException(f"Failed to execute GraphQL query: {e}")

    def query_to_dataframe(
        self,
        query: str,
        variables: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Execute a GraphQL query and return results as a pandas DataFrame.

        Args:
            query: GraphQL query string
            variables: Optional dictionary of variables for the query
            key: Key in response to convert to DataFrame. If None, uses first list found.

        Returns:
            DataFrame with query results
        """
        data = self.query(query, variables)

        if key:
            if key in data and isinstance(data[key], list):
                return pd.DataFrame(data[key])
            else:
                raise ValueError(f"Key '{key}' not found or not a list in response")
        else:
            # Try to find the first list in the response
            for key, value in data.items():
                if isinstance(value, list):
                    return pd.DataFrame(value)
            raise ValueError("No list found in response to convert to DataFrame")


class UniV3GraphClient(GraphClient):
    """
    Specialized client for Uniswap V3 subgraph queries.

    Inherits from GraphClient and adds Uniswap V3 specific methods.

    Example:
        client = UniV3GraphClient(api_key='your-api-key')
        pool = client.get_pool("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")
        ticks = client.get_all_ticks("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: int = 30
    ):
        """
        Initialize the Uniswap V3 Graph client.

        Args:
            api_key: Optional API key for authenticated requests
            endpoint: Optional custom endpoint. Defaults to UNISWAP_V3_ENDPOINT
            timeout: Request timeout in seconds
        """
        if endpoint is None:
            endpoint = UNISWAP_V3_ENDPOINT
        super().__init__(endpoint=endpoint, api_key=api_key, timeout=timeout)

    def get_pool(self, pool_address: str) -> Dict[str, Any]:
        """
        Fetch pool data for a given pool address.

        Args:
            pool_address: Pool contract address (case-insensitive)

        Returns:
            Dictionary containing pool data including:
            - id, feeTier, tick, liquidity, sqrtPrice
            - token0 and token1 information
        """
        query = f"""
        {{
          pool(id: "{pool_address.lower()}") {{
            id
            feeTier
            tick
            liquidity
            sqrtPrice
            totalValueLockedUSD
            volumeUSD
            txCount
            createdAtTimestamp
            token0 {{
              id
              symbol
              name
              decimals
            }}
            token1 {{
              id
              symbol
              name
              decimals
            }}
          }}
        }}
        """
        data = self.query(query)
        return data.get('pool', {})

    def get_pools(
        self,
        first: int = 50,
        skip: int = 0,
        order_by: str = "totalValueLockedUSD",
        order_direction: str = "desc",
        where: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Fetch multiple pools with optional filtering.

        Args:
            first: Number of results to return (max 1000)
            skip: Number of results to skip
            order_by: Field to order by
            order_direction: 'asc' or 'desc'
            where: Optional filter conditions (e.g., {'feeTier': '3000'})

        Returns:
            DataFrame with pool data
        """
        where_clause = ""
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, str):
                    conditions.append(f'{key}: "{value}"')
                elif isinstance(value, (int, float)):
                    conditions.append(f"{key}: {value}")
                elif isinstance(value, bool):
                    conditions.append(f"{key}: {str(value).lower()}")
            if conditions:
                where_clause = f", where: {{{', '.join(conditions)}}}"

        query = f"""
        {{
          pools(
            first: {first}
            skip: {skip}
            orderBy: {order_by}
            orderDirection: {order_direction}
            {where_clause}
          ) {{
            id
            feeTier
            tick
            liquidity
            sqrtPrice
            totalValueLockedUSD
            volumeUSD
            txCount
            createdAtTimestamp
            token0 {{ id symbol name decimals }}
            token1 {{ id symbol name decimals }}
          }}
        }}
        """
        return self.query_to_dataframe(query, key='pools')

    def get_ticks(
        self,
        pool_address: str,
        skip: int = 0,
        first: int = 1000,
        liquidity_net_not_zero: bool = True
    ) -> list:
        """
        Fetch initialized ticks for a pool.

        Args:
            pool_address: Pool contract address
            skip: Number of results to skip
            first: Number of results to return (max 1000)
            liquidity_net_not_zero: If True, only fetch ticks with liquidityNet != 0

        Returns:
            List of tick dictionaries with tickIdx, liquidityGross, liquidityNet
        """
        where_condition = f'poolAddress: "{pool_address.lower()}"'
        if liquidity_net_not_zero:
            where_condition += ', liquidityNet_not: "0"'

        query = f"""
        {{
          ticks(
            where: {{{where_condition}}}
            first: {first}
            skip: {skip}
            orderBy: tickIdx
            orderDirection: asc
          ) {{
            tickIdx
            liquidityGross
            liquidityNet
          }}
        }}
        """
        data = self.query(query)
        return data.get('ticks', [])

    def get_all_ticks(self, pool_address: str, liquidity_net_not_zero: bool = True) -> list:
        """
        Fetch all initialized ticks for a pool, handling pagination.

        Args:
            pool_address: Pool contract address
            liquidity_net_not_zero: If True, only fetch ticks with liquidityNet != 0

        Returns:
            List of all tick dictionaries
        """
        all_ticks = []
        skip = 0
        first = 1000

        while True:
            ticks = self.get_ticks(
                pool_address,
                skip=skip,
                first=first,
                liquidity_net_not_zero=liquidity_net_not_zero
            )
            if not ticks:
                break
            all_ticks.extend(ticks)
            if len(ticks) < first:
                break
            skip += first

        return all_ticks

    def get_positions(
        self,
        pool_address: Optional[str] = None,
        owner: Optional[str] = None,
        first: int = 100,
        skip: int = 0
    ) -> pd.DataFrame:
        """
        Fetch positions for a pool or owner.

        Args:
            pool_address: Optional pool address to filter positions
            owner: Optional owner address to filter positions
            first: Number of results to return
            skip: Number of results to skip

        Returns:
            DataFrame with position data
        """
        where_conditions = []
        if pool_address:
            where_conditions.append(f'pool: "{pool_address.lower()}"')
        if owner:
            where_conditions.append(f'owner: "{owner.lower()}"')

        where_clause = ""
        if where_conditions:
            where_clause = f", where: {{{', '.join(where_conditions)}}}"

        query = f"""
        {{
          positions(
            first: {first}
            skip: {skip}
            {where_clause}
          ) {{
            id
            owner
            pool {{
              id
              token0 {{ symbol decimals }}
              token1 {{ symbol decimals }}
              feeTier
            }}
            tickLower
            tickUpper
            liquidity
            depositedToken0
            depositedToken1
            withdrawnToken0
            withdrawnToken1
            collectedFeesToken0
            collectedFeesToken1
            transaction {{
              timestamp
              blockNumber
            }}
          }}
        }}
        """
        return self.query_to_dataframe(query, key='positions')

    def get_swaps(
        self,
        pool_address: Optional[str] = None,
        first: int = 100,
        skip: int = 0,
        order_by: str = "timestamp",
        order_direction: str = "desc"
    ) -> pd.DataFrame:
        """
        Fetch swaps for a pool.

        Args:
            pool_address: Optional pool address to filter swaps
            first: Number of results to return
            skip: Number of results to skip
            order_by: Field to order by
            order_direction: 'asc' or 'desc'

        Returns:
            DataFrame with swap data
        """
        where_clause = ""
        if pool_address:
            where_clause = f', where: {{pool: "{pool_address.lower()}"}}'

        query = f"""
        {{
          swaps(
            first: {first}
            skip: {skip}
            orderBy: {order_by}
            orderDirection: {order_direction}
            {where_clause}
          ) {{
            id
            transaction {{
              id
              timestamp
              blockNumber
            }}
            pool {{
              id
              token0 {{ symbol }}
              token1 {{ symbol }}
            }}
            amount0
            amount1
            amountUSD
            sqrtPriceX96
            tick
          }}
        }}
        """
        return self.query_to_dataframe(query, key='swaps')

    # Utility functions for Uniswap V3 calculations

    @staticmethod
    def get_tick_spacing(fee_tier: int) -> int:
        """
        Get tick spacing for a given fee tier.

        Args:
            fee_tier: Fee tier in basis points (e.g., 3000 for 0.3%)

        Returns:
            Tick spacing value
        """
        TICK_SPACINGS = {
            100: 1,      # 0.01% fee
            500: 10,     # 0.05% fee
            3000: 60,    # 0.3% fee
            10000: 200   # 1% fee
        }
        return TICK_SPACINGS.get(fee_tier, 60)  # Default to 60 if not found

    @staticmethod
    def tick_to_price(tick: int, token0_decimals: int, token1_decimals: int) -> float:
        """
        Convert tick to price.

        Price = 1.0001^tick * (10^token1_decimals / 10^token0_decimals)

        Args:
            tick: Tick index
            token0_decimals: Decimals for token0
            token1_decimals: Decimals for token1

        Returns:
            Price of token0 in terms of token1
        """
        price = 1.0001 ** tick
        # Adjust for token decimals
        dec_factor = (10 ** token1_decimals) / (10 ** token0_decimals)
        return price * dec_factor

    def calculate_active_liquidity(
        self,
        pool_data: Dict[str, Any],
        ticks: List[Dict[str, Any]],
        num_ticks_display: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Calculate active liquidity for ticks around the current price.

        Based on the Uniswap documentation approach:
        1. Find the initializable tick closest to current tick
        2. Start from current tick with pool liquidity
        3. Iterate outward, adding/subtracting liquidityNet at each initialized tick

        Args:
            pool_data: Pool data dictionary from get_pool()
            ticks: List of tick dictionaries from get_all_ticks()
            num_ticks_display: Number of ticks to calculate in each direction

        Returns:
            List of processed tick dictionaries with active liquidity calculated
        """
        # Tick math constants (from Uniswap V3)
        MIN_TICK = -887272
        MAX_TICK = 887272

        fee_tier = int(pool_data.get('feeTier', 3000))
        tick_spacing = self.get_tick_spacing(fee_tier)
        tick_current = int(pool_data.get('tick', 0))
        liquidity_current = int(pool_data.get('liquidity', 0))

        token0_decimals = int(pool_data.get('token0', {}).get('decimals', 18))
        token1_decimals = int(pool_data.get('token1', {}).get('decimals', 18))

        # Create dictionary for quick lookup
        tick_dict = {int(tick['tickIdx']): tick for tick in ticks}

        # Find initializable tick closest to current price
        active_tick_idx = (tick_current // tick_spacing) * tick_spacing

        # Initialize active tick
        active_tick_processed = {
            'tickIdx': active_tick_idx,
            'liquidityActive': liquidity_current,
            'liquidityNet': 0,
            'price0': self.tick_to_price(active_tick_idx, token0_decimals, token1_decimals),
            'price1': self.tick_to_price(active_tick_idx, token1_decimals, token0_decimals),
            'isCurrent': True
        }

        # If current tick is initialized, set liquidityNet
        if active_tick_idx in tick_dict:
            active_tick_processed['liquidityNet'] = int(tick_dict[active_tick_idx]['liquidityNet'])

        processed_ticks = []
        previous_tick = active_tick_processed.copy()

        # Iterate forward (higher ticks)
        subsequent_ticks = []
        for i in range(num_ticks_display):
            current_tick_idx = previous_tick['tickIdx'] + tick_spacing

            if current_tick_idx > MAX_TICK:
                break

            current_tick_processed = {
                'tickIdx': current_tick_idx,
                'liquidityActive': previous_tick['liquidityActive'],
                'liquidityNet': 0,
                'price0': self.tick_to_price(current_tick_idx, token0_decimals, token1_decimals),
                'price1': self.tick_to_price(current_tick_idx, token1_decimals, token0_decimals),
                'isCurrent': False
            }

            # If tick is initialized, update liquidity
            if current_tick_idx in tick_dict:
                current_tick_processed['liquidityNet'] = int(tick_dict[current_tick_idx]['liquidityNet'])
                # Add liquidity when moving forward
                current_tick_processed['liquidityActive'] = previous_tick['liquidityActive'] + \
                    current_tick_processed['liquidityNet']

            subsequent_ticks.append(current_tick_processed)
            previous_tick = current_tick_processed

        # Iterate backward (lower ticks)
        # When moving backward, we need to work in reverse order and subtract liquidity
        previous_ticks = []
        previous_tick = active_tick_processed.copy()

        for i in range(num_ticks_display):
            current_tick_idx = previous_tick['tickIdx'] - tick_spacing

            if current_tick_idx < MIN_TICK:
                break

            current_tick_processed = {
                'tickIdx': current_tick_idx,
                'liquidityActive': previous_tick['liquidityActive'],
                'liquidityNet': 0,
                'price0': self.tick_to_price(current_tick_idx, token0_decimals, token1_decimals),
                'price1': self.tick_to_price(current_tick_idx, token1_decimals, token0_decimals),
                'isCurrent': False
            }

            # If tick is initialized, update liquidity
            # When moving backward (to lower ticks), we subtract the liquidityNet
            # because liquidityNet represents the change when crossing from left to right
            if current_tick_idx in tick_dict:
                liquidity_net = int(tick_dict[current_tick_idx]['liquidityNet'])
                current_tick_processed['liquidityNet'] = liquidity_net
                # Subtract because we're moving in the opposite direction
                current_tick_processed['liquidityActive'] = previous_tick['liquidityActive'] - liquidity_net

            previous_ticks.insert(0, current_tick_processed)  # Insert at beginning to maintain order
            previous_tick = current_tick_processed

        # Combine all ticks
        all_processed_ticks = previous_ticks + [active_tick_processed] + subsequent_ticks

        return all_processed_ticks
