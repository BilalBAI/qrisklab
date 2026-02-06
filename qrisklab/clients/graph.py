"""
Simple Graph Protocol client for querying blockchain and DeFi data.

This client provides a simple interface to The Graph Protocol subgraphs.
"""

import requests
import pandas as pd
from typing import Dict, Any, Optional


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
