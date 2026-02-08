# QRiskLib

qrisklab is a python-based Quantitative Risk Analytics Toolkit including:
* stress testing tools designed to evaluate portfolio risk under complex stress scenarios 
* vol curves fitting models (WingModel, SVIModel)
* option pricing, greeks calcualtion, greeks pnl attribution
* visualization tools
* AMM (Automated Market Maker) position valuation and analysis
* Uniswap V3 active liquidity calculation and visualization
* Graph Protocol client for querying blockchain and DeFi data

## Installation

Run the following to create a qrisklab conda environment:

```bash
conda create -n qrisklab python=3.12 pip wheel
conda activate qrisklab

# Download and install package
python -m pip install -e .
# Or install directly
pip install git+https://github.com/BilalBAI/qrisklab.git

```

Run the following to create a qrisklab venv:

```bash
# Create the virtual environment with Python 3.12
which python3.12
python3.12 -m venv qrisklab_env

# Activate the virtual environment
# Windows
qrisklab_env\Scripts\activate
# macOS / Linux
source qrisklab_env/bin/activate

# Verify the Python version
python --version

# Download and install package
python -m pip install -e .
# Or install directly
pip install git+https://github.com/BilalBAI/qrisklab.git

# Deactivate the virtual environment when done
deactivate
```

### Bloomberg (BBG Terminal should be logged in and API enabled)
```ini
python -m pip install --index-url=https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi

pip install blp

pip install sqlalchemy
```

In the `.env` you need the following keys:

```ini
# Dir to a sqlite db for BBG Cached data. e.g. C:/qrisklab/db/ 
# Default to ./ if not provided
# Auto create new db if db doesn't exist
BBG_CACHE_DB_DIR=<DIR>

# The Graph Protocol API key for querying blockchain data
# Get your API key from: https://thegraph.com/studio/apikeys/
GRAPH_API_KEY=<your-api-key>
```

## Graph Protocol Client

qrisklab includes a Graph Protocol client for querying blockchain and DeFi data, with specialized support for Uniswap V3.

### Basic Usage

```python
from qrisklab.clients.graph import GraphClient, UniV3GraphClient

# Generic Graph client for any subgraph
client = GraphClient(
    endpoint='https://gateway.thegraph.com/api/subgraphs/id/YOUR_DEPLOYMENT_ID',
    api_key='your-api-key'
)

# Uniswap V3 specialized client (uses default Uniswap V3 endpoint)
uni_client = UniV3GraphClient(api_key='your-api-key')

# Query pools
pool = uni_client.get_pool("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")
pools_df = uni_client.get_pools(first=10, order_by="totalValueLockedUSD")

# Query ticks and calculate active liquidity
ticks = uni_client.get_all_ticks("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")
processed_ticks = uni_client.calculate_active_liquidity(pool, ticks)

# Query positions and swaps
positions_df = uni_client.get_positions(pool_address="0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")
swaps_df = uni_client.get_swaps(pool_address="0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")
```

### Uniswap V3 Active Liquidity

The `UniV3GraphClient` includes utilities for calculating and visualizing active liquidity in Uniswap V3 pools, following the [Uniswap documentation](https://docs.uniswap.org/sdk/v3/guides/advanced/active-liquidity).

Example script: `examples/uniswap_v3_active_liquidity.py`

```python
from qrisklab.clients.graph import UniV3GraphClient

client = UniV3GraphClient(api_key='your-api-key')
pool = client.get_pool("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")
ticks = client.get_all_ticks("0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8")

# Calculate active liquidity
processed_ticks = client.calculate_active_liquidity(pool, ticks, num_ticks_display=100)

# Utility methods
tick_spacing = client.get_tick_spacing(int(pool['feeTier']))
price = client.tick_to_price(100, 18, 6)  # tick, token0_decimals, token1_decimals
```

## AMM Position Valuation

qrisklab includes tools for valuing AMM (Automated Market Maker) positions, particularly Uniswap V3 LP positions:

```python
from qrisklab.core.amm_scenarios import amm_lp_valuation, calc_univ3_current_holdings

# Calculate current holdings in a Uniswap V3 position
holdings = calc_univ3_current_holdings(
    current_price=3200.0,
    lower_price=2980.97,
    upper_price=3463.36,
    initial_price=3212.8,
    initial_amount0=0.25,
    initial_amount1=801.13,
    decimals0=18,
    decimals1=6
)

# Value AMM LP positions with option hedges
amm_positions = [{
    'lower_price': 2980.97,
    'upper_price': 3463.36,
    'initial_price': 3212.8,
    'initial_amount0': 0.25,
    'initial_amount1': 801.13,
    'decimals0': 18,
    'decimals1': 6
}]

option_hedges = [{
    'strike': 3000,
    'expiry': '27FEB26',
    'put_call': 'put',
    'vol': 0.5576,
    'rate': 0.0
}]

result = amm_lp_valuation(
    current_price=3200.0,
    valuation_datetime="2026-01-20 00:00:00",
    amm_pos=amm_positions,
    option_hedge_pos=option_hedges
)
```

See `examples/amm/` for more examples including Monte Carlo simulation of AMM positions.