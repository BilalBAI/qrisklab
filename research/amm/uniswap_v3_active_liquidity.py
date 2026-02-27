"""
Uniswap V3 Active Liquidity Visualization

This script follows the Uniswap documentation guide:
https://docs.uniswap.org/sdk/v3/guides/advanced/active-liquidity

It demonstrates:
1. Getting tickSpacing and currently active Tick from the Pool
2. Calculating active liquidity from net liquidity
3. Drawing a chart from the Tick data
"""

from qrisklab.clients.graph import UniV3GraphClient
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
from typing import Dict, List, Optional
from decimal import Decimal

# Try to load from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    try:
        load_dotenv(dotenv_path=env_path)
    except (PermissionError, FileNotFoundError):
        # Fallback: try loading from current directory
        try:
            load_dotenv()
        except:
            pass
except ImportError:
    pass

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Note: Utility functions have been moved to UniV3GraphClient:
# - get_tick_spacing() -> client.get_tick_spacing()
# - tick_to_price() -> client.tick_to_price()
# - calculate_active_liquidity() -> client.calculate_active_liquidity()


def visualize_liquidity(processed_ticks: List[Dict], pool_data: Dict):
    """Draw a chart visualizing the liquidity density."""
    # Convert to format suitable for plotting
    chart_data = []
    for tick in processed_ticks:
        chart_data.append({
            'tickIdx': tick['tickIdx'],
            'liquidityActive': float(tick['liquidityActive']),
            'price0': tick['price0'],
            'isCurrent': tick.get('isCurrent', False)
        })

    df = pd.DataFrame(chart_data)

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color bars: red for current tick, blue for others
    colors = ['#F51E87' if is_current else '#2172E5'
              for is_current in df['isCurrent']]

    # Create bar chart
    bars = ax.bar(df['tickIdx'], df['liquidityActive'],
                  color=colors, width=df['tickIdx'].diff().abs().fillna(1))

    # Highlight current tick
    current_tick_idx = df[df['isCurrent']]['tickIdx'].values
    if len(current_tick_idx) > 0:
        ax.axvline(x=current_tick_idx[0], color='red', linestyle='--',
                   linewidth=2, label='Current Tick')

    # Labels and title
    token0_symbol = pool_data.get('token0', {}).get('symbol', 'Token0')
    token1_symbol = pool_data.get('token1', {}).get('symbol', 'Token1')

    ax.set_xlabel('Tick Index', fontsize=12)
    ax.set_ylabel('Active Liquidity', fontsize=12)
    ax.set_title(f'Active Liquidity Distribution: {token0_symbol}/{token1_symbol}',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Format y-axis to show large numbers
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x / 1e18:.2f}'))

    plt.tight_layout()

    # Save the plot instead of showing (for headless environments)
    output_path = Path(__file__).parent / 'uniswap_v3_active_liquidity_chart.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Chart saved to: {output_path}")

    # Try to show if display is available
    try:
        plt.show()
    except:
        pass  # No display available

    plt.close()

    return fig


def main():
    """Main function to demonstrate active liquidity calculation."""
    # Configuration
    API_KEY = os.getenv('GRAPH_API_KEY')
    if not API_KEY:
        print("ERROR: GRAPH_API_KEY not found in environment variables.")
        print("Please set it in your .env file.")
        return

    # Example pool address (WETH/USDC 0.3% pool on Ethereum mainnet)
    # You can change this to any Uniswap V3 pool address
    POOL_ADDRESS = "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8"  # WETH/USDC 0.3%

    print("=" * 80)
    print("Uniswap V3 Active Liquidity Visualization")
    print("=" * 80)
    print(f"\nPool Address: {POOL_ADDRESS}")

    # Initialize client
    client = UniV3GraphClient(api_key=API_KEY)

    # Step 1: Get pool data (tickSpacing and current tick)
    print("\nStep 1: Fetching pool data...")
    pool_data = client.get_pool(POOL_ADDRESS)

    if not pool_data:
        print(f"ERROR: Pool {POOL_ADDRESS} not found.")
        return

    fee_tier = int(pool_data.get('feeTier', 3000))
    tick_spacing = client.get_tick_spacing(fee_tier)
    tick_current = int(pool_data.get('tick', 0))
    liquidity_current = int(pool_data.get('liquidity', 0))

    token0_symbol = pool_data.get('token0', {}).get('symbol', 'Token0')
    token1_symbol = pool_data.get('token1', {}).get('symbol', 'Token1')

    print(f"  Pool: {token0_symbol}/{token1_symbol}")
    print(f"  Fee Tier: {fee_tier / 10000}%")
    print(f"  Tick Spacing: {tick_spacing}")
    print(f"  Current Tick: {tick_current}")
    print(f"  Current Liquidity: {liquidity_current}")

    # Step 2: Fetch initialized ticks
    print("\nStep 2: Fetching initialized ticks...")
    ticks = client.get_all_ticks(POOL_ADDRESS)
    print(f"  Found {len(ticks)} initialized ticks")

    # Step 3: Calculate active liquidity
    print("\nStep 3: Calculating active liquidity...")
    processed_ticks = client.calculate_active_liquidity(pool_data, ticks, num_ticks_display=100)
    print(f"  Processed {len(processed_ticks)} ticks for visualization")

    # Display some statistics
    if processed_ticks:
        current_tick_data = next((t for t in processed_ticks if t.get('isCurrent')), None)
        if current_tick_data:
            print(f"\n  Current Tick Info:")
            print(f"    Tick Index: {current_tick_data['tickIdx']}")
            print(f"    Active Liquidity: {current_tick_data['liquidityActive']}")
            print(f"    Price ({token0_symbol}/{token1_symbol}): {current_tick_data['price0']:.6f}")
            print(f"    Price ({token1_symbol}/{token0_symbol}): {current_tick_data['price1']:.6f}")

        max_liquidity_tick = max(processed_ticks, key=lambda x: x['liquidityActive'])
        print(f"\n  Max Liquidity Tick:")
        print(f"    Tick Index: {max_liquidity_tick['tickIdx']}")
        print(f"    Active Liquidity: {max_liquidity_tick['liquidityActive']}")

    # Step 4: Visualize
    print("\nStep 4: Drawing chart...")
    try:
        fig = visualize_liquidity(processed_ticks, pool_data)
        print("  Chart displayed successfully!")
    except Exception as e:
        print(f"  Error displaying chart: {e}")
        print("  Make sure matplotlib is installed: pip install matplotlib")

    print("\n" + "=" * 80)
    print("Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
