"""
Monte Carlo simulation demo for AMM LP positions.

This script demonstrates how to:
1. Simulate ETH price paths using geometric Brownian motion
2. Run AMM LP valuations along the price paths
3. Analyze the results
"""

from datetime import datetime
import pandas as pd
from qrisklab.core.amm_mc_simulation import (
    run_monte_carlo_analysis,
    summarize_mc_results
)

# Example AMM position data
amm_positions = [{
    'lower_price': 2980.97,      # Lower bound
    'upper_price': 3463.36,      # Upper bound
    'initial_price': 3212.8,     # Initial price
    'initial_amount0': 0.25,     # 0.25 ETH
    'initial_amount1': 801.13,   # 801 USDT
    'decimals0': 18,
    'decimals1': 6
}]

# Example option hedge position data
option_hedge_positions = [{
    'strike': 3000,
    'expiry': '27FEB26',
    'put_call': 'put',
    'vol': 0.5576,
    'rate': 0.0
}]


def run_single_path_demo():
    """Demonstrate single path simulation."""
    print("=" * 80)
    print("Monte Carlo Simulation - Single Path Demo")
    print("=" * 80)

    # Simulation parameters
    initial_price = 3200.0  # Starting ETH price in USDT
    drift = 0.0  # 0% annual drift (neutral expectation)
    volatility = 0.8  # 80% annual volatility (typical for crypto)
    num_steps = 100  # 100 time steps
    time_horizon_days = 365  # 1 year horizon
    start_datetime = datetime(2025, 1, 15, 0, 0, 0)

    # Run simulation
    results = run_monte_carlo_analysis(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        num_steps=num_steps,
        time_horizon_days=time_horizon_days,
        start_datetime=start_datetime,
        amm_positions=amm_positions,
        option_hedge_positions=option_hedge_positions,
        num_paths=1,
        random_seed=42
    )

    # Display results
    print(f"\nSimulated {len(results)} time steps")
    print(f"\nFirst few steps:")
    print(results[['datetime', 'price', 'total_portfolio_value', 'amm_in_range']].head(10))

    print(f"\nLast few steps:")
    print(results[['datetime', 'price', 'total_portfolio_value', 'amm_in_range']].tail(10))

    print(f"\nSummary Statistics:")
    print(f"  Initial Portfolio Value: ${results.iloc[0]['total_portfolio_value']:,.2f}")
    print(f"  Final Portfolio Value: ${results.iloc[-1]['total_portfolio_value']:,.2f}")
    print(f"  Min Portfolio Value: ${results['total_portfolio_value'].min():,.2f}")
    print(f"  Max Portfolio Value: ${results['total_portfolio_value'].max():,.2f}")
    print(f"  Min Price: ${results['price'].min():,.2f}")
    print(f"  Max Price: ${results['price'].max():,.2f}")

    return results


def run_multiple_paths_demo():
    """Demonstrate multiple paths simulation with statistics."""
    print("\n" + "=" * 80)
    print("Monte Carlo Simulation - Multiple Paths Demo")
    print("=" * 80)

    # Simulation parameters
    initial_price = 3200.0  # Starting ETH price in USDT
    drift = 0.0  # 0% annual drift
    volatility = 0.8  # 80% annual volatility
    num_steps = 50  # 50 time steps (fewer for faster computation)
    time_horizon_days = 365  # 1 year horizon
    start_datetime = datetime(2026, 1, 1, 0, 0, 0)
    num_paths = 10  # Simulate 10 independent paths

    # Run simulation
    results = run_monte_carlo_analysis(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        num_steps=num_steps,
        time_horizon_days=time_horizon_days,
        start_datetime=start_datetime,
        amm_positions=amm_positions,
        option_hedge_positions=option_hedge_positions,
        num_paths=num_paths,
        random_seed=42
    )

    # Summarize results
    summary = summarize_mc_results(results, path_id_col='path_id')

    print(f"\nSimulated {num_paths} paths with {num_steps + 1} time steps each")
    print(f"\nSummary Statistics by Time Step:")
    print(f"\nFirst few steps:")
    cols_to_show = [
        'step', 'datetime', 'price_mean', 'price_std',
        'total_portfolio_value_mean', 'total_portfolio_value_std',
        'total_portfolio_value_min', 'total_portfolio_value_max'
    ]
    print(summary[cols_to_show].head(10))

    print(f"\nLast few steps:")
    print(summary[cols_to_show].tail(10))

    print(f"\nPortfolio Value Statistics Across All Paths:")
    print(f"  Initial Mean Portfolio Value: ${summary.iloc[0]['total_portfolio_value_mean']:,.2f}")
    print(f"  Final Mean Portfolio Value: ${summary.iloc[-1]['total_portfolio_value_mean']:,.2f}")
    print(f"  Final Min Portfolio Value: ${summary.iloc[-1]['total_portfolio_value_min']:,.2f}")
    print(f"  Final Max Portfolio Value: ${summary.iloc[-1]['total_portfolio_value_max']:,.2f}")
    print(f"  Final 5th Percentile: ${summary.iloc[-1]['total_portfolio_value_p5']:,.2f}")
    print(f"  Final 95th Percentile: ${summary.iloc[-1]['total_portfolio_value_p95']:,.2f}")

    return results, summary


if __name__ == "__main__":
    # Run single path demo
    single_results = run_single_path_demo()

    # Run multiple paths demo
    multiple_results, summary = run_multiple_paths_demo()

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
