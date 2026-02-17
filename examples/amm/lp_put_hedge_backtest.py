"""
Backtest: ETHUSDT Uniswap V3 LP + Put Hedge Strategy

Strategy:
- LP position with configurable range around entry price
- Put option at strike = range lower bound
- Put quantity = ETH exposure when price falls below range (full conversion to ETH)

Improvement levers when results are bearish:
- spot_drift: positive = bullish paths (e.g., 0.5 = 50% annual)
- rebalance_on_out_of_range: False = hold through drawdowns, puts protect better
- wider range (±20%): less IL, fewer rebalances
"""

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from qrisklab.core.amm_backtest import (
    LPPutHedgeStrategy,
    run_backtest_from_simulation,
)

# Improvement levers (tune to reduce bearish bias)
SPOT_DRIFT = 0.5            # Annual drift: 0 = flat, 0.5 = 50% expected return
REBALANCE = True             # Recenter when out of range (with delay)
REBALANCE_DELAY_HOURS = 3   # Rebalance only after 3+ consecutive hours out of range
RANGE_PCT = 0.20            # ±20% range (wider = less IL, fewer rebalances)
PUT_STRIKE_PCT = 0.80       # 80% = 20% OTM (cheaper puts, less theta)
PUT_EXPIRY_DAYS = 14        # Shorter tenor = less theta decay
STEP_INTERVAL_HOURS = 1.0   # Hourly simulation (realistic per-step price moves)


def run_backtest():
    """Run backtest with joint spot + implied vol (OU) simulation."""
    print("=" * 80)
    print("ETHUSDT Uniswap V3 LP + Put Hedge Backtest")
    print("=" * 80)
    print(f"\nConfig: hourly sim, drift={SPOT_DRIFT:.0%}, range=±{RANGE_PCT:.0%}")
    print(f"  Rebalance: after {REBALANCE_DELAY_HOURS}h out of range")
    print(f"  Put: strike {PUT_STRIKE_PCT:.0%} spot, {PUT_EXPIRY_DAYS}d tenor")
    print("  Implied vol: OU process per step")
    print()

    # Strategy parameters (target $10k initial portfolio: LP + puts)
    initial_portfolio_usd = 10_000.0
    initial_price = 3000.0
    # Base 50/50 allocation; scale to hit target initial portfolio
    base_eth, base_usdt = initial_portfolio_usd / 2 / initial_price, initial_portfolio_usd / 2
    _strat = LPPutHedgeStrategy(range_lower_pct=-RANGE_PCT, range_upper_pct=RANGE_PCT,
                               put_strike_pct=PUT_STRIKE_PCT, initial_eth=base_eth, initial_usdt=base_usdt,
                               vol=0.6, put_expiry_days=PUT_EXPIRY_DAYS)
    _r = run_backtest_from_simulation(initial_price, 0.6, 2, 365, datetime(2025, 1, 15), _strat,
                                      num_paths=1, use_spot_vol=True, rebalance_on_out_of_range=REBALANCE,
                                      rebalance_delay_steps=REBALANCE_DELAY_HOURS, spot_drift=SPOT_DRIFT,
                                      random_seed=42)
    scale = initial_portfolio_usd / _r["portfolio_value"].iloc[0]
    strategy = LPPutHedgeStrategy(
        range_lower_pct=-RANGE_PCT,
        range_upper_pct=RANGE_PCT,
        put_strike_pct=PUT_STRIKE_PCT,
        initial_eth=base_eth * scale,
        initial_usdt=base_usdt * scale,
        vol=0.6,
        put_expiry_days=PUT_EXPIRY_DAYS
    )

    num_paths = 20
    results = run_backtest_from_simulation(
        initial_price=initial_price,
        volatility=strategy.vol,
        num_steps=100,
        time_horizon_days=365,
        start_datetime=datetime(2025, 1, 15),
        strategy=strategy,
        num_paths=num_paths,
        use_spot_vol=True,
        rebalance_on_out_of_range=REBALANCE,
        rebalance_delay_steps=REBALANCE_DELAY_HOURS,
        spot_drift=SPOT_DRIFT,
        step_interval_hours=STEP_INTERVAL_HOURS,
        random_seed=42
    )

    # Save to CSV
    output_path = Path(__file__).parent / "lp_put_hedge_backtest_output.csv"
    results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Display results (path 0 sample)
    has_paths = 'path_id' in results.columns
    path0 = results[results['path_id'] == 0] if has_paths else results
    display_cols = ['datetime', 'price', 'implied_vol', 'lp_value', 'put_value', 'portfolio_value',
                    'delta_lp', 'delta_put', 'theta_approx', 'in_range', 'rebalanced']
    print(f"\nPath 0 — first 10 steps:")
    print(path0[display_cols].head(10))

    print(f"\nPath 0 — last 10 steps:")
    print(path0[display_cols].tail(10))

    # Per-path summary: initial and final portfolio value
    if has_paths:
        path_summary = results.groupby('path_id').agg(
            initial_value=('portfolio_value', 'first'),
            final_value=('portfolio_value', 'last'),
            rebalances=('rebalanced', 'sum'),
        ).reset_index()
        path_summary['return_pct'] = (path_summary['final_value'] / path_summary['initial_value'] - 1) * 100

        print(f"\nSummary across {num_paths} paths:")
        print(f"  Initial portfolio (mean): ${path_summary['initial_value'].mean():,.2f}")
        print(f"  Final portfolio — mean:   ${path_summary['final_value'].mean():,.2f}")
        print(f"  Final portfolio — min:    ${path_summary['final_value'].min():,.2f}")
        print(f"  Final portfolio — max:    ${path_summary['final_value'].max():,.2f}")
        print(f"  Return — mean:            {path_summary['return_pct'].mean():.2f}%")
        print(f"  Return — median:          {path_summary['return_pct'].median():.2f}%")
        print(f"  Return — min:             {path_summary['return_pct'].min():.2f}%")
        print(f"  Return — max:             {path_summary['return_pct'].max():.2f}%")
        print(f"  Rebalances per path (mean): {path_summary['rebalances'].mean():.1f}")
    else:
        initial_portfolio = results['portfolio_value'].iloc[0]
        final_portfolio = results['portfolio_value'].iloc[-1]
        print(f"\nSummary:")
        print(f"  Initial portfolio value: ${initial_portfolio:,.2f}")
        print(f"  Final portfolio value:   ${final_portfolio:,.2f}")
        print(f"  Return:                  {(final_portfolio/initial_portfolio - 1)*100:.2f}%")

    # PnL breakdown (across all steps, excl. first)
    valid_delta = results['delta_lp'].notna()
    if valid_delta.any():
        print(f"\nPnL breakdown (step-to-step, all paths):")
        print(f"  Mean delta_lp per step:    ${results.loc[valid_delta, 'delta_lp'].mean():,.2f}")
        print(f"  Mean delta_put per step:   ${results.loc[valid_delta, 'delta_put'].mean():,.2f}")
        print(f"  Mean theta_approx per step: ${results.loc[valid_delta, 'theta_approx'].mean():,.2f}")

    # Chart: underlying price paths + portfolio value paths
    if has_paths:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        for path_id in results['path_id'].unique():
            path_df = results[results['path_id'] == path_id].sort_values('step')
            steps = path_df['step'].values
            ax1.plot(steps, path_df['price'].values, alpha=0.7, linewidth=1)
            ax2.plot(steps, path_df['portfolio_value'].values, alpha=0.7, linewidth=1)
        ax1.set_ylabel('ETH price ($)')
        ax1.set_title('Simulated underlying price')
        ax1.grid(True, alpha=0.3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Portfolio value ($)')
        ax2.set_title('Portfolio value')
        ax2.grid(True, alpha=0.3)
        fig.suptitle(f'LP + Put Hedge Backtest — {num_paths} Paths')
        fig.tight_layout()
        chart_path = Path(__file__).parent / "lp_put_hedge_portfolio_paths.png"
        fig.savefig(chart_path, dpi=150)
        plt.close()
        print(f"\nChart saved to: {chart_path}")

    print("\n" + "=" * 80)
    return results


def run_comparison():
    """Run baseline vs improved config for comparison."""
    from qrisklab.core.amm_backtest import run_backtest_from_simulation

    initial_price = 3000.0
    init_eth, init_usdt = 10_000 / 2 / initial_price, 10_000 / 2
    strategy_base = LPPutHedgeStrategy(
        range_lower_pct=-0.05, range_upper_pct=0.05, put_strike_pct=0.95,
        initial_eth=init_eth, initial_usdt=init_usdt, vol=0.6, put_expiry_days=30
    )
    strategy_improved = LPPutHedgeStrategy(
        range_lower_pct=-0.20, range_upper_pct=0.20, put_strike_pct=0.80,
        initial_eth=init_eth, initial_usdt=init_usdt, vol=0.6, put_expiry_days=14
    )
    common = dict(initial_price=3000.0, volatility=0.6, num_steps=100,
                  time_horizon_days=365, start_datetime=datetime(2025, 1, 15),
                  num_paths=20, use_spot_vol=True, random_seed=42)

    print("Comparison: baseline (drift=0, rebalance, ±5%) vs improved (drift=50%, no rebal, ±20%)")
    print("-" * 60)
    for label, kwargs in [
        ("Baseline", {**common, "strategy": strategy_base, "spot_drift": 0.0, "rebalance_on_out_of_range": True}),
        ("Improved",  {**common, "strategy": strategy_improved, "spot_drift": 0.5, "rebalance_on_out_of_range": False}),
    ]:
        r = run_backtest_from_simulation(**kwargs)
        ps = r.groupby("path_id")["portfolio_value"].agg(first="first", last="last")
        ret = (ps["last"] / ps["first"] - 1) * 100
        print(f"{label:12}  return mean={ret.mean():+.1f}%  median={ret.median():+.1f}%  min={ret.min():+.1f}%  max={ret.max():+.1f}%")
    print("-" * 60)


if __name__ == "__main__":
    run_backtest()
    # run_comparison()  # Uncomment to compare baseline vs improved (runs 2 extra backtests)
