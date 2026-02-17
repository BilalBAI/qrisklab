"""
Backtest module for AMM LP strategies with option hedges.

Implements the LP + put hedge strategy:
- LP position with configurable range (e.g., -5% to +5%)
- Put option at strike = range lower bound (e.g., 95% of spot)
- Put quantity = ETH exposure when price is below range (full conversion to ETH)
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass

from .amm_calc import calc_univ3_current_holdings
from .black_scholes import bsm_pricing


@dataclass
class LPPutHedgeStrategy:
    """Strategy parameters for LP + put hedge backtest."""
    range_lower_pct: float = -0.05   # -5% range
    range_upper_pct: float = 0.05    # +5% range
    put_strike_pct: float = 0.95     # Strike at 95% of spot (= lower bound)
    initial_eth: float = 1.0         # Initial ETH to deposit
    initial_usdt: float = 3000.0     # Initial USDT to deposit
    vol: float = 0.6                 # Implied vol for put pricing
    rate: float = 0.0                # Risk-free rate
    put_expiry_days: int = 30        # Put option tenor
    decimals0: int = 18               # ETH decimals
    decimals1: int = 6               # USDT decimals
    # Fee model (volume-based, 0.3% pool)
    pool_fee_rate: float = 0.003            # 0.3% fee tier
    pool_daily_volume: float = 50_000_000   # $50M base daily volume
    pool_active_tvl: float = 30_000_000     # $30M active liquidity in similar ranges
    vol_volume_sensitivity: float = 1.5     # Volume scales with vol: 1 + sens*(vol/vol_lr - 1)
    vol_long_run: float = 0.6              # Long-run vol for volume scaling baseline


def _parse_datetime(dt) -> datetime:
    """Parse datetime from various formats."""
    if isinstance(dt, datetime):
        return dt
    dt_str = str(dt)
    if 'T' in dt_str:
        return datetime.strptime(dt_str[:19].replace('T', ' '), '%Y-%m-%d %H:%M:%S')
    return datetime.strptime(dt_str[:10], '%Y-%m-%d')


def _recenter_lp_and_put(
    total_capital: float,
    current_price: float,
    vol: float,
    dt_val: datetime,
    strategy: LPPutHedgeStrategy,
    max_iter: int = 20,
    tol: float = 1e-6
) -> tuple:
    """
    Solve for new LP amounts and put quantity when recentering.
    Accounting: total_capital = lp_value (from unwrap) + put_value (from sale).
    We allocate lp_capital to new LP and put_cost to new puts, with
    lp_capital + put_cost = total_capital. Surplus from put roll (sell > buy)
    goes to LP; deficit (sell < buy) comes from LP.
    Uses fixed-point iteration.
    """
    new_lower = current_price * (1 + strategy.range_lower_pct)
    new_upper = current_price * (1 + strategy.range_upper_pct)
    new_put_strike = current_price * strategy.put_strike_pct
    new_put_expiry = dt_val + timedelta(days=strategy.put_expiry_days)
    tte = _tte_years(new_put_expiry, dt_val)
    put_price_per_unit = bsm_pricing(
        strike=new_put_strike, time_to_expiry=tte, spot=current_price,
        rate=strategy.rate, vol=vol, put_call='put'
    )
    lp_capital = total_capital
    for _ in range(max_iter):
        new_amount0 = (lp_capital / 2) / current_price
        new_amount1 = lp_capital / 2
        put_qty = _compute_put_quantity(
            initial_price=current_price, lower_price=new_lower, upper_price=new_upper,
            initial_amount0=new_amount0, initial_amount1=new_amount1,
            decimals0=strategy.decimals0, decimals1=strategy.decimals1
        )
        put_cost = put_qty * put_price_per_unit
        new_lp_capital = total_capital - put_cost
        if abs(new_lp_capital - lp_capital) < tol:
            lp_capital = new_lp_capital
            break
        lp_capital = new_lp_capital
    # Final amounts from converged lp_capital
    new_amount0 = (lp_capital / 2) / current_price
    new_amount1 = lp_capital / 2
    put_qty = _compute_put_quantity(
        initial_price=current_price, lower_price=new_lower, upper_price=new_upper,
        initial_amount0=new_amount0, initial_amount1=new_amount1,
        decimals0=strategy.decimals0, decimals1=strategy.decimals1
    )
    return (current_price, new_lower, new_upper, new_amount0, new_amount1,
            put_qty, new_put_strike, new_put_expiry)


def _compute_put_quantity(
    initial_price: float,
    lower_price: float,
    upper_price: float,
    initial_amount0: float,
    initial_amount1: float,
    decimals0: int,
    decimals1: int
) -> float:
    """
    Compute ETH quantity when price is below range (all liquidity in ETH).
    This is the quantity to hedge with puts.
    """
    # Use price slightly below lower bound to ensure we're in "below range" regime
    price_below_range = lower_price * 0.999
    holdings = calc_univ3_current_holdings(
        current_price=price_below_range,
        lower_price=lower_price,
        upper_price=upper_price,
        initial_price=initial_price,
        initial_amount0=initial_amount0,
        initial_amount1=initial_amount1,
        decimals0=decimals0,
        decimals1=decimals1
    )
    return holdings.get('amount0', 0.0)


def _tte_years(expiry: datetime, dt_val: datetime) -> float:
    """Time to expiry in years, with sub-day precision."""
    delta = expiry - dt_val
    return max(0.0, delta.total_seconds() / (365.25 * 24 * 3600))


def run_lp_put_hedge_backtest(
    price_path_df: pd.DataFrame,
    strategy: Optional[LPPutHedgeStrategy] = None,
    path_id_col: Optional[str] = 'path_id',
    rebalance_on_out_of_range: bool = False,
    rebalance_delay_steps: int = 0
) -> pd.DataFrame:
    """
    Run backtest of LP + put hedge strategy along a price path.

    Args:
        price_path_df: DataFrame with columns: datetime, price (or spot), step.
                       Optional: vol (for path-dependent vol), path_id.
        strategy: Strategy parameters. If None, uses defaults.
        path_id_col: Column name for path id when multiple paths.
        rebalance_on_out_of_range: If True, recenter LP and roll put when out of range.
        rebalance_delay_steps: Rebalance only after this many consecutive steps out of range
                              (e.g., 3 for "out of range 3+ hours" with hourly steps).

    Returns:
        DataFrame with columns: datetime, step, price, lp_value, put_value,
        portfolio_value, put_quantity, in_range, rebalanced, (path_id if multi-path).
    """
    if strategy is None:
        strategy = LPPutHedgeStrategy()

    # Determine price column (spot or price)
    price_col = 'spot' if 'spot' in price_path_df.columns else 'price'

    initial_price = float(price_path_df[price_col].iloc[0])
    start_datetime = price_path_df['datetime'].iloc[0]

    # LP range
    lower_price = initial_price * (1 + strategy.range_lower_pct)
    upper_price = initial_price * (1 + strategy.range_upper_pct)

    # Put strike = 95% of spot (= lower bound for -5% range)
    put_strike = initial_price * strategy.put_strike_pct

    # Compute initial LP allocation (50/50 value split at entry, or use provided amounts)
    # For -5% to +5% range around 3000: we need amounts that satisfy the range
    initial_amount0 = strategy.initial_eth
    initial_amount1 = strategy.initial_usdt

    # Put quantity = ETH we'd hold when below range
    put_quantity = _compute_put_quantity(
        initial_price=initial_price,
        lower_price=lower_price,
        upper_price=upper_price,
        initial_amount0=initial_amount0,
        initial_amount1=initial_amount1,
        decimals0=strategy.decimals0,
        decimals1=strategy.decimals1
    )

    # Put expiry date
    if isinstance(start_datetime, str):
        start_dt = datetime.strptime(start_datetime[:10], '%Y-%m-%d')
    else:
        start_dt = start_datetime
    put_expiry = start_dt + timedelta(days=strategy.put_expiry_days)

    # Compute step size in years for fee accrual
    dts = price_path_df['datetime']
    if len(dts) > 1:
        dt0 = _parse_datetime(dts.iloc[0])
        dt1 = _parse_datetime(dts.iloc[1])
        step_years = (dt1 - dt0).total_seconds() / (365.25 * 24 * 3600)
    else:
        step_years = 1.0 / 365.25

    # Run backtest
    has_paths = path_id_col and path_id_col in price_path_df.columns
    path_ids = price_path_df[path_id_col].unique() if has_paths else [None]

    all_results = []
    for path_id in path_ids:
        if path_id is not None:
            path_data = price_path_df[price_path_df[path_id_col] == path_id]
        else:
            path_data = price_path_df

        # State carried across steps (updated on rebalance)
        lp_lower, lp_upper = lower_price, upper_price
        lp_initial_price = initial_price
        lp_amount0, lp_amount1 = initial_amount0, initial_amount1
        put_strike_cur = put_strike
        put_expiry_cur = put_expiry
        put_quantity_cur = put_quantity

        # For pnl_breakdown: previous step values (None on first step)
        prev_lp_value = None
        prev_put_value = None
        prev_price = None
        prev_dt = None
        prev_put_strike = None
        prev_put_expiry = None
        prev_put_quantity = None
        prev_vol = None
        consecutive_out_of_range_steps = 0
        cumulative_fees = 0.0

        for idx, row in path_data.iterrows():
            current_price = float(row[price_col])
            dt = row['datetime']
            step = row.get('step', idx)

            # Vol: use path-dependent if available, else strategy default
            vol = float(row['vol']) if 'vol' in price_path_df.columns and pd.notna(row.get('vol')) else strategy.vol
            dt_val = _parse_datetime(dt)

            # LP value using current position state
            holdings = calc_univ3_current_holdings(
                current_price=current_price,
                lower_price=lp_lower,
                upper_price=lp_upper,
                initial_price=lp_initial_price,
                initial_amount0=lp_amount0,
                initial_amount1=lp_amount1,
                decimals0=strategy.decimals0,
                decimals1=strategy.decimals1
            )
            lp_value = holdings.get('value_in_token1', 0.0)
            in_range = holdings.get('in_range', False)

            if in_range:
                consecutive_out_of_range_steps = 0
            else:
                consecutive_out_of_range_steps += 1

            rebalanced = False
            put_roll_cashflow = float('nan')
            delay_met = consecutive_out_of_range_steps >= rebalance_delay_steps
            if rebalance_on_out_of_range and not in_range and delay_met:
                # Recenter LP and roll put hedge
                # 1. Unwrap LP -> lp_value; 2. Sell old puts -> put_value_cur
                # 3. Total cash = lp_value + put_value_cur
                # 4. Allocate: lp_capital to new LP, put_cost to new puts
                #    lp_capital + put_cost = total. Surplus/deficit from put roll goes to/from LP.
                total_capital = lp_value
                tte_cur = _tte_years(put_expiry_cur, dt_val)
                put_value_cur = put_quantity_cur * bsm_pricing(
                    strike=put_strike_cur, time_to_expiry=tte_cur, spot=current_price,
                    rate=strategy.rate, vol=vol, put_call='put'
                )
                total_capital += put_value_cur
                (lp_initial_price, lp_lower, lp_upper, lp_amount0, lp_amount1,
                 put_quantity_cur, put_strike_cur, put_expiry_cur) = _recenter_lp_and_put(
                    total_capital, current_price, vol, dt_val, strategy
                )
                # Put roll cashflow: + = surplus to LP, - = drawn from LP
                put_cost = put_quantity_cur * bsm_pricing(
                    strike=put_strike_cur, time_to_expiry=_tte_years(put_expiry_cur, dt_val),
                    spot=current_price, rate=strategy.rate, vol=vol, put_call='put'
                )
                put_roll_cashflow = put_value_cur - put_cost
                holdings = calc_univ3_current_holdings(
                    current_price=current_price,
                    lower_price=lp_lower,
                    upper_price=lp_upper,
                    initial_price=lp_initial_price,
                    initial_amount0=lp_amount0,
                    initial_amount1=lp_amount1,
                    decimals0=strategy.decimals0,
                    decimals1=strategy.decimals1
                )
                lp_value = holdings.get('value_in_token1', 0.0)
                in_range = holdings.get('in_range', True)
                rebalanced = True
                consecutive_out_of_range_steps = 0

            # Fee accrual (volume-based, only when in range)
            step_fee = 0.0
            if in_range and strategy.pool_fee_rate > 0 and strategy.pool_daily_volume > 0:
                vol_ratio = vol / strategy.vol_long_run if strategy.vol_long_run > 0 else 1.0
                vol_adj = 1.0 + strategy.vol_volume_sensitivity * (vol_ratio - 1.0)
                vol_adj = max(vol_adj, 0.1)
                hourly_volume = strategy.pool_daily_volume / 24.0 * vol_adj
                step_volume = hourly_volume * (step_years * 365.25 * 24)
                liq_share = lp_value / strategy.pool_active_tvl if strategy.pool_active_tvl > 0 else 0.0
                step_fee = step_volume * strategy.pool_fee_rate * liq_share
            cumulative_fees += step_fee

            # Put value: time to expiry decreases along path
            time_to_expiry = _tte_years(put_expiry_cur, dt_val)

            put_price = bsm_pricing(
                strike=put_strike_cur,
                time_to_expiry=time_to_expiry,
                spot=current_price,
                rate=strategy.rate,
                vol=vol,
                put_call='put'
            )
            put_value = put_quantity_cur * put_price

            # PnL breakdown: delta_lp, delta_put, theta_approx (NaN on first step)
            delta_lp = float('nan')
            delta_put = float('nan')
            theta_approx = float('nan')
            if prev_lp_value is not None:
                delta_lp = lp_value - prev_lp_value
                delta_put = put_value - prev_put_value
                # Theta: put decay from prev to curr time, holding price at prev_price
                tte_prev = _tte_years(prev_put_expiry, prev_dt)
                tte_curr = _tte_years(prev_put_expiry, dt_val)
                put_val_at_prev_time = prev_put_quantity * bsm_pricing(
                    strike=prev_put_strike, time_to_expiry=tte_prev, spot=prev_price,
                    rate=strategy.rate, vol=prev_vol, put_call='put'
                )
                put_val_at_curr_time_same_price = prev_put_quantity * bsm_pricing(
                    strike=prev_put_strike, time_to_expiry=tte_curr, spot=prev_price,
                    rate=strategy.rate, vol=prev_vol, put_call='put'
                )
                theta_approx = float(put_val_at_curr_time_same_price - put_val_at_prev_time)

            result_row = {
                'datetime': dt,
                'step': step,
                'price': current_price,
                'lp_token0': holdings.get('amount0', 0.0),
                'lp_token1': holdings.get('amount1', 0.0),
                'lp_value': lp_value,
                'put_value': put_value,
                'fee_earned': step_fee,
                'cumulative_fees': cumulative_fees,
                'portfolio_value': lp_value + put_value + cumulative_fees,
                'delta_lp': delta_lp,
                'delta_put': delta_put,
                'theta_approx': theta_approx,
                'put_roll_cashflow': put_roll_cashflow,
                'put_quantity': put_quantity_cur,
                'in_range': in_range,
                'rebalanced': rebalanced,
                'put_strike': put_strike_cur,
                'put_expiry': put_expiry_cur,
                'put_expiry_days': strategy.put_expiry_days,
                'implied_vol': vol,
            }
            if has_paths:
                result_row[path_id_col] = path_id
            all_results.append(result_row)

            # Update prev for next step
            prev_lp_value = lp_value
            prev_put_value = put_value
            prev_price = current_price
            prev_dt = dt_val
            prev_put_strike = put_strike_cur
            prev_put_expiry = put_expiry_cur
            prev_put_quantity = put_quantity_cur
            prev_vol = vol

    return pd.DataFrame(all_results)


def run_backtest_from_simulation(
    initial_price: float,
    volatility: float,
    num_steps: int,
    time_horizon_days: int,
    start_datetime: datetime,
    strategy: Optional[LPPutHedgeStrategy] = None,
    num_paths: int = 1,
    use_spot_vol: bool = False,
    rebalance_on_out_of_range: bool = False,
    rebalance_delay_steps: int = 0,
    spot_drift: float = 0.0,
    step_interval_hours: Optional[float] = None,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Run backtest using simulated price paths.

    Args:
        initial_price: Starting ETH price
        volatility: Annual volatility
        num_steps: Number of steps (ignored if step_interval_hours is set)
        time_horizon_days: Horizon in days
        start_datetime: Start datetime
        strategy: Strategy params
        num_paths: Number of simulated paths
        use_spot_vol: If True, use joint spot+vol simulation (stochastic vol)
        rebalance_on_out_of_range: If True, recenter LP and roll put when out of range
        rebalance_delay_steps: Rebalance only after this many consecutive steps out of range
        spot_drift: Annual drift for spot (e.g., 0.5 = 50% expected return)
        step_interval_hours: If set, use hourly steps (e.g., 1.0 = hourly, realistic per-step moves)
        random_seed: Random seed

    Returns:
        Backtest results DataFrame
    """
    from .path_simulation import simulate_price_path, simulate_multiple_paths, simulate_spot_vol_paths

    if use_spot_vol:
        paths = simulate_spot_vol_paths(
            initial_spot=initial_price,
            initial_vol=volatility,
            num_steps=num_steps,
            time_horizon_days=time_horizon_days,
            start_datetime=start_datetime,
            spot_drift=spot_drift,
            num_paths=num_paths,
            step_interval_hours=step_interval_hours,
            random_seed=random_seed
        )
        # Rename spot to price for compatibility
        paths = paths.rename(columns={'spot': 'price'})
    else:
        if num_paths == 1:
            paths = simulate_price_path(
                initial_price=initial_price,
                drift=spot_drift,
                volatility=volatility,
                num_steps=num_steps,
                time_horizon_days=time_horizon_days,
                start_datetime=start_datetime,
                random_seed=random_seed
            )
        else:
            paths = simulate_multiple_paths(
                initial_price=initial_price,
                drift=spot_drift,
                volatility=volatility,
                num_steps=num_steps,
                time_horizon_days=time_horizon_days,
                start_datetime=start_datetime,
                num_paths=num_paths,
                random_seed=random_seed
            )

    return run_lp_put_hedge_backtest(
        price_path_df=paths,
        strategy=strategy,
        path_id_col='path_id' if num_paths > 1 else None,
        rebalance_on_out_of_range=rebalance_on_out_of_range,
        rebalance_delay_steps=rebalance_delay_steps
    )
