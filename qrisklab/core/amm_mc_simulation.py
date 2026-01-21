"""
Monte Carlo simulation for AMM LP positions with price path simulation.

This module provides functions to simulate ETH price paths using geometric
Brownian motion and evaluate AMM LP positions along those paths.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from .amm_scenarios import amm_lp_valuation


def simulate_price_path(
    initial_price: float,
    drift: float,
    volatility: float,
    num_steps: int,
    time_horizon_days: int,
    start_datetime: datetime,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate a single price path using geometric Brownian motion.

    Args:
        initial_price: Starting price
        drift: Annual drift rate (mu, e.g., 0.05 for 5% annual growth)
        volatility: Annual volatility (sigma, e.g., 0.6 for 60%)
        num_steps: Number of time steps in the simulation
        time_horizon_days: Total time horizon in days
        start_datetime: Starting datetime for the simulation
        random_seed: Optional random seed for reproducibility

    Returns:
        DataFrame with columns:
            - datetime: Timestamp for each step
            - price: Simulated price at each step
            - step: Step number (0 to num_steps-1)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Calculate time step size in years
    dt = (time_horizon_days / 365.0) / num_steps

    # Generate random shocks (standard normal distribution)
    random_shocks = np.random.standard_normal(num_steps)

    # Calculate price changes using geometric Brownian motion
    # S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    price_changes = np.exp(
        (drift - 0.5 * volatility ** 2) * dt +
        volatility * np.sqrt(dt) * random_shocks
    )

    # Calculate cumulative price path
    price_path = initial_price * np.cumprod(price_changes)
    price_path = np.insert(price_path, 0, initial_price)  # Include initial price

    # Generate datetime for each step
    datetime_index = [
        start_datetime + timedelta(days=time_horizon_days * i / num_steps)
        for i in range(num_steps + 1)
    ]

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': datetime_index,
        'price': price_path,
        'step': range(num_steps + 1)
    })

    return df


def simulate_multiple_paths(
    initial_price: float,
    drift: float,
    volatility: float,
    num_steps: int,
    time_horizon_days: int,
    start_datetime: datetime,
    num_paths: int,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate multiple price paths using geometric Brownian motion.

    Args:
        initial_price: Starting price
        drift: Annual drift rate (mu)
        volatility: Annual volatility (sigma)
        num_steps: Number of time steps in each simulation
        time_horizon_days: Total time horizon in days
        start_datetime: Starting datetime for the simulation
        num_paths: Number of independent paths to simulate
        random_seed: Optional random seed for reproducibility

    Returns:
        DataFrame with columns:
            - path_id: Path identifier (0 to num_paths-1)
            - datetime: Timestamp for each step
            - price: Simulated price at each step
            - step: Step number (0 to num_steps)
    """
    all_paths = []

    for path_id in range(num_paths):
        # Use different seed for each path if seed is provided
        path_seed = random_seed + path_id if random_seed is not None else None
        df_path = simulate_price_path(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            num_steps=num_steps,
            time_horizon_days=time_horizon_days,
            start_datetime=start_datetime,
            random_seed=path_seed
        )
        df_path['path_id'] = path_id
        all_paths.append(df_path)

    return pd.concat(all_paths, ignore_index=True)


def run_mc_valuation(
    price_path_df: pd.DataFrame,
    amm_positions: List[Dict],
    option_hedge_positions: List[Dict],
    path_id_col: str = 'path_id'
) -> pd.DataFrame:
    """
    Run AMM LP valuation along a price path or multiple paths.

    Args:
        price_path_df: DataFrame with price paths (must have 'datetime', 'price', and optionally path_id_col)
        amm_positions: List of AMM position dictionaries
        option_hedge_positions: List of option hedge position dictionaries
        path_id_col: Column name for path identifier (if simulating multiple paths)

    Returns:
        DataFrame with valuation results at each time step, including:
            - All columns from input price_path_df
            - total_amm_value: Total AMM position value at each step
            - total_option_value: Total option position value at each step
            - total_portfolio_value: Total portfolio value at each step
            - Additional details from amm_lp_valuation
    """
    results = []

    # Check if we have multiple paths or single path
    has_multiple_paths = path_id_col is not None and path_id_col in price_path_df.columns

    if has_multiple_paths:
        path_ids = price_path_df[path_id_col].unique()
    else:
        path_ids = [None]

    for path_id in path_ids:
        if path_id is not None:
            path_data = price_path_df[price_path_df[path_id_col] == path_id].copy()
        else:
            path_data = price_path_df.copy()

        for idx, row in path_data.iterrows():
            current_price = row['price']
            valuation_datetime = row['datetime']

            # Run valuation
            try:
                valuation_result = amm_lp_valuation(
                    current_price=current_price,
                    valuation_datetime=valuation_datetime,
                    amm_pos=amm_positions,
                    option_hedge_pos=option_hedge_positions
                )

                # Extract key metrics
                result_row = {
                    'datetime': valuation_datetime,
                    'price': current_price,
                    'step': row.get('step', idx),
                    'total_amm_value': valuation_result['total_amm_value'],
                    'total_option_value': valuation_result['total_option_value'],
                    'total_portfolio_value': valuation_result['total_portfolio_value'],
                }

                # Add path_id if present
                if path_id is not None:
                    result_row[path_id_col] = path_id

                # Add detailed AMM position info (for first position, or aggregate if needed)
                if valuation_result['amm_positions']:
                    first_amm = valuation_result['amm_positions'][0]
                    result_row['amm_in_range'] = first_amm.get('in_range', False)
                    result_row['amm_amount0'] = first_amm.get('amount0', 0.0)
                    result_row['amm_amount1'] = first_amm.get('amount1', 0.0)

                # Add option details (for first option, or aggregate if needed)
                if valuation_result['option_positions']:
                    first_option = valuation_result['option_positions'][0]
                    result_row['option_price'] = first_option.get('option_price', 0.0)
                    result_row['time_to_expiry'] = first_option.get('time_to_expiry', 0.0)

                results.append(result_row)

            except Exception as e:
                # Log error and continue
                print(f"Warning: Error at step {idx}, path {path_id}: {e}")
                continue

    return pd.DataFrame(results)


def run_monte_carlo_analysis(
    initial_price: float,
    drift: float,
    volatility: float,
    num_steps: int,
    time_horizon_days: int,
    start_datetime: datetime,
    amm_positions: List[Dict],
    option_hedge_positions: List[Dict],
    num_paths: int = 1,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Complete Monte Carlo analysis: simulate price paths and run valuations.

    Args:
        initial_price: Starting ETH price
        drift: Annual drift rate (mu)
        volatility: Annual volatility (sigma)
        num_steps: Number of time steps in each simulation
        time_horizon_days: Total time horizon in days
        start_datetime: Starting datetime for the simulation
        amm_positions: List of AMM position dictionaries
        option_hedge_positions: List of option hedge position dictionaries
        num_paths: Number of independent paths to simulate (default: 1)
        random_seed: Optional random seed for reproducibility

    Returns:
        DataFrame with price paths and valuation results at each time step
    """
    # Simulate price paths
    if num_paths == 1:
        price_paths = simulate_price_path(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            num_steps=num_steps,
            time_horizon_days=time_horizon_days,
            start_datetime=start_datetime,
            random_seed=random_seed
        )
    else:
        price_paths = simulate_multiple_paths(
            initial_price=initial_price,
            drift=drift,
            volatility=volatility,
            num_steps=num_steps,
            time_horizon_days=time_horizon_days,
            start_datetime=start_datetime,
            num_paths=num_paths,
            random_seed=random_seed
        )

    # Run valuations along the paths
    path_id_col = 'path_id' if num_paths > 1 and 'path_id' in price_paths.columns else None
    results = run_mc_valuation(
        price_path_df=price_paths,
        amm_positions=amm_positions,
        option_hedge_positions=option_hedge_positions,
        path_id_col=path_id_col
    )

    return results


def summarize_mc_results(results_df: pd.DataFrame, path_id_col: Optional[str] = 'path_id') -> pd.DataFrame:
    """
    Summarize Monte Carlo results with statistics across paths.

    Args:
        results_df: DataFrame from run_monte_carlo_analysis
        path_id_col: Column name for path identifier (None if single path)

    Returns:
        DataFrame with summary statistics by time step
    """
    if path_id_col and path_id_col in results_df.columns:
        # Multiple paths - calculate statistics
        summary_cols = ['datetime', 'step']
        value_cols = ['price', 'total_amm_value', 'total_option_value', 'total_portfolio_value']

        summary_stats = []

        for step in results_df['step'].unique():
            step_data = results_df[results_df['step'] == step]

            stats = {
                'step': step,
                'datetime': step_data['datetime'].iloc[0] if len(step_data) > 0 else None,
            }

            for col in value_cols:
                if col in step_data.columns:
                    stats[f'{col}_mean'] = step_data[col].mean()
                    stats[f'{col}_std'] = step_data[col].std()
                    stats[f'{col}_min'] = step_data[col].min()
                    stats[f'{col}_max'] = step_data[col].max()
                    stats[f'{col}_p5'] = step_data[col].quantile(0.05)
                    stats[f'{col}_p95'] = step_data[col].quantile(0.95)

            summary_stats.append(stats)

        return pd.DataFrame(summary_stats)
    else:
        # Single path - just return the results
        return results_df.copy()

