"""
General module for simulating spot (price) and implied volatility paths.

All path simulation functions live here. Outputs can be used by other modules
(AMM valuation, option pricing, stress testing, etc.)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

__all__ = [
    'simulate_price_path',
    'simulate_multiple_paths',
    'simulate_spot_vol_paths',
    'simulate_spot_only',
]


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
    Simulate a single price path using geometric Brownian motion (GBM).

    Args:
        initial_price: Starting price
        drift: Annual drift rate (mu, e.g., 0.05 for 5% annual growth)
        volatility: Annual volatility (sigma, e.g., 0.6 for 60%)
        num_steps: Number of time steps in the simulation
        time_horizon_days: Total time horizon in days
        start_datetime: Starting datetime for the simulation
        random_seed: Optional random seed for reproducibility

    Returns:
        DataFrame with columns: datetime, price, step
    """
    df = simulate_spot_only(
        initial_spot=initial_price,
        volatility=volatility,
        num_steps=num_steps,
        time_horizon_days=time_horizon_days,
        start_datetime=start_datetime,
        drift=drift,
        num_paths=1,
        random_seed=random_seed
    )
    return df.rename(columns={'spot': 'price'})


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
    Simulate multiple price paths using geometric Brownian motion (GBM).

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
        DataFrame with columns: path_id, datetime, price, step
    """
    df = simulate_spot_only(
        initial_spot=initial_price,
        volatility=volatility,
        num_steps=num_steps,
        time_horizon_days=time_horizon_days,
        start_datetime=start_datetime,
        drift=drift,
        num_paths=num_paths,
        random_seed=random_seed
    )
    return df.rename(columns={'spot': 'price'})


def simulate_spot_vol_paths(
    initial_spot: float,
    initial_vol: float,
    num_steps: int,
    time_horizon_days: int,
    start_datetime: datetime,
    spot_drift: float = 0.0,
    vol_mean_reversion: float = 2.0,
    vol_long_run: Optional[float] = None,
    vol_of_vol: float = 0.3,
    spot_vol_correlation: float = -0.5,
    num_paths: int = 1,
    step_interval_hours: Optional[float] = None,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate joint spot and implied volatility paths using correlated Brownian motions.

    Spot follows GBM:       dS = μS dt + σS dW1  (σ from vol path)
    Vol follows Ornstein–Uhlenbeck (OU) process: dσ = κ(θ - σ) dt + ξ dW2
    with Corr(dW1, dW2) = ρ

    Args:
        initial_spot: Starting spot price
        initial_vol: Starting implied volatility (e.g., 0.6 for 60%)
        num_steps: Number of time steps in the simulation
        time_horizon_days: Total time horizon in days
        start_datetime: Starting datetime for the simulation
        spot_drift: Annual drift rate for spot (mu)
        vol_mean_reversion: Mean reversion speed for vol (kappa)
        vol_long_run: Long-run mean for vol (theta). If None, uses initial_vol
        vol_of_vol: Volatility of volatility (xi)
        spot_vol_correlation: Correlation between spot and vol shocks (rho, typically negative)
        num_paths: Number of independent paths to simulate
        step_interval_hours: If set, use hourly (or N-hourly) steps; overrides num_steps.
                             E.g., 1.0 = hourly, 24.0 = daily. dt scales for realistic per-step moves.
        random_seed: Optional random seed for reproducibility

    Returns:
        DataFrame with columns:
            - path_id: Path identifier (0 to num_paths-1, omitted if num_paths=1)
            - datetime: Timestamp for each step
            - step: Step number (0 to num_steps)
            - spot: Simulated spot price
            - vol: Simulated implied volatility (e.g., 0.55 means 55%)
    """
    vol_long_run = vol_long_run if vol_long_run is not None else initial_vol

    if step_interval_hours is not None:
        num_steps = int(time_horizon_days * 24 / step_interval_hours)
        dt = step_interval_hours / (365.25 * 24)
        use_hourly_datetimes = True
    else:
        dt = (time_horizon_days / 365.0) / num_steps
        use_hourly_datetimes = False
    rng = np.random.default_rng(random_seed)

    all_rows = []

    for path_id in range(num_paths):
        if random_seed is not None:
            rng = np.random.default_rng(random_seed + path_id)

        # Correlated standard normals: Z1, Z2 where Corr(Z1, Z2) = spot_vol_correlation
        # Z1 independent, Z2 = rho*Z1 + sqrt(1-rho^2)*Z_indep
        Z1 = rng.standard_normal(num_steps)
        Z_indep = rng.standard_normal(num_steps)
        rho = np.clip(spot_vol_correlation, -1.0, 1.0)
        Z2 = rho * Z1 + np.sqrt(1 - rho ** 2) * Z_indep

        # Spot path (GBM)
        spot_path = np.empty(num_steps + 1)
        spot_path[0] = initial_spot

        # Vol path (OU / mean-reverting)
        vol_path = np.empty(num_steps + 1)
        vol_path[0] = initial_vol

        for i in range(num_steps):
            # Use current vol for spot diffusion (or could use spot_vol constant)
            sig = vol_path[i]  # stochastic vol drives spot, or use spot_vol for constant
            spot_path[i + 1] = spot_path[i] * np.exp(
                (spot_drift - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * Z1[i]
            )
            # OU for vol: dσ = κ(θ - σ) dt + ξ dW
            vol_path[i + 1] = vol_path[i] + vol_mean_reversion * (vol_long_run - vol_path[i]) * dt \
                + vol_of_vol * np.sqrt(dt) * Z2[i]
            vol_path[i + 1] = np.maximum(vol_path[i + 1], 0.01)  # Floor at 1%

        if use_hourly_datetimes:
            datetimes = [
                start_datetime + timedelta(hours=i * step_interval_hours)
                for i in range(num_steps + 1)
            ]
        else:
            datetimes = [
                start_datetime + timedelta(days=time_horizon_days * i / num_steps)
                for i in range(num_steps + 1)
            ]

        for i in range(num_steps + 1):
            row = {
                'datetime': datetimes[i],
                'step': i,
                'spot': spot_path[i],
                'vol': vol_path[i],
            }
            if num_paths > 1:
                row['path_id'] = path_id
            all_rows.append(row)

    df = pd.DataFrame(all_rows)

    if num_paths > 1:
        df = df[['path_id', 'datetime', 'step', 'spot', 'vol']]
    else:
        df = df[['datetime', 'step', 'spot', 'vol']]

    return df


def simulate_spot_only(
    initial_spot: float,
    volatility: float,
    num_steps: int,
    time_horizon_days: int,
    start_datetime: datetime,
    drift: float = 0.0,
    num_paths: int = 1,
    random_seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate spot paths only (constant vol, GBM). Convenience wrapper.

    Returns:
        DataFrame with columns: path_id (if num_paths>1), datetime, step, spot
    """
    df = simulate_spot_vol_paths(
        initial_spot=initial_spot,
        initial_vol=volatility,
        num_steps=num_steps,
        time_horizon_days=time_horizon_days,
        start_datetime=start_datetime,
        spot_drift=drift,
        vol_mean_reversion=0.0,  # No vol dynamics
        vol_long_run=volatility,
        vol_of_vol=0.0,
        spot_vol_correlation=0.0,
        num_paths=num_paths,
        random_seed=random_seed
    )
    return df[['path_id', 'datetime', 'step', 'spot']] if num_paths > 1 else df[['datetime', 'step', 'spot']]
