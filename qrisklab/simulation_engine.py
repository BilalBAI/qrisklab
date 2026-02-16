"""
Monte Carlo simulation for AMM LP positions.

Uses path simulation from path_simulation and evaluates AMM LP positions
along those paths. Follows the same class-based pattern as StressTestEngine.
"""

import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

from .core.amm_calc import amm_lp_valuation
from .core.path_simulation import simulate_price_path, simulate_multiple_paths


class SimulationEngine:
    """
    Monte Carlo simulation engine for AMM LP positions.

    Usage:
        engine = SimulationEngine(
            initial_price=3200.0,
            drift=0.0,
            volatility=0.8,
            num_steps=100,
            time_horizon_days=365,
            start_datetime=datetime(2025, 1, 15)
        )
        engine.update_positions(amm_positions, option_hedge_positions)
        results = engine.run(num_paths=10, random_seed=42)
        summary = engine.summarize()
    """

    def __init__(
        self,
        initial_price: float,
        drift: float = 0.0,
        volatility: float = 0.8,
        num_steps: int = 100,
        time_horizon_days: int = 365,
        start_datetime: Optional[datetime] = None
    ) -> None:
        """
        Initialize the simulation engine.

        Args:
            initial_price: Starting spot price
            drift: Annual drift rate (mu)
            volatility: Annual volatility (sigma)
            num_steps: Number of time steps in each path
            time_horizon_days: Total time horizon in days
            start_datetime: Starting datetime. If None, uses current datetime
        """
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.num_steps = num_steps
        self.time_horizon_days = time_horizon_days
        self.start_datetime = start_datetime if start_datetime is not None else datetime.now()

        self.amm_positions: List[Dict] = []
        self.option_hedge_positions: List[Dict] = []
        self.results: Optional[pd.DataFrame] = None

    def set_simulation_params(
        self,
        initial_price: Optional[float] = None,
        drift: Optional[float] = None,
        volatility: Optional[float] = None,
        num_steps: Optional[int] = None,
        time_horizon_days: Optional[int] = None,
        start_datetime: Optional[datetime] = None
    ) -> None:
        """Update simulation parameters."""
        if initial_price is not None:
            self.initial_price = initial_price
        if drift is not None:
            self.drift = drift
        if volatility is not None:
            self.volatility = volatility
        if num_steps is not None:
            self.num_steps = num_steps
        if time_horizon_days is not None:
            self.time_horizon_days = time_horizon_days
        if start_datetime is not None:
            self.start_datetime = start_datetime

    def update_positions(
        self,
        amm_positions: List[Dict],
        option_hedge_positions: List[Dict]
    ) -> None:
        """
        Set AMM and option hedge positions for valuation.

        Args:
            amm_positions: List of AMM position dictionaries
            option_hedge_positions: List of option hedge position dictionaries
        """
        self.amm_positions = amm_positions
        self.option_hedge_positions = option_hedge_positions

    def run(
        self,
        num_paths: int = 1,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run Monte Carlo simulation: simulate price paths and evaluate positions.

        Args:
            num_paths: Number of independent paths (default: 1)
            random_seed: Optional random seed for reproducibility

        Returns:
            DataFrame with valuation results at each time step
        """
        if not self.amm_positions and not self.option_hedge_positions:
            raise ValueError("No positions set. Call update_positions() first.")

        # Simulate price paths
        if num_paths == 1:
            price_paths = simulate_price_path(
                initial_price=self.initial_price,
                drift=self.drift,
                volatility=self.volatility,
                num_steps=self.num_steps,
                time_horizon_days=self.time_horizon_days,
                start_datetime=self.start_datetime,
                random_seed=random_seed
            )
        else:
            price_paths = simulate_multiple_paths(
                initial_price=self.initial_price,
                drift=self.drift,
                volatility=self.volatility,
                num_steps=self.num_steps,
                time_horizon_days=self.time_horizon_days,
                start_datetime=self.start_datetime,
                num_paths=num_paths,
                random_seed=random_seed
            )

        # Run valuations along the paths
        path_id_col = 'path_id' if num_paths > 1 and 'path_id' in price_paths.columns else None
        self.results = self._run_mc_valuation(
            price_path_df=price_paths,
            path_id_col=path_id_col
        )

        return self.results

    def summarize(self, path_id_col: Optional[str] = 'path_id') -> pd.DataFrame:
        """
        Summarize Monte Carlo results with statistics across paths.

        Args:
            path_id_col: Column name for path identifier (None if single path)

        Returns:
            DataFrame with summary statistics by time step
        """
        if self.results is None:
            raise ValueError("No results. Call run() first.")

        return self._summarize_results(self.results, path_id_col)

    def _run_mc_valuation(
        self,
        price_path_df: pd.DataFrame,
        path_id_col: Optional[str] = None
    ) -> pd.DataFrame:
        """Run AMM LP valuation along price paths."""
        results = []

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

                try:
                    valuation_result = amm_lp_valuation(
                        current_price=current_price,
                        valuation_datetime=valuation_datetime,
                        amm_pos=self.amm_positions,
                        option_hedge_pos=self.option_hedge_positions
                    )

                    result_row = {
                        'datetime': valuation_datetime,
                        'price': current_price,
                        'step': row.get('step', idx),
                        'total_amm_value': valuation_result['total_amm_value'],
                        'total_option_value': valuation_result['total_option_value'],
                        'total_portfolio_value': valuation_result['total_portfolio_value'],
                    }

                    if path_id is not None:
                        result_row[path_id_col] = path_id

                    if valuation_result['amm_positions']:
                        first_amm = valuation_result['amm_positions'][0]
                        result_row['amm_in_range'] = first_amm.get('in_range', False)
                        result_row['amm_amount0'] = first_amm.get('amount0', 0.0)
                        result_row['amm_amount1'] = first_amm.get('amount1', 0.0)

                    if valuation_result['option_positions']:
                        first_option = valuation_result['option_positions'][0]
                        result_row['option_price'] = first_option.get('option_price', 0.0)
                        result_row['time_to_expiry'] = first_option.get('time_to_expiry', 0.0)

                    results.append(result_row)

                except Exception as e:
                    print(f"Warning: Error at step {idx}, path {path_id}: {e}")
                    continue

        return pd.DataFrame(results)

    def _summarize_results(
        self,
        results_df: pd.DataFrame,
        path_id_col: Optional[str]
    ) -> pd.DataFrame:
        """Summarize results with statistics across paths."""
        if path_id_col and path_id_col in results_df.columns:
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
            return results_df.copy()


# Backward-compatible function wrappers
def run_mc_valuation(
    price_path_df: pd.DataFrame,
    amm_positions: List[Dict],
    option_hedge_positions: List[Dict],
    path_id_col: str = 'path_id'
) -> pd.DataFrame:
    """
    Run AMM LP valuation along a price path or multiple paths.

    (Convenience function; consider using SimulationEngine for new code.)
    """
    engine = SimulationEngine(initial_price=0.0, num_steps=0, time_horizon_days=0)
    engine.update_positions(amm_positions, option_hedge_positions)
    return engine._run_mc_valuation(price_path_df, path_id_col)


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

    (Convenience function; consider using SimulationEngine for new code.)
    """
    engine = SimulationEngine(
        initial_price=initial_price,
        drift=drift,
        volatility=volatility,
        num_steps=num_steps,
        time_horizon_days=time_horizon_days,
        start_datetime=start_datetime
    )
    engine.update_positions(amm_positions, option_hedge_positions)
    return engine.run(num_paths=num_paths, random_seed=random_seed)


def summarize_mc_results(
    results_df: pd.DataFrame,
    path_id_col: Optional[str] = 'path_id'
) -> pd.DataFrame:
    """
    Summarize Monte Carlo results with statistics across paths.

    (Convenience function; consider using SimulationEngine.summarize() for new code.)
    """
    engine = SimulationEngine(initial_price=0.0, num_steps=0, time_horizon_days=0)
    engine.results = results_df
    return engine.summarize(path_id_col=path_id_col)
