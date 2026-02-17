# AMM LP + Put Hedge Backtest

Monte Carlo backtest of a Uniswap V3 concentrated-liquidity LP position hedged with put options, with a volume-based fee model.

## Strategy

1. **LP position**: Deposit ETH + USDT into a Uniswap V3 concentrated-liquidity range around the current spot price.
2. **Put hedge**: Buy a put option with strike at the LP range's lower bound. Notional = ETH exposure when price falls below the range (full conversion to ETH at that point).
3. **Rebalance**: If price stays out of range for N consecutive hours, withdraw liquidity, sell the old put, recenter the LP around the current price, and buy a new put.
4. **Fees**: Earn swap fees proportional to your share of active liquidity, scaled by trading volume.

## Model components

### Price simulation

Spot and implied volatility are simulated jointly at hourly resolution:

- **Spot**: Geometric Brownian Motion (GBM)
  ```
  dS = μ·S·dt + σ·S·dW₁
  ```
  where σ is the current implied vol from the vol path.

- **Implied vol**: Ornstein–Uhlenbeck (OU) mean-reverting process
  ```
  dσ = κ(θ − σ)·dt + ξ·dW₂
  ```
  with `Corr(dW₁, dW₂) = ρ` (typically negative: vol rises when price falls).

### LP valuation

Uses the Uniswap V3 concentrated-liquidity math (`calc_univ3_current_holdings`):
- When price is **in range**: position holds both ETH and USDT.
- When price is **below range**: position is 100% ETH (impermanent loss maximized on downside).
- When price is **above range**: position is 100% USDT.

### Put pricing

Black-Scholes-Merton with the per-step implied vol from the OU path. Time to expiry uses sub-day precision for hourly steps.

### Fee model (volume-based)

Fees are earned only when the LP position is in range:

```
hourly_volume = daily_volume / 24 × vol_adjustment
vol_adjustment = 1 + sensitivity × (current_vol / long_run_vol − 1)
step_fee = hourly_volume × fee_rate × (lp_value / pool_active_tvl)
```

This captures:
- **Volume–volatility correlation**: More trading activity when vol is high.
- **Liquidity share**: Your fee share is proportional to your LP value relative to total active liquidity.
- **In-range only**: No fees earned when price is outside your LP range.

### Rebalance logic

When `rebalance_on_out_of_range=True`:
1. Track consecutive hours out of range.
2. If `consecutive_out_of_range >= rebalance_delay_steps`, trigger rebalance.
3. **Rebalance**:
   - Withdraw LP → receive `lp_value` in cash.
   - Sell old puts at current BSM fair value → receive `put_value`.
   - Total capital = `lp_value + put_value`.
   - Solve (via fixed-point iteration) for new LP size and put quantity such that `lp_capital + put_cost = total_capital`.
   - Deploy new LP centered at current price; buy new puts.
4. `put_roll_cashflow` = old put value − new put cost (positive = surplus to LP).

## Parameters

### Strategy (`LPPutHedgeStrategy`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `range_lower_pct` | −0.05 | LP range lower bound (% from spot) |
| `range_upper_pct` | +0.05 | LP range upper bound (% from spot) |
| `put_strike_pct` | 0.95 | Put strike as fraction of spot (0.95 = 5% OTM) |
| `initial_eth` | 1.0 | Initial ETH deposit |
| `initial_usdt` | 3000.0 | Initial USDT deposit |
| `vol` | 0.6 | Starting implied vol (60%) |
| `rate` | 0.0 | Risk-free rate |
| `put_expiry_days` | 30 | Put option tenor (days) |
| `pool_fee_rate` | 0.003 | Uniswap fee tier (0.3%) |
| `pool_daily_volume` | 50,000,000 | Base daily pool volume ($) |
| `pool_active_tvl` | 30,000,000 | Active liquidity in similar ranges ($) |
| `vol_volume_sensitivity` | 1.5 | Volume scaling with vol |
| `vol_long_run` | 0.6 | Baseline vol for volume scaling |

### Simulation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spot_drift` | 0.0 | Annual drift for spot (μ) |
| `vol_mean_reversion` | 2.0 | OU mean-reversion speed (κ) |
| `vol_of_vol` | 0.3 | Vol of vol (ξ) |
| `spot_vol_correlation` | −0.5 | Spot–vol correlation (ρ) |
| `step_interval_hours` | 1.0 | Simulation frequency (hours per step) |
| `time_horizon_days` | 365 | Backtest horizon |
| `num_paths` | 20 | Number of Monte Carlo paths |

### Rebalance

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rebalance_on_out_of_range` | True | Enable rebalancing |
| `rebalance_delay_steps` | 3 | Hours out of range before rebalancing (with hourly steps) |

## Output columns

| Column | Description |
|--------|-------------|
| `datetime` | Timestamp |
| `step` | Step index |
| `price` | Simulated ETH price |
| `lp_token0` | ETH held in LP |
| `lp_token1` | USDT held in LP |
| `lp_value` | LP position value ($) |
| `put_value` | Put hedge value ($) |
| `fee_earned` | Fee earned this step ($) |
| `cumulative_fees` | Running total of fees ($) |
| `portfolio_value` | lp_value + put_value + cumulative_fees |
| `delta_lp` | LP value change vs prior step |
| `delta_put` | Put value change vs prior step |
| `theta_approx` | Put theta (time decay at constant price) |
| `put_roll_cashflow` | Net cashflow from put roll on rebalance |
| `put_quantity` | Put notional (ETH) |
| `in_range` | Whether price is in LP range |
| `rebalanced` | Whether rebalance occurred this step |
| `put_strike` | Current put strike |
| `put_expiry` | Current put expiry date |
| `implied_vol` | Implied vol used for put pricing |

## Assumptions and limitations

1. **No transaction costs or slippage** on LP withdrawals, deposits, or put trades.
2. **Put options priced at BSM fair value** — no bid-ask spread, no margin requirements.
3. **Fee model is stylized** — actual fees depend on real-time order flow, MEV, and pool composition. Calibrate `pool_daily_volume` and `pool_active_tvl` from on-chain data for realistic results.
4. **No impermanent loss beyond the Uniswap V3 math** — the model does not account for JIT liquidity or concentrated-liquidity competition dynamics.
5. **Continuous rebalancing** — in practice, gas costs and execution delays would reduce rebalancing frequency.
6. **Single fee tier** — does not model fee tier switching or multi-pool routing.

## Files

| File | Description |
|------|-------------|
| `lp_put_hedge_backtest.py` | Main backtest script with configurable parameters |
| `lp_put_hedge_backtest_output.csv` | Example output (20 paths, hourly, 365 days) |
| `lp_put_hedge_portfolio_paths.png` | Chart: price paths + portfolio value paths |
| `test_amm_lp_valuation.py` | Unit-level test for AMM LP valuation |
| `mc_simulation_demo.py` | Monte Carlo simulation demo |

## Usage

```bash
PYTHONPATH=. python examples/amm/lp_put_hedge_backtest.py
```

Edit the constants at the top of `lp_put_hedge_backtest.py` to change strategy parameters.
