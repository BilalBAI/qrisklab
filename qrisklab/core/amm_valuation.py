import math
from datetime import datetime

from .black_scholes import bsm_pricing
# Valuation for Liquidity Provider's Position on Constant-product AMM with Concentrated Liquidity


# ============================================================================
# CORE VALUATION FUNCTION - calculate the USD value of an LP position (risk-stable pair)
# Note that this assumes that the initial investment is optimally allocated to the range.
# i.e. the invesment is split between the two tokens such that no leftovers.
# ============================================================================

def _calc_optimal_liq(
    initial_price_s0: float,
    initial_investment_c: float,
    lower_boundary_pa: float,
    upper_boundary_pb: float,
) -> float:
    """
    Compute liquidity L from an initial capital C (in token1 units, e.g. USDT)
    at inception price S0 (token1/token0), for a range [Pa, Pb].

    This version works for S0 below/inside/above the range and guarantees V(S0)=C, i.e. C is optimally allocated
    under the same continuous-math assumptions as your valuation function (no fees, no rounding).
    """
    if initial_price_s0 <= 0:
        raise ValueError("initial_price_s0 must be > 0")
    if lower_boundary_pa <= 0 or upper_boundary_pb <= 0:
        raise ValueError("boundaries must be > 0")
    if not (lower_boundary_pa < upper_boundary_pb):
        raise ValueError("lower_boundary_pa must be < upper_boundary_pb")
    if initial_investment_c < 0:
        raise ValueError("initial_investment_c must be >= 0")

    a = math.sqrt(lower_boundary_pa)
    b = math.sqrt(upper_boundary_pb)
    s0 = math.sqrt(initial_price_s0)

    # Below range: all token0 at inception
    if initial_price_s0 <= lower_boundary_pa:
        amount0 = initial_investment_c / initial_price_s0  # token0 units
        # getLiquidityForAmount0
        L = amount0 * (a * b) / (b - a)
        return L

    # Above range: all token1 at inception
    if initial_price_s0 >= upper_boundary_pb:
        amount1 = initial_investment_c                      # token1 units
        # getLiquidityForAmount1
        L = amount1 / (b - a)
        return L

    # In range: assume balanced/max-L deployment (your original formula)
    denom = (2 * s0 - a - initial_price_s0 / b)
    if denom <= 0:
        # Shouldn't happen for a < s0 < b in well-formed ranges, but keep it safe.
        raise ZeroDivisionError(
            "Invalid denominator when computing in-range liquidity.")
    return initial_investment_c / denom


def lp_valuation(
    current_price_s: float,
    initial_price_s0: float,
    initial_investment: float,
    lower_boundary_pa: float,
    upper_boundary_pb: float,
    L: float = None
) -> float:
    """
    Value in token1 units (e.g., USDT) of a Uniswap V3-like position with constant liquidity L.
    """
    if L is None:
        L = _calc_optimal_liq(
            initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb
        )

    a = math.sqrt(lower_boundary_pa)
    b = math.sqrt(upper_boundary_pb)

    if current_price_s <= lower_boundary_pa:
        value = L * current_price_s * (1 / a - 1 / b)
    elif current_price_s >= upper_boundary_pb:
        value = L * (b - a)
    else:
        value = L * (2 * math.sqrt(current_price_s) - a - current_price_s / b)

    return value


# ============================================================================
# CORE VALUATION FUNCTION - calculate current holdings given inital holdings
# ============================================================================

def calc_current_holdings(
    current_price,  # Current token0 price in token1 (e.g., 1 WETH = 3000 USDT)
    lower_price,    # Lower bound price (token1/token0)
    upper_price,    # Upper bound price (token1/token0)
    initial_price,  # Initial price when position was created
    initial_amount0,  # Initial amount of token0
    initial_amount1,  # Initial amount of token1
    decimals0=18,     # Token0 decimals (default 18)
    decimals1=18      # Token1 decimals (default 18)
):
    """
    Calculate current token holdings in a Uniswap V3/V4 position given price ranges.

    Args:
        current_price: Current token0 price in token1 terms (e.g., 1 WETH = 3000 USDT)
        lower_price: Lower bound of price range (token1/token0)
        upper_price: Upper bound of price range (token1/token0)
        initial_price: Initial price when position was created (token1/token0)
        initial_amount0: Initial amount of token0 (human-readable, e.g., 1.5)
        initial_amount1: Initial amount of token1 (human-readable, e.g., 4500)
        decimals0: Decimals for token0 (default 18)
        decimals1: Decimals for token1 (default 18)

    Returns:
        dict with:
            - amount0: Current amount of token0 (human-readable)
            - amount1: Current amount of token1 (human-readable)
            - value_in_token1: Total value in token1 terms
            - in_range: Boolean indicating if current price is within range
    """

    # Convert prices to sqrt prices
    # In Uniswap V3, price = (amount1 / 10^dec1) / (amount0 / 10^dec0) = (amount1 / amount0) * (10^dec0 / 10^dec1)
    # So sqrt_price_raw = sqrt(price * 10^dec1 / 10^dec0)
    # But since we're working with human-readable prices, we need to account for decimals

    dec_factor = math.sqrt((10 ** decimals1) / (10 ** decimals0))

    sqrt_pl = math.sqrt(lower_price) * dec_factor
    sqrt_pu = math.sqrt(upper_price) * dec_factor
    sqrt_pc = math.sqrt(current_price) * dec_factor
    sqrt_pi = math.sqrt(initial_price) * dec_factor  # Initial sqrt price

    # Convert initial amounts to raw units (accounting for decimals)
    amount0_raw = int(initial_amount0 * (10 ** decimals0))
    amount1_raw = int(initial_amount1 * (10 ** decimals1))

    # Calculate initial liquidity from initial amounts and price
    # When price is in range: both tokens are used
    # L can be calculated from either token amount
    # We use both and take the one that's more constrained

    if sqrt_pi < sqrt_pl:
        # Initial price below range - all token0 (price is lower, so we have token0)
        if amount0_raw > 0 and sqrt_pl > 0:
            L = amount0_raw / (1 / sqrt_pl - 1 / sqrt_pu)
        else:
            L = 0
    elif sqrt_pi > sqrt_pu:
        # Initial price above range - all token1 (price is higher, so we have token1)
        if amount1_raw > 0 and sqrt_pu > sqrt_pl:
            L = amount1_raw / (sqrt_pu - sqrt_pl)
        else:
            L = 0
    else:
        # Initial price in range - calculate from both tokens
        # Use the formula that gives the smaller liquidity (the constraint)
        if amount0_raw > 0 and (1 / sqrt_pi - 1 / sqrt_pu) > 0:
            L0 = amount0_raw / (1 / sqrt_pi - 1 / sqrt_pu)
        else:
            L0 = float('inf')

        if amount1_raw > 0 and (sqrt_pi - sqrt_pl) > 0:
            L1 = amount1_raw / (sqrt_pi - sqrt_pl)
        else:
            L1 = float('inf')

        # Take the minimum (most constrained)
        L = min(L0, L1) if L0 != float('inf') and L1 != float(
            'inf') else (L0 if L0 != float('inf') else L1)

    if L <= 0 or L == float('inf'):
        return {
            'amount0': 0.0,
            'amount1': 0.0,
            'value_in_token1': 0.0,
            'in_range': False,
            'error': 'Invalid liquidity calculation'
        }

    # Calculate current amounts based on current price
    if sqrt_pc < sqrt_pl:
        # Current price below range - all token0 (price is lower, so we have token0)
        amount0_current = L * (1 / sqrt_pl - 1 / sqrt_pu)
        amount1_current = 0.0
        in_range = False
    elif sqrt_pc > sqrt_pu:
        # Current price above range - all token1 (price is higher, so we have token1)
        amount0_current = 0.0
        amount1_current = L * (sqrt_pu - sqrt_pl)
        in_range = False
    else:
        # Current price in range - both tokens
        # Ensure we calculate correctly even if values are small
        diff0 = (1 / sqrt_pc - 1 / sqrt_pu)
        diff1 = (sqrt_pc - sqrt_pl)
        amount0_current = L * diff0 if diff0 > 0 else 0.0
        amount1_current = L * diff1 if diff1 > 0 else 0.0
        in_range = True

    # Convert back to human-readable amounts
    amount0_human = float(amount0_current) / (10 ** decimals0)
    amount1_human = float(amount1_current) / (10 ** decimals1)

    # Calculate total value in token1 terms
    value_in_token1 = amount0_human * current_price + amount1_human

    return {
        'amount0': amount0_human,
        'amount1': amount1_human,
        'value_in_token1': value_in_token1,
        'in_range': in_range,
        'liquidity': float(L)
    }


# ============================================================================
# GREEK CALCULATION FUNCTIONS (With Formula Comments)
# ============================================================================

def calc_delta_gamma_fdm(current_price_s, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb, L=None):
    """Calculates Delta and Gamma using the numerical finite difference method."""
    # Define a small price shift (dS) for the numerical calculation
    ds = current_price_s * 0.001

    # Let V(S) be the function that calculates the product's value at price S.
    # We calculate the value at the current price, a price slightly higher and slightly lower
    value_mid = lp_valuation(
        # V(S)
        current_price_s, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb, L=L)
    value_up = lp_valuation(
        # V(S + dS)
        current_price_s + ds, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb, L=L)
    value_down = lp_valuation(
        # V(S - dS)
        current_price_s - ds, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb, L=L)

    # Calculate Delta using the central difference formula for the first derivative.
    # Formula: Delta = dV/dS = (V(S + dS) - V(S - dS)) / (2 * dS)
    delta = (value_up - value_down) / (2 * ds)

    # Calculate Gamma using the central difference formula for the second derivative.
    # Formula: Gamma = d²V/dS² = (V(S + dS) - 2*V(S) + V(S - dS)) / (dS²)
    gamma = (value_up - 2 * value_mid + value_down) / (ds ** 2)

    return delta, gamma


def calc_delta_gamma_analytic(
    current_price_s: float,
    initial_price_s0: float,
    initial_investment_c: float,
    lower_boundary_pa: float,
    upper_boundary_pb: float,
    L: float = None
):
    """
    Analytic Delta and Gamma for the same piecewise valuation model used in lp_valuation.

    Conventions:
      - S is token1/token0 (e.g., USDT per ETH when token0=ETH, token1=USDT)
      - Value V(S) is in token1 units
      - Delta = dV/dS has units of token0 (ETH)
      - Gamma = d²V/dS² has units of token0 per token1 (ETH per USDT)

    Notes:
      - At the boundaries S=Pa or S=Pb, V(S) is continuous but Delta has a kink.
        This function returns the zone formula according to <= and >= comparisons.
      - In continuous math, gamma is 0 outside the range and negative inside the range.
    """
    if current_price_s <= 0:
        raise ValueError("current_price_s must be > 0")

    if L is None:
        L = _calc_optimal_liq(
            initial_price_s0, initial_investment_c, lower_boundary_pa, upper_boundary_pb
        )

    a = math.sqrt(lower_boundary_pa)
    b = math.sqrt(upper_boundary_pb)

    # Zone: below range
    if current_price_s <= lower_boundary_pa:
        # V = L * S * (1/a - 1/b)
        delta = L * (1.0 / a - 1.0 / b)
        gamma = 0.0
        return delta, gamma

    # Zone: above range
    if current_price_s >= upper_boundary_pb:
        # V = L * (b - a)
        delta = 0.0
        gamma = 0.0
        return delta, gamma

    # Zone: inside range
    s = math.sqrt(current_price_s)
    # V = L * (2*sqrt(S) - a - S/b)
    # dV/dS = L * (1/s - 1/b)
    delta = L * (1.0 / s - 1.0 / b)
    # d²V/dS² = - L / (2 * S^(3/2))
    gamma = -L / (2.0 * (current_price_s ** 1.5))
    return delta, gamma


def amm_lp_valuation(
    current_price, valuation_datetime, amm_pos, option_hedge_pos
):
    """
    Calculate valuation for AMM LP positions and option hedge positions.

    Args:
        current_price: Current token0 price in token1 terms (e.g., 1 WETH = 3000 USDT)
        valuation_datetime: Valuation datetime (datetime object or string like "2025-06-17 13:45:30")
        amm_pos: List of dicts containing AMM position parameters
        option_hedge_pos: List of dicts containing option hedge position parameters

    Returns:
        dict with:
            - amm_positions: List of valuation results for each AMM position
            - option_positions: List of valuation results for each option position
            - total_amm_value: Sum of all AMM position values
            - total_option_value: Sum of all option position values
            - total_portfolio_value: Total portfolio value (AMM + options)
    """
    # Convert valuation_datetime to datetime object if it's a string
    if isinstance(valuation_datetime, str):
        try:
            valuation_datetime = datetime.strptime(
                valuation_datetime, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                valuation_datetime = datetime.strptime(
                    valuation_datetime, "%Y-%m-%d")
            except ValueError:
                raise ValueError(
                    f"Unable to parse valuation_datetime: {valuation_datetime}")

    # Price all option hedge positions
    option_results = []
    total_option_value = 0.0

    for idx, opt_pos in enumerate(option_hedge_pos):
        # Parse expiry date (format like '27FEB26')
        expiry_str = opt_pos.get('expiry')
        if expiry_str:
            try:
                expiry_date = datetime.strptime(expiry_str, '%d%b%y')
            except ValueError:
                raise ValueError(f"Unable to parse expiry date: {expiry_str}")

            # Calculate time to expiry in years
            time_to_expiry = (expiry_date - valuation_datetime).days / 365.0
            # Ensure time_to_expiry is not negative
            time_to_expiry = max(0.0, time_to_expiry)
        else:
            time_to_expiry = 0.0

        # Get option parameters
        strike = opt_pos.get('strike', 0)
        vol = opt_pos.get('vol', 0)
        rate = opt_pos.get('rate', 0.0)
        put_call = opt_pos.get('put_call', 'call')

        # Price the option using Black-Scholes
        option_price = bsm_pricing(
            strike=strike,
            time_to_expiry=time_to_expiry,
            spot=current_price,
            rate=rate,
            vol=vol,
            put_call=put_call
        )

        option_result = {
            'position_id': idx,
            'strike': strike,
            'expiry': expiry_str,
            'put_call': put_call,
            'vol': vol,
            'rate': rate,
            'time_to_expiry': time_to_expiry,
            'option_price': float(option_price)
        }
        option_results.append(option_result)
        total_option_value += option_price

    # Price all AMM positions
    amm_results = []
    total_amm_value = 0.0

    for idx, amm in enumerate(amm_pos):
        holdings = calc_current_holdings(
            current_price=current_price,
            lower_price=amm.get('lower_price', 0),
            upper_price=amm.get('upper_price', 0),
            initial_price=amm.get('initial_price', current_price),
            initial_amount0=amm.get('initial_amount0', 0),
            initial_amount1=amm.get('initial_amount1', 0),
            decimals0=amm.get('decimals0', 18),
            decimals1=amm.get('decimals1', 18)
        )

        amm_result = {
            'position_id': idx,
            'lower_price': amm.get('lower_price'),
            'upper_price': amm.get('upper_price'),
            'initial_price': amm.get('initial_price'),
            'current_price': current_price,
            'amount0': holdings.get('amount0', 0.0),
            'amount1': holdings.get('amount1', 0.0),
            'value_in_token1': holdings.get('value_in_token1', 0.0),
            'in_range': holdings.get('in_range', False),
            'liquidity': holdings.get('liquidity', 0.0)
        }
        amm_results.append(amm_result)
        total_amm_value += holdings.get('value_in_token1', 0.0)

    # Calculate total portfolio value
    total_portfolio_value = total_amm_value + total_option_value

    return {
        'amm_positions': amm_results,
        'option_positions': option_results,
        'total_amm_value': float(total_amm_value),
        'total_option_value': float(total_option_value),
        'total_portfolio_value': float(total_portfolio_value),
        'valuation_datetime': valuation_datetime.strftime("%Y-%m-%d %H:%M:%S") if isinstance(valuation_datetime, datetime) else str(valuation_datetime),
        'current_price': current_price
    }


# def lp_valuation(
#     current_price_s: float,
#     initial_price_s0: float,
#     initial_investment: float,
#     lower_boundary_pa: float,
#     upper_boundary_pb: float
# ) -> float:
#     """
#     Calculates the value of a structured range accrual note based on the current price.

#     Args:
#         current_price_s: The current price of the underlying asset (S).
#         initial_price_s0: The price of the asset when the investment was made (S0).
#         initial_investment: The initial principal amount of the investment.
#         lower_boundary_pa: The lower price boundary of the range (Pa).
#         upper_boundary_pb: The upper price boundary of the range (Pb).

#     Returns:
#         The calculated current value of the structured product.
#     """

#     # Step 1: Calculate the constant Leverage Factor (L)
#     # This is calculated once based on initial conditions.
#     try:
#         leverage_factor_l = initial_investment / (
#             2 * math.sqrt(initial_price_s0)
#             - math.sqrt(lower_boundary_pa)
#             - initial_price_s0 / math.sqrt(upper_boundary_pb)
#         )
#     except (ValueError, ZeroDivisionError) as e:
#         print(f"Error calculating Leverage Factor L: {e}")
#         return 0.0

#     # Step 2: Determine which price zone S is in and apply the correct formula
#     if current_price_s <= lower_boundary_pa:
#         # Zone 2: Price is BELOW the range
#         value = leverage_factor_l * current_price_s * (
#             1 / math.sqrt(lower_boundary_pa) - 1 / math.sqrt(upper_boundary_pb)
#         )
#     elif current_price_s >= upper_boundary_pb:
#         # Zone 3: Price is ABOVE the range
#         value = leverage_factor_l * (
#             math.sqrt(upper_boundary_pb) - math.sqrt(lower_boundary_pa)
#         )
#     else:
#         # Zone 1: Price is WITHIN the range
#         value = leverage_factor_l * (
#             2 * math.sqrt(current_price_s)
#             - math.sqrt(lower_boundary_pa)
#             - current_price_s / math.sqrt(upper_boundary_pb)
#         )

#     return value
