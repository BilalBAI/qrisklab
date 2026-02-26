import math


# ============================================================================
# 1. CORE VALUATION FUNCTION
# Note that this fucntion assumes that the initial investment is optimally allocated to the range.
# i.e. the invesment is split between the two tokens such that no leftovers.
# ============================================================================

import math


def calc_univ3_liquidity_from_capital(
    initial_price_s0: float,
    initial_investment_c: float,
    lower_boundary_pa: float,
    upper_boundary_pb: float,
) -> float:
    """
    Compute Uniswap V3 liquidity L from an initial capital C (in token1 units, e.g. USDT)
    at inception price S0 (token1/token0), for a range [Pa, Pb].

    This version works for S0 below/inside/above the range and guarantees V(S0)=C
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


def calc_univ3_value(
    current_price_s: float,
    initial_price_s0: float,
    initial_investment: float,
    lower_boundary_pa: float,
    upper_boundary_pb: float
) -> float:
    """
    Value in token1 units (e.g., USDT) of a Uniswap V3-like position with constant liquidity L.
    """
    L = calc_univ3_liquidity_from_capital(
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


# def calc_univ3_value(
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


# ============================================================================
# 2. GREEK CALCULATION FUNCTIONS (With Formula Comments)
# ============================================================================

def calc_univ3_delta_gamma_fdm(current_price_s, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb):
    """Calculates Delta and Gamma using the numerical finite difference method."""
    # Define a small price shift (dS) for the numerical calculation
    ds = current_price_s * 0.001

    # Let V(S) be the function that calculates the product's value at price S.
    # We calculate the value at the current price, a price slightly higher and slightly lower
    value_mid = calc_univ3_value(
        # V(S)
        current_price_s, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb)
    value_up = calc_univ3_value(
        # V(S + dS)
        current_price_s + ds, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb)
    value_down = calc_univ3_value(
        # V(S - dS)
        current_price_s - ds, initial_price_s0, initial_investment, lower_boundary_pa, upper_boundary_pb)

    # Calculate Delta using the central difference formula for the first derivative.
    # Formula: Delta = dV/dS = (V(S + dS) - V(S - dS)) / (2 * dS)
    delta = (value_up - value_down) / (2 * ds)

    # Calculate Gamma using the central difference formula for the second derivative.
    # Formula: Gamma = d²V/dS² = (V(S + dS) - 2*V(S) + V(S - dS)) / (dS²)
    gamma = (value_up - 2 * value_mid + value_down) / (ds ** 2)

    return delta, gamma


def calc_univ3_delta_gamma_analytic(
    current_price_s: float,
    initial_price_s0: float,
    initial_investment_c: float,
    lower_boundary_pa: float,
    upper_boundary_pb: float,
):
    """
    Analytic Delta and Gamma for the same piecewise valuation model used in calc_univ3_value.

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

    L = calc_univ3_liquidity_from_capital(
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
