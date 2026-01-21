import math
from datetime import datetime

from .black_scholes import bsm_pricing


def calc_univ3_current_holdings(
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
    Calculate current token holdings in a Uniswap V3 position given price ranges.

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


# amm_pos = [{
#     'lower_price': 2980.97,      # Lower bound
#     'upper_price': 3463.36,      # Upper bound
#     'initial_price': 3212.8,     # Initial price
#     'initial_amount0': 0.25,     # 0.25 ETH
#     'initial_amount1': 801.13,    # 801 USDT
#     'decimals0': 18,
#     'decimals1': 6
# }]

# option_hedge_pos = [{
#     'strike': 3000,
#     'expiry': '27FEB26',
#     'put_call': 'put',
#     'vol': 0.6179,
#     'rate': 0.0
# }]


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
            valuation_datetime = datetime.strptime(valuation_datetime, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                valuation_datetime = datetime.strptime(valuation_datetime, "%Y-%m-%d")
            except ValueError:
                raise ValueError(f"Unable to parse valuation_datetime: {valuation_datetime}")

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
        holdings = calc_univ3_current_holdings(
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
