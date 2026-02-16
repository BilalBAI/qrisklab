"""
Test script for AMM LP valuation.

This script demonstrates how to use amm_lp_valuation with:
- Example AMM position (0.25 ETH + 801.13 USDT in range 2980.97-3463.36)
- Example option hedge position (3000 strike put expiring 27FEB26)
"""

from qrisklab.core.amm_calc import amm_lp_valuation


def test_amm_lp_valuation():
    """
    Test function for amm_lp_valuation using the example data.

    This function demonstrates how to use amm_lp_valuation with:
    - Example AMM position (0.25 ETH + 801.13 USDT in range 2980.97-3463.36)
    - Example option hedge position (3000 strike put expiring 27FEB26)
    """
    # Example AMM position data
    amm_pos = [{
        'lower_price': 2980.97,      # Lower bound
        'upper_price': 3463.36,      # Upper bound
        'initial_price': 3212.8,     # Initial price
        'initial_amount0': 0.25,     # 0.25 ETH
        'initial_amount1': 801.13,    # 801 USDT
        'decimals0': 18,
        'decimals1': 6
    }]

    # Example option hedge position data
    option_hedge_pos = [{
        'strike': 3000,
        'expiry': '27FEB26',
        'put_call': 'put',
        'vol': 0.5576,
        'rate': 0.0
    }]

    # Test parameters
    current_price = 2960.0  # Current ETH price in USDT
    valuation_datetime = "2026-01-20 00:00:00"  # Valuation date

    # Run valuation
    result = amm_lp_valuation(
        current_price=current_price,
        valuation_datetime=valuation_datetime,
        amm_pos=amm_pos,
        option_hedge_pos=option_hedge_pos
    )

    # Print results in a readable format
    print("=" * 80)
    print("AMM LP Valuation Test Results")
    print("=" * 80)
    print(f"\nValuation Date: {result['valuation_datetime']}")
    print(f"Current Price: ${result['current_price']:,.2f}")
    print("\n" + "-" * 80)
    print("AMM Positions:")
    print("-" * 80)
    for pos in result['amm_positions']:
        print(f"\nPosition ID: {pos['position_id']}")
        print(f"  Price Range: ${pos['lower_price']:,.2f} - ${pos['upper_price']:,.2f}")
        print(f"  Initial Price: ${pos['initial_price']:,.2f}")
        print(f"  Current Holdings:")
        print(f"    Token0: {pos['amount0']:.6f}")
        print(f"    Token1: ${pos['amount1']:,.2f}")
        print(f"  Value in Token1: ${pos['value_in_token1']:,.2f}")
        print(f"  In Range: {pos['in_range']}")
        print(f"  Liquidity: {pos['liquidity']:.2e}")

    print("\n" + "-" * 80)
    print("Option Positions:")
    print("-" * 80)
    for pos in result['option_positions']:
        print(f"\nPosition ID: {pos['position_id']}")
        print(f"  Strike: ${pos['strike']:,.2f}")
        print(f"  Expiry: {pos['expiry']}")
        print(f"  Type: {pos['put_call'].upper()}")
        print(f"  Vol: {pos['vol']:.4f} ({pos['vol'] * 100:.2f}%)")
        print(f"  Rate: {pos['rate']:.4f} ({pos['rate'] * 100:.2f}%)")
        print(f"  Time to Expiry: {pos['time_to_expiry']:.4f} years ({pos['time_to_expiry'] * 365:.1f} days)")
        print(f"  Option Price: ${pos['option_price']:,.2f}")

    print("\n" + "-" * 80)
    print("Summary:")
    print("-" * 80)
    print(f"Total AMM Value: ${result['total_amm_value']:,.2f}")
    print(f"Total Option Value: ${result['total_option_value']:,.2f}")
    print(f"Total Portfolio Value: ${result['total_portfolio_value']:,.2f}")
    print("=" * 80)

    # Also return JSON for programmatic access
    return result


if __name__ == "__main__":
    test_amm_lp_valuation()
