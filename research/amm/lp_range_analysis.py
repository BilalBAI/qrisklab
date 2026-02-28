"""
LP Range Analysis: Gamma Coverage of Delta-First Put Hedge

Computes, for a range of symmetric LP ranges [S×(1-r), S×(1+r)]:
  - Liquidity L, LP delta, LP gamma
  - Number of puts (delta-first sizing)
  - Gamma hedge coverage
  - Residual gamma drag (monthly)

Also solves analytically for the crossover range where coverage = 100%.

Usage:
    python lp_range_analysis.py
"""

import math
from scipy.stats import norm

# ─────────────────────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────────────────────
S               = 2_000.0       # Current ETH price ($)
C               = 1_000_000.0   # Capital deployed ($)
PUT_DELTA_TGT   = 0.30          # Target put delta (|Δ_put|)
SIGMA           = 0.75          # Annual implied vol
T_DAYS          = 30            # Put tenor (days)
RATE            = 0.05          # Risk-free rate (annual)

RANGES = [0.05, 0.10, 0.15, 0.20, 0.22, 0.30]  # Symmetric ±r%


# ─────────────────────────────────────────────────────────────────────────────
# Put parameters (BSM, fixed across all ranges)
# ─────────────────────────────────────────────────────────────────────────────

def calc_put_params(S, sigma, T_days, rate, put_delta_tgt):
    """
    Compute 30Δ put strike, gamma per contract, and put price via BSM.

    For a put: Δ_put = N(d1) - 1 = -put_delta_tgt
      → N(d1) = 1 - put_delta_tgt
      → d1 = N^{-1}(1 - put_delta_tgt)

    Returns:
        K          : put strike ($)
        gamma_put  : BSM gamma per contract ($/($²))
        put_price  : BSM put price per contract ($)
    """
    T = T_days / 365.0
    sqrtT = math.sqrt(T)
    sigma_sqrtT = sigma * sqrtT

    # Invert d1 for target delta
    d1 = norm.ppf(1.0 - put_delta_tgt)

    # d1 = [ln(S/K) + (r + σ²/2)·T] / (σ√T)
    # → ln(S/K) = d1·σ√T − (r + σ²/2)·T
    ln_S_over_K = d1 * sigma_sqrtT - (rate + 0.5 * sigma ** 2) * T
    K = S / math.exp(ln_S_over_K)

    d2 = d1 - sigma_sqrtT

    # BSM gamma (same for call and put)
    gamma_put = norm.pdf(d1) / (S * sigma_sqrtT)

    # BSM put price: P = K·e^{-rT}·N(-d2) - S·N(-d1)
    put_price = K * math.exp(-rate * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return K, gamma_put, put_price


# ─────────────────────────────────────────────────────────────────────────────
# LP Greeks
# ─────────────────────────────────────────────────────────────────────────────

def calc_lp_greeks(S, C, range_pct):
    """
    Compute LP liquidity, delta, and gamma for a symmetric ±range_pct position.

    Formulas (from lp_greeks_model.md):
        pa   = S × (1 − range_pct)
        pb   = S × (1 + range_pct)
        L    = C / (2√S − √pa − S/√pb)
        Δ_LP = L × (1/√S − 1/√pb)      [ETH]
        Γ_LP = −L / (2 × S^1.5)        [$/($²), always ≤ 0]

    Returns:
        pa, pb, L, delta_lp, gamma_lp
    """
    pa = S * (1.0 - range_pct)
    pb = S * (1.0 + range_pct)

    sqrt_S  = math.sqrt(S)
    sqrt_pa = math.sqrt(pa)
    sqrt_pb = math.sqrt(pb)

    denom = 2.0 * sqrt_S - sqrt_pa - S / sqrt_pb
    L = C / denom

    delta_lp = L * (1.0 / sqrt_S - 1.0 / sqrt_pb)     # ETH
    gamma_lp = -L / (2.0 * S ** 1.5)                   # $/($²)

    return pa, pb, L, delta_lp, gamma_lp


# ─────────────────────────────────────────────────────────────────────────────
# Per-range row
# ─────────────────────────────────────────────────────────────────────────────

def calc_range_row(range_pct, S, C, gamma_put, put_delta_tgt, sigma):
    """
    Compute all metrics for a given symmetric LP range.

    Returns a dict with all table columns.
    """
    pa, pb, L, delta_lp, gamma_lp = calc_lp_greeks(S, C, range_pct)

    # Delta-first put sizing
    n_puts    = delta_lp / put_delta_tgt
    gamma_hdg = n_puts * gamma_put

    # Coverage and residual
    coverage  = gamma_hdg / abs(gamma_lp)           # ratio (1.0 = 100%)
    gamma_net = gamma_lp + gamma_hdg                 # ≤0 when under-hedged

    # Monthly expected ΔS² = S²·σ²/12
    expected_ds2_monthly = S ** 2 * sigma ** 2 / 12.0
    residual_drag_monthly = 0.5 * gamma_net * expected_ds2_monthly

    return {
        "range_pct":             range_pct,
        "pa":                    pa,
        "pb":                    pb,
        "L":                     L,
        "delta_lp_eth":          delta_lp,
        "gamma_lp":              gamma_lp,
        "n_puts":                n_puts,
        "gamma_hedge":           gamma_hdg,
        "coverage_pct":          coverage * 100.0,
        "gamma_net":             gamma_net,
        "residual_drag_monthly": residual_drag_monthly,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Crossover range (100% coverage) — analytical solution
# ─────────────────────────────────────────────────────────────────────────────

def find_crossover_range(S, gamma_put, put_delta_tgt):
    """
    Solve for pb such that delta-first puts achieve exactly 100% gamma coverage.

    Derivation:
        Γ_hedge = |Γ_LP|
        (Δ_LP / |Δ_put|) × Γ_put = L / (2S^1.5)
        (Δ_LP / L) × Γ_put / |Δ_put| = 1 / (2S^1.5)

        Since Δ_LP / L = (1/√S − 1/√pb):
            1/√pb = 1/√S − |Δ_put| / (Γ_put × 2S^1.5)
    """
    inv_sqrt_S  = 1.0 / math.sqrt(S)
    two_S15     = 2.0 * S ** 1.5

    inv_sqrt_pb = inv_sqrt_S - put_delta_tgt / (gamma_put * two_S15)
    pb_crossover = (1.0 / inv_sqrt_pb) ** 2
    range_crossover = (pb_crossover - S) / S

    return pb_crossover, range_crossover


# ─────────────────────────────────────────────────────────────────────────────
# Printing
# ─────────────────────────────────────────────────────────────────────────────

def print_put_params(K, gamma_put, put_price, put_delta_tgt, sigma, T_days):
    print("=" * 76)
    print("LP RANGE ANALYSIS — Delta-First Put Hedge: Gamma Coverage")
    print("=" * 76)
    print(f"  Inputs : S=${S:,.0f}  C=${C:,.0f}  σ={sigma:.0%}  T={T_days}D  r={RATE:.0%}")
    print(f"  30Δ Put: K=${K:,.1f}  Γ_put={gamma_put:.6f} /($²)/contract  price=${put_price:,.2f}")
    print()


def print_table(rows, put_price):
    hdr = (
        f"{'Range':>6}  {'pa':>6}  {'pb':>6}  {'L':>9}  {'Δ_LP':>8}  "
        f"{'Γ_LP':>8}  {'N_puts':>7}  {'Γ_hedge':>8}  {'Cover%':>7}  {'Drag/mo':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        drag_str = f"${r['residual_drag_monthly']:+,.0f}"
        coverage_flag = ""
        if abs(r['coverage_pct'] - 100.0) < 2.0:
            coverage_flag = " ←"
        elif r['coverage_pct'] >= 100.0:
            coverage_flag = " ↑"
        print(
            f"  ±{r['range_pct']:.0%}  "
            f"{r['pa']:>6,.0f}  {r['pb']:>6,.0f}  "
            f"{r['L']:>9,.0f}  "
            f"{r['delta_lp_eth']:>8.1f}  "
            f"{r['gamma_lp']:>8.4f}  "
            f"{r['n_puts']:>7.0f}  "
            f"{r['gamma_hedge']:>8.4f}  "
            f"{r['coverage_pct']:>6.1f}%{coverage_flag:<3}  "
            f"{drag_str:>10}"
        )

    print()
    # Put cost row
    print("  Put cost per contract: ${:,.0f}  (30D roll)".format(put_price))
    total_puts_range = f"{min(r['n_puts'] for r in rows):.0f}–{max(r['n_puts'] for r in rows):.0f}"
    print(f"  N_puts range across all rows: {total_puts_range}  "
          f"(Δ_LP nearly flat; gamma coverage varies 5×)")


def print_crossover(pb_cross, range_cross):
    print()
    print(f"  Crossover range (100% coverage): ≈±{range_cross:.1%}  "
          f"(pb ≈ ${pb_cross:,.0f})")
    print(f"  → CORE range (−20%/+25%) lands just above crossover → ~114% coverage.")
    print()


def print_insights():
    print("=" * 76)
    print("KEY INSIGHTS")
    print("=" * 76)
    print("""
  1. Δ_LP is nearly range-insensitive (−12% from ±5% to ±30%) because
     the two effects (L↓ and (1/√S−1/√pb)↑ as range widens) partially cancel.

  2. Γ_LP tracks L exactly (Γ = −L/2S^1.5), so it drops 83% as range widens.

  3. N_puts ~ constant (~750–815), Γ_hedge ~ constant (~0.58–0.66).
     Coverage varies 5× purely because Γ_LP varies.

  4. Crossover ≈ ±21.4%: at this range, delta-first = simultaneously
     delta-neutral + gamma-neutral. CORE range (−20%/+25%) was calibrated here.

  5. Below crossover (<±21%): gamma structurally under-hedged with puts alone.
     Perp+puts can close the gap but costs more than the gamma drag saved.

  6. ±20%: 94% coverage, ~$4k/mo residual drag — practically gamma-neutral.
     ±10%: 50% coverage, ~$60k/mo drag — requires dynamic fee to compensate.
""")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Fixed put parameters
    K, gamma_put, put_price = calc_put_params(S, SIGMA, T_DAYS, RATE, PUT_DELTA_TGT)

    print_put_params(K, gamma_put, put_price, PUT_DELTA_TGT, SIGMA, T_DAYS)

    # Compute all rows
    rows = [
        calc_range_row(r, S, C, gamma_put, PUT_DELTA_TGT, SIGMA)
        for r in RANGES
    ]

    print_table(rows, put_price)

    # Crossover
    pb_cross, range_cross = find_crossover_range(S, gamma_put, PUT_DELTA_TGT)
    print_crossover(pb_cross, range_cross)

    # Standard example cross-check (−20%/+25% asymmetric)
    print("  Cross-check: standard example (−20%/+25%, pa=1600, pb=2500)")
    pa_std, pb_std = 1600.0, 2500.0
    sqrt_S, sqrt_pa, sqrt_pb = math.sqrt(S), math.sqrt(pa_std), math.sqrt(pb_std)
    L_std = C / (2 * sqrt_S - sqrt_pa - S / sqrt_pb)
    dLP_std = L_std * (1 / sqrt_S - 1 / sqrt_pb)
    gLP_std = -L_std / (2 * S ** 1.5)
    n_std = dLP_std / PUT_DELTA_TGT
    g_hdg_std = n_std * gamma_put
    cov_std = g_hdg_std / abs(gLP_std)
    print(f"    L={L_std:,.0f}  Δ_LP={dLP_std:.1f} ETH  Γ_LP={gLP_std:.4f}  "
          f"N_puts={n_std:.0f}  Coverage={cov_std:.1%}")
    print()

    print_insights()


if __name__ == "__main__":
    main()
