import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds, NonlinearConstraint
from uuid import uuid4

# import plotly.graph_objects as go
from typing import Tuple, Optional
from datetime import datetime

import plotly.graph_objs as go
from ipywidgets import (
    FloatSlider, FloatText, Layout, HBox, VBox,
    interactive_output, Button, Output
)


class WingModel:
    def __init__(self):
        self.param_widgets = {}
        self.market_data = None
        self.forward = None

    # --- Wing vol curve model ---
    @staticmethod
    def wing_vol_curve_with_smoothing(strikes,
                                      F,              # Forward price used to define log-moneyness
                                      vr,             # Volatility reference at central skew point
                                      sr,             # Slope reference at central skew point
                                      pc,             # Put curvature: bend on put wing (left of ATM)
                                      cc,             # Call curvature: bend on call wing (right of ATM)
                                      dc,             # Down cutoff (log-moneyness, negative value)
                                      uc,             # Up cutoff (log-moneyness, positive value)
                                      dsm=0.5,        # Down smoothing range beyond dc
                                      usm=0.5,        # Up smoothing range beyond uc
                                      VCR=0.0,        # Volatility change rate w.r.t forward move
                                      SCR=0.0,        # Slope change rate w.r.t forward move
                                      SSR=100,        # Skew swimmingness rate (0 = fixed, 100 = ATM follows F)
                                      Ref=None,       # Reference price (for fixed skew)
                                      ATM=None        # ATM forward price (for floating skew)
    ):
        """
        Generate a volatility curve using the Orc Wing Model with parabolic smoothing.

        This version reproduces Orc's exact smoothing style:
        - Parabolic skew on both put and call wings.
        - Smooth cubic (Hermite-style) interpolation in the smoothing zones.
        - Flat extrapolation beyond the smoothing cutoffs.

        Parameters:
            strikes (array): Strike prices to compute volatility for.
            F (float): Forward price for normalization (used to compute log-moneyness).
            vr (float): Volatility reference at the central skew point.
            sr (float): Slope reference at the central skew point.
            pc (float): Curvature for the put wing (left of ATM).
            cc (float): Curvature for the call wing (right of ATM).
            dc (float): Down cutoff in log-moneyness where put wing ends.
            uc (float): Up cutoff in log-moneyness where call wing ends.
            dsm (float): Down smoothing width beyond dc (default 0.5).
            usm (float): Up smoothing width beyond uc (default 0.5).
            VCR (float): Volatility change rate when forward deviates from reference.
            SCR (float): Slope change rate when forward deviates from reference.
            SSR (float): Skew swimmingness rate (0 = fixed skew, 100 = ATM anchored).
            Ref (float): Reference forward price.
            ATM (float): ATM forward used in floating skew calculation.


        Notes on Skew Swimmingness Rate (SSR):
            Definition: SSR (ranging from 0 to 100) controls the degree to which the volatility curve’s ATM point is anchored to the forward price (F) versus the reference price (Ref). It is expressed as a percentage, where:
            SSR = 0: The curve is fully fixed to Ref (fixed skew).
            SSR = 100: The curve is fully anchored to F via ATM (floating skew).
            0 < SSR < 100: The curve interpolates between Ref and ATM.

            Fixed Skew (SSR = 0) ≈ Sticky Strike: The volatility curve is anchored to Ref, keeping volatilities constant per strike (with VCR/SCR adjustments), suitable for stable markets or long-term options.
            Floating Skew (SSR = 100) = Sticky Delta: The curve shifts with F, keeping volatilities constant for fixed log-moneyness or delta, ideal for dynamic markets like cryptocurrencies.
            Hybrid SSR: Offers a unique blend, not directly equivalent to sticky delta or sticky strike, allowing partial curve shifts for flexible modeling.
            Choosing the Model: Use sticky delta/floating skew for short-term, volatile markets (e.g., BTC options on Deribit). Use sticky strike/fixed skew for stable markets or when modeling fixed volatility structures. Test hybrid SSR for intermediate cases.


        Returns:
            strikes (array): The original strike array.
            vols (array): Computed implied volatilities corresponding to the strikes.
        """
        ssr_frac = SSR / 100.0
        F_eff = F if SSR == 100 else (Ref if SSR == 0 else (1 - ssr_frac) * Ref + ssr_frac * ATM)
        vc = vr - VCR * ssr_frac * (F_eff - Ref) / Ref
        sc = sr - SCR * ssr_frac * (F_eff - Ref) / Ref

        x = np.log(strikes / F_eff)
        vols = np.zeros_like(x)

        xL0, xL1 = dc - dsm, dc
        yL0 = vc + sc * xL0 + pc * xL0**2
        yL1 = vc + sc * xL1 + pc * xL1**2
        dyL0 = sc + 2 * pc * xL0
        dyL1 = sc + 2 * pc * xL1

        xR0, xR1 = uc, uc + usm
        yR0 = vc + sc * xR0 + cc * xR0**2
        yR1 = vc + sc * xR1 + cc * xR1**2
        dyR0 = sc + 2 * cc * xR0
        dyR1 = sc + 2 * cc * xR1

        for i, xi in enumerate(x):
            if xi < xL0:
                # Far left (x < dc - dsm): Flat extrapolation using the volatility and slope at the down cutoff.
                vols[i] = yL0 + dyL0 * (xi - xL0)
            elif xL0 <= xi < xL1:
                # Left smoothing zone (dc - dsm ≤ x < dc): Cubic Hermite interpolation for smooth transition.
                t = (xi - xL0) / (xL1 - xL0)
                h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
                h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
                vols[i] = h00 * yL0 + h10 * (xL1 - xL0) * dyL0 + h01 * yL1 + h11 * (xL1 - xL0) * dyL1
            elif dc <= xi < 0:
                # Put wing (dc ≤ x < 0): Parabolic skew
                vols[i] = vc + sc * xi + pc * xi ** 2
            elif 0 <= xi <= uc:
                # Call wing (0 ≤ x ≤ uc): Parabolic skew
                vols[i] = vc + sc * xi + cc * xi ** 2
            elif xR0 < xi <= xR1:
                # Right smoothing zone (uc < x ≤ uc + usm): Cubic Hermite interpolation.
                t = (xi - xR0) / (xR1 - xR0)
                h00, h10 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t
                h01, h11 = -2 * t**3 + 3 * t**2, t**3 - t**2
                vols[i] = h00 * yR0 + h10 * (xR1 - xR0) * dyR0 + h01 * yR1 + h11 * (xR1 - xR0) * dyR1
            else:
                # Far right (x > uc + usm): Flat extrapolation.
                vols[i] = yR1 + dyR1 * (xi - xR1)

        return strikes, vols

    # --- Fit function ---
    @staticmethod
    def fit_wing_model_to_data(strikes, market_vols, F_init, Ref_init=None, ATM_init=None, fit_vcr_scr_ssr=True, fit_f_ref_atm=False):
        Ref_init = Ref_init or F_init
        ATM_init = ATM_init or F_init

        x_moneyness = np.log(strikes / F_init)
        weights = 1 / (np.abs(x_moneyness) + 0.05)

        def unpack_params(p):
            idx = 0
            keys = ["vr", "sr", "pc", "cc", "dc", "uc", "dsm", "usm"]
            vals = dict(zip(keys, p[idx:idx + 8]))
            idx += 8
            if fit_vcr_scr_ssr:
                vals.update(dict(VCR=p[idx], SCR=p[idx + 1], SSR=p[idx + 2]))
                idx += 3
            else:
                vals.update(dict(VCR=0.0, SCR=0.0, SSR=100.0))
            if fit_f_ref_atm:
                vals.update(dict(F=p[idx], Ref=p[idx + 1], ATM=p[idx + 2]))
            else:
                vals.update(dict(F=F_init, Ref=Ref_init, ATM=ATM_init))
            return vals

        def objective(p):
            params = unpack_params(p)
            _, vols = WingModel.wing_vol_curve_with_smoothing(strikes=strikes, **params)
            return np.mean(weights * (vols - market_vols) ** 2)

        p0 = [0.2, -0.1, 1.5, 1.0, -0.3, 0.3, 0.4, 0.4]
        bounds = [(0.05, 1.0), (-2.0, 2.0), (0.01, 50.0), (0.01, 50.0),
                  (-1.5, -0.01), (0.01, 1.5), (0.01, 2.0), (0.01, 2.0)]

        if fit_vcr_scr_ssr:
            p0 += [0.0, 0.0, 100.0]
            bounds += [(-2.0, 2.0), (-2.0, 2.0), (0, 100)]

        if fit_f_ref_atm:
            p0 += [F_init, Ref_init, ATM_init]
            bounds += [(F_init * 0.9, F_init * 1.1)] * 3

        result = minimize(objective, p0, bounds=bounds, method='L-BFGS-B')
        return unpack_params(result.x)

    # --- Helper: slider + input box ---
    def float_slider_text(self, label, minv, maxv, step, value):
        slider = FloatSlider(min=minv, max=maxv, step=step, value=value,
                             description=label, continuous_update=False, layout=Layout(width='60%'))
        text = FloatText(value=value, layout=Layout(width='80px'))

        def on_slider_change(change): text.value = change['new']
        def on_text_change(change): slider.value = change['new']

        slider.observe(on_slider_change, names='value')
        text.observe(on_text_change, names='value')

        return slider, text

    # --- Market Data Setter ---
    def set_market_data(self, strikes, vols):
        self.market_data = (strikes, vols)

    def set_forward(self, set_forward):
        self.forward = set_forward

    # --- Plot ---
    def interactive_wing_plot_with_data(self, vr, sr, pc, cc, dc, uc, dsm, usm, VCR, SCR, SSR, F, Ref, ATM):
        if self.market_data is not None:
            mk_strikes, mk_vols = self.market_data
            strike_min = np.min(mk_strikes)
            strike_max = np.max(mk_strikes)
        else:
            strike_min, strike_max = 30, 180
            mk_strikes, mk_vols = [], []

        strikes = np.linspace(strike_min, strike_max, 300)
        _, vols = self.wing_vol_curve_with_smoothing(
            strikes, F, vr, sr, pc, cc, dc, uc, dsm, usm,
            VCR, SCR, SSR, Ref, ATM
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=strikes, y=vols, mode='lines', name="Wing Model Vol"))

        if len(mk_strikes) > 0:
            fig.add_trace(go.Scatter(x=mk_strikes, y=mk_vols, mode='markers', name="Market Data",
                                     marker=dict(color="red", size=6, symbol="circle")))

        fig.update_layout(title="Wing Volatility Curve Fit (Interactive)",
                          xaxis_title="Strike", yaxis_title="Implied Volatility",
                          template="plotly_white", hovermode="x unified")
        fig.show()

    # --- UI ---
    def create_interactive_wing_fit_ui(self):
        defs = {
            'vr': ("Vol Ref", 0.05, 0.95, 0.01, 0.20),
            'sr': ("Slope", -2.0, 2.0, 0.1, 0.0),
            'pc': ("Put Curv", 0.0, 50.0, 0.1, 2.0),
            'cc': ("Call Curv", 0.0, 50.0, 0.1, 1.0),
            'dc': ("Down Cut", -1.0, 0.0, 0.05, -0.2),
            'uc': ("Up Cut", 0.0, 1.0, 0.05, 0.2),
            'dsm': ("Down Sm", 0.0, 2.0, 0.1, 0.5),
            'usm': ("Up Sm", 0.0, 2.0, 0.1, 0.5),
            'VCR': ("VCR", -2.0, 2.0, 0.1, 0.0),
            'SCR': ("SCR", -2.0, 2.0, 0.1, 0.0),
            'SSR': ("SSR", 0, 100, 1, 100),
            'F': ("Forward", 80, 120, 1, 100),
            'Ref': ("Ref", 80, 120, 1, 100),
            'ATM': ("ATM", 80, 120, 1, 100),
        }

        if self.market_data is not None:
            mk_strikes, _ = self.market_data
            s_min, s_max = np.min(mk_strikes), np.max(mk_strikes)
            if self.forward is None:
                self.forward = (s_min + s_max) / 2
            defs['F'] = ("Forward", s_min, s_max, 1, self.forward)
            defs['Ref'] = ("Ref", s_min, s_max, 1, self.forward)
            defs['ATM'] = ("ATM", s_min, s_max, 1, self.forward)

        widget_boxes = []
        for k, v in defs.items():
            s, t = self.float_slider_text(*v)
            self.param_widgets[k] = (s, t)
            widget_boxes.append(HBox([s, t]))

        plot_out = interactive_output(self.interactive_wing_plot_with_data, {
                                      k: v[0] for k, v in self.param_widgets.items()})
        fit_button = Button(description="Fit to Market Data", button_style="success")

        def on_fit_click(b):
            if self.market_data is None:
                print("⚠️ Market data not loaded!")
                return
            mk_strikes, mk_vols = self.market_data
            F = self.param_widgets['F'][0].value
            Ref = self.param_widgets['Ref'][0].value
            ATM = self.param_widgets['ATM'][0].value
            fitted = self.fit_wing_model_to_data(mk_strikes, mk_vols, F, Ref, ATM)
            for k, v in fitted.items():
                self.param_widgets[k][0].value = v
                self.param_widgets[k][1].value = v

        fit_button.on_click(on_fit_click)
        display(fit_button, VBox(widget_boxes), plot_out)


##################################################################
# Functions
##################################################################

def wing_vol_curve_with_smoothing(
    strikes,
    F,              # Forward price used to define log-moneyness
    vr,             # Volatility reference at central skew point
    sr,             # Slope reference at central skew point
    pc,             # Put curvature: bend on put wing (left of ATM)
    cc,             # Call curvature: bend on call wing (right of ATM)
    dc,             # Down cutoff (log-moneyness, negative value)
    uc,             # Up cutoff (log-moneyness, positive value)
    dsm=0.5,        # Down smoothing range beyond dc
    usm=0.5,        # Up smoothing range beyond uc
    VCR=0.0,        # Volatility change rate w.r.t forward move
    SCR=0.0,        # Slope change rate w.r.t forward move
    SSR=100,        # Skew swimmingness rate (0 = fixed, 100 = ATM follows F)
    Ref=None,       # Reference price (for fixed skew)
    ATM=None        # ATM forward price (for floating skew)
):
    """
    Generate a volatility curve using the Orc Wing Model with parabolic smoothing.

    This version reproduces Orc's exact smoothing style:
    - Parabolic skew on both put and call wings.
    - Smooth cubic (Hermite-style) interpolation in the smoothing zones.
    - Flat extrapolation beyond the smoothing cutoffs.

    Parameters:
        strikes (array): Strike prices to compute volatility for.
        F (float): Forward price for normalization (used to compute log-moneyness).
        vr (float): Volatility reference at the central skew point.
        sr (float): Slope reference at the central skew point.
        pc (float): Curvature for the put wing (left of ATM).
        cc (float): Curvature for the call wing (right of ATM).
        dc (float): Down cutoff in log-moneyness where put wing ends.
        uc (float): Up cutoff in log-moneyness where call wing ends.
        dsm (float): Down smoothing width beyond dc (default 0.5).
        usm (float): Up smoothing width beyond uc (default 0.5).
        VCR (float): Volatility change rate when forward deviates from reference.
        SCR (float): Slope change rate when forward deviates from reference.
        SSR (float): Skew swimmingness rate (0 = fixed skew, 100 = ATM anchored).
        Ref (float): Reference forward price.
        ATM (float): ATM forward used in floating skew calculation.

    Returns:
        strikes (array): The original strike array.
        vols (array): Computed implied volatilities corresponding to the strikes.
    """
    # Convert SSR to a fractional weight (0 to 1)
    ssr_frac = SSR / 100.0

    # Compute effective forward (F_eff) based on SSR blend between Ref and ATM
    if SSR == 0:
        F_eff = Ref
    elif SSR == 100:
        F_eff = ATM
    else:
        F_eff = (1 - ssr_frac) * Ref + ssr_frac * ATM

    # Central volatility adjusted by forward deviation from reference
    vc = vr - VCR * ssr_frac * (F_eff - Ref) / Ref

    # Central slope adjusted by forward deviation
    sc = sr - SCR * ssr_frac * (F_eff - Ref) / Ref

    # Convert strike to log-moneyness x = ln(K/F_eff)
    x = np.log(strikes / F_eff)
    vols = np.zeros_like(x)

    # Precompute Hermite endpoints
    xL0, xL1 = dc - dsm, dc
    yL0 = vc + sc * xL0 + pc * xL0 ** 2
    yL1 = vc + sc * xL1 + pc * xL1 ** 2
    dyL0 = sc + 2 * pc * xL0
    dyL1 = sc + 2 * pc * xL1

    xR0, xR1 = uc, uc + usm
    yR0 = vc + sc * xR0 + cc * xR0 ** 2
    yR1 = vc + sc * xR1 + cc * xR1 ** 2
    dyR0 = sc + 2 * cc * xR0
    dyR1 = sc + 2 * cc * xR1

    for i, xi in enumerate(x):
        if xi < xL0:
            # Extrapolate left of smoothing using tangent at xL0
            vols[i] = yL0 + dyL0 * (xi - xL0)

        elif xL0 <= xi < xL1:
            # Hermite interpolation in down smoothing zone
            t = (xi - xL0) / (xL1 - xL0)
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            vols[i] = h00 * yL0 + h10 * (xL1 - xL0) * dyL0 + h01 * yL1 + h11 * (xL1 - xL0) * dyL1

        elif dc <= xi < 0:
            # Put wing
            vols[i] = vc + sc * xi + pc * xi ** 2

        elif 0 <= xi <= uc:
            # Call wing
            vols[i] = vc + sc * xi + cc * xi ** 2

        elif xR0 < xi <= xR1:
            # Hermite interpolation in up smoothing zone
            t = (xi - xR0) / (xR1 - xR0)
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            vols[i] = h00 * yR0 + h10 * (xR1 - xR0) * dyR0 + h01 * yR1 + h11 * (xR1 - xR0) * dyR1

        elif xi > xR1:
            # Extrapolate right of smoothing using tangent at xR1
            vols[i] = yR1 + dyR1 * (xi - xR1)

    return strikes, vols


def fit_wing_model_to_data(
    strikes, market_vols, F_init, Ref_init=None, ATM_init=None,
    fit_vcr_scr_ssr=True, fit_f_ref_atm=False
):
    Ref_init = Ref_init or F_init
    ATM_init = ATM_init or F_init

    x_moneyness = np.log(strikes / F_init)
    weights = 1 / (np.abs(x_moneyness) + 0.05)

    def unpack_params(p):
        idx = 0
        keys = ["vr", "sr", "pc", "cc", "dc", "uc", "dsm", "usm"]
        vals = dict(zip(keys, p[idx:idx + 8]))
        idx += 8
        if fit_vcr_scr_ssr:
            vals.update(dict(VCR=p[idx], SCR=p[idx + 1], SSR=p[idx + 2]))
            idx += 3
        else:
            vals.update(dict(VCR=0.0, SCR=0.0, SSR=100.0))
        if fit_f_ref_atm:
            vals.update(dict(F=p[idx], Ref=p[idx + 1], ATM=p[idx + 2]))
        else:
            vals.update(dict(F=F_init, Ref=Ref_init, ATM=ATM_init))
        return vals

    def objective(p):
        params = unpack_params(p)
        _, vols = wing_vol_curve_with_smoothing(
            strikes=strikes,
            **params
        )
        return np.mean(weights * (vols - market_vols) ** 2)

    # Initial guess
    p0 = [
        0.2,   # vr
        -0.1,  # sr
        1.5,   # pc
        1.0,   # cc
        -0.3,  # dc
        0.3,   # uc
        0.4,   # dsm
        0.4,   # usm
    ]
    bounds = [
        (0.05, 1.0),   # vr
        (-2.0, 2.0),   # sr
        (0.01, 5.0),   # pc
        (0.01, 5.0),   # cc
        (-1.5, -0.01),  # dc
        (0.01, 1.5),   # uc
        (0.01, 2.0),   # dsm
        (0.01, 2.0),   # usm
    ]

    if fit_vcr_scr_ssr:
        p0 += [0.0, 0.0, 100.0]
        bounds += [(-2.0, 2.0), (-2.0, 2.0), (0, 100)]

    if fit_f_ref_atm:
        p0 += [F_init, Ref_init, ATM_init]
        bounds += [(F_init * 0.9, F_init * 1.1)] * 3

    result = minimize(objective, p0, bounds=bounds, method='L-BFGS-B')
    return unpack_params(result.x)
