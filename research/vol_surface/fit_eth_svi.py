"""
Fit the current ETH implied volatility surface using SVI (Stochastic Volatility Inspired) model.

Fetches live option data from Deribit, transforms it into the format required by
qrisklab's SVIModel, fits arbitrage-free smiles per expiry, and visualises the result.
"""

import argparse
import datetime
import sys

import numpy as np
import pandas as pd

from qrisklab.clients.deribit import DeribitClient
from qrisklab.core.vol_models.svi_model import SVIModel, SVIPlot
from qrisklab.utils import process_instruments


def fetch_eth_options(client: DeribitClient) -> pd.DataFrame:
    """Fetch live ETH option chain from Deribit public API."""
    df = client.fetch_deribit_option_data(currency="ETH", kind="option")
    if df.empty:
        raise RuntimeError("No data returned from Deribit — check connectivity.")
    return df


def prepare_svi_input(
    df_raw: pd.DataFrame,
    valuation_date: str,
    min_iv: float = 1.0,
    min_open_interest: float = 0.0,
) -> pd.DataFrame:
    """
    Transform raw Deribit option data into the DataFrame expected by SVIModel.

    SVIModel requires columns: IV, Strike, Date, Tau, F
      - IV   : implied vol in percent (e.g. 75.0 for 75%)
      - Strike: absolute strike price
      - Date : expiry as pd.Timestamp
      - Tau  : time to maturity in years
      - F    : forward price for that expiry
    """
    df = df_raw.copy()
    df["instrument"] = df["instrument_name"]

    df = process_instruments(df, valuation_date)

    # Calls only — SVI fits on calls; by put-call parity the IV is identical
    df = df[df["put_call"] == "call"].copy()

    # Filter out zero / negative time-to-expiry (already expired)
    df = df[df["time_to_expiry"] > 0]

    # Filter out options with very low IV (illiquid / stale quotes)
    df = df[df["mark_iv"] >= min_iv]

    # Optional: filter by open interest
    if min_open_interest > 0:
        df = df[df["open_interest"] >= min_open_interest]

    df["IV"] = df["mark_iv"].astype(float)
    df["Strike"] = df["strike"].astype(float)
    df["Date"] = pd.to_datetime(df["expiry"])
    df["Tau"] = df["time_to_expiry"].astype(float)
    df["F"] = df["forward_price"].astype(float)

    df = df.dropna(subset=["IV", "Strike", "Date", "Tau", "F"])

    return df


def fit_svi(
    df: pd.DataFrame,
    no_butterfly: bool = True,
    no_calendar: bool = True,
    min_fit: int = 5,
) -> tuple[SVIModel, pd.DataFrame]:
    """
    Fit SVI model to prepared data.

    Returns
    -------
    svim : SVIModel  — fitted model object (needed for plotting)
    df_fit : DataFrame — calibrated parameters per expiry
    """
    svi_input = df[["IV", "Strike", "Date", "Tau", "F"]].drop_duplicates(
        ["IV", "Strike", "Date"]
    )
    svim = SVIModel(svi_input, min_fit=min_fit)
    df_fit = (
        svim.fit(no_butterfly=no_butterfly, no_calendar=no_calendar)
        .reset_index(drop=False)
        .rename(columns={"index": "Date"})
    )

    df_fit = pd.merge(
        df_fit,
        df[["Date", "F", "time_to_expiry"]]
        .drop_duplicates(subset=["Date"], keep="first"),
        on="Date",
        how="left",
    )
    return svim, df_fit


def print_calibration_report(df_fit: pd.DataFrame) -> None:
    """Pretty-print calibrated SVI parameters per expiry."""
    print("\n" + "=" * 80)
    print("SVI CALIBRATION REPORT — ETH")
    print("=" * 80)
    pd.set_option("display.float_format", "{:,.6f}".format)
    display_cols = ["Date", "a", "b", "rho", "m", "sigma", "F", "time_to_expiry"]
    available = [c for c in display_cols if c in df_fit.columns]
    print(df_fit[available].to_string(index=False))
    print("=" * 80 + "\n")


def plot_surface(svim: SVIModel) -> None:
    """Render interactive 3-D vol surface via Plotly."""
    plotter = SVIPlot()
    plotter.allsmiles(svim)


def main():
    parser = argparse.ArgumentParser(
        description="Fit ETH implied vol surface with SVI model using live Deribit data."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=datetime.date.today().isoformat(),
        help="Valuation date (YYYY-MM-DD). Default: today.",
    )
    parser.add_argument(
        "--no-butterfly",
        action="store_true",
        default=True,
        help="Enforce no-butterfly-arbitrage constraint (default: True).",
    )
    parser.add_argument(
        "--no-calendar",
        action="store_true",
        default=True,
        help="Enforce no-calendar-arbitrage constraint (default: True).",
    )
    parser.add_argument(
        "--min-fit",
        type=int,
        default=5,
        help="Minimum strikes per smile to include in fit (default: 5).",
    )
    parser.add_argument(
        "--min-iv",
        type=float,
        default=1.0,
        help="Minimum mark IV (%%) to include (filters stale quotes).",
    )
    parser.add_argument(
        "--min-oi",
        type=float,
        default=0.0,
        help="Minimum open interest to include (default: 0 = no filter).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the 3-D surface plot.",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Save calibrated parameters to this CSV path.",
    )
    args = parser.parse_args()

    print(f"Valuation date: {args.date}")
    print("Fetching live ETH option data from Deribit...")

    client = DeribitClient()
    df_raw = fetch_eth_options(client)
    print(f"  {len(df_raw)} instruments fetched.")

    print("Preparing data for SVI fitting...")
    df_prepared = prepare_svi_input(
        df_raw,
        valuation_date=args.date,
        min_iv=args.min_iv,
        min_open_interest=args.min_oi,
    )
    n_expiries = df_prepared["Date"].nunique()
    n_strikes = len(df_prepared)
    print(f"  {n_strikes} call options across {n_expiries} expiries after filtering.")

    print("Fitting SVI model...")
    svim, df_fit = fit_svi(
        df_prepared,
        no_butterfly=args.no_butterfly,
        no_calendar=args.no_calendar,
        min_fit=args.min_fit,
    )

    print_calibration_report(df_fit)

    if args.save_csv:
        df_fit.to_csv(args.save_csv, index=False)
        print(f"Parameters saved to {args.save_csv}")

    if not args.no_plot:
        print("Rendering 3-D vol surface...")
        plot_surface(svim)


if __name__ == "__main__":
    main()
