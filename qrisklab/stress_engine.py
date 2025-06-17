import pandas as pd
import datetime
import numpy as np
from .core.stress_tests import StressTest2
from .core.vol_models.svi_model import SVIModel
from .utils import process_instruments


class StressTestEngine:
    def __init__(self, mode='sticky_delta', vol_model='SVI', time_to_expiry_offset=-1 / 365) -> None:
        self.mode = mode
        self.positions = pd.DataFrame()
        self.vol_model = vol_model
        # TODO:  enable Wing Model
        self.time_to_expiry_offset = time_to_expiry_offset

    def set_mode(self, mode):
        if mode in ['sticky_delta', 'sticky_strike']:
            self.mode = mode
        else:
            print(f"Mode {mode}: Not Supported")

    def set_valuation_datetime(self, dt_str=None):
        # valuation_datetime: e.g. "2025-06-17 13:45:30"
        # Always use UTC
        if dt_str is None:
            self.valuation_datetime = datetime.datetime.now(datetime.UTC)
        else:
            self.valuation_datetime = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")

    def update_vol_curves(self, df_market_data):
        """
        df_market_data: Options market data to fit SVI Model
        """
        missing_columns = ['IV', 'Strike', 'Date', 'Tau', 'F', 'S'] - set(df_market_data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        df_market_data = df_market_data[df_market_data['Tau'] > 0]
        svim = SVIModel(df_market_data[['IV', 'Strike', 'Date', 'Tau', 'F']].drop_duplicates(['IV', 'Strike', 'Date']))
        df_fit = svim.fit(no_butterfly=False, no_calendar=False).reset_index(
            drop=False).rename(columns={'index': 'Date'})
        df_fit = pd.merge(df_fit, df_market_data[['Date', 'F', 'time_to_expiry']].drop_duplicates(
            subset=['Date'], keep='first'), on='Date', how='left')
        df_fit['tt'] = df_fit['time_to_expiry']
        self.svi_df = df_fit

    def update_positions(self, df_pos, ticker):

        missing_columns = ['instrument', 'multiplier', 'quantity'] - set(df_pos.columns)
        if missing_columns:
            raise ValueError(f"Missing columns: {', '.join(missing_columns)}")

        print('Process Positions')
        df_pos = self._process_positions(df_pos).copy()
        df_pos = df_pos[df_pos['underlying'].str.contains(ticker)]
        df_pos['underlying'] = ticker
        df_pos = df_pos[df_pos['time_to_expiry'] != 0]
        self.positions = df_pos.copy()

    def shock(self, spot_shock, vol_shock, name='stress_pnl'):
        df = self.positions.copy()
        df['spot_shock'] = spot_shock
        df['spot'] = self.svi_df['S'].values[0]
        df['post_shock_spot'] = df['spot'] * (1 + df['spot_shock'])

        df[['post_shock_vol', 'vol']] = df.apply(
            lambda row: pd.Series(self._get_post_shock_vol(row)), axis=1
        )
        df['vol_shock'] = vol_shock
        df['post_shock_vol'] = (1 + vol_shock) * df['post_shock_vol']

        st = StressTest2(time_to_expiry_offset=self.time_to_expiry_offset)
        re = st.shock_df(df, name).reset_index(drop=True)
        re['shock_mode'] = self.mode
        return re

    def _calc_svi_vol(self, strike, a, b, rho, m, sigma, F, tt):
        d = rho * b * sigma
        c = b * sigma
        y = (np.log(strike / F) - m) / sigma
        svi_vol = np.sqrt((a + d * y + c * np.sqrt(y**2 + 1)) / tt)
        return svi_vol

    def _get_post_shock_vol(self, row):
        strike = row['strike']
        time_to_expiry = row['time_to_expiry']
        if np.isnan(strike) or np.isnan(time_to_expiry):
            return (None, None)

        spot = row['spot']
        post_shock_spot = row['post_shock_spot']
        if self.mode == 'sticky_delta':
            post_shock_moneyness = post_shock_spot / spot
            post_shock_strike = spot / post_shock_moneyness
        elif self.mode == 'sticky_strike':
            post_shock_strike = strike

        self.svi_df['ivol'] = self.svi_df.apply(
            lambda row: self._calc_svi_vol(
                strike,
                row["a"],
                row["b"],
                row["rho"],
                row["m"],
                row["sigma"],
                row["F"],
                row["tt"]
            ),
            axis=1
        )

        self.svi_df['post_shock_vol'] = self.svi_df.apply(
            lambda row: self._calc_svi_vol(
                post_shock_strike,
                row["a"],
                row["b"],
                row["rho"],
                row["m"],
                row["sigma"],
                row["F"],
                row["tt"]
            ),
            axis=1
        )

        re = self.svi_df.iloc[(self.svi_df['tt'] - time_to_expiry).abs().idxmin()]
        post_shock_vol = re['post_shock_vol']
        vol = re['ivol']
        return (post_shock_vol, vol)

    def _process_positions(self, df):

        df = df.dropna(how='all')

        df = process_instruments(
            df, self.valuation_datetime.strftime('%Y-%m-%d')
        ).reset_index(drop=True)

        # calculate other inputs for BSM
        risk_free_rate = 0.0
        df['cost_of_carry_rate'] = 'default'
        df['rate'] = risk_free_rate
        df['snapshot_timestamp'] = int(self.valuation_datetime.timestamp() * 1e3)
        df['valuation_datetime'] = self.valuation_datetime.strftime('%Y-%m-%d %H:%M:%S')
        df['valuation_date'] = self.valuation_datetime.strftime('%Y-%m-%d')
        return df
