from __future__ import annotations
from datetime import datetime, date, timedelta
import pandas as pd
import numpy as np
import os


def parse_date(date_str):
    # Function to convert the dates to 'yyyy-mm-dd'

    if date_str is None:
        return None  # Return None if the input is None
    try:
        if any(c.isalpha() for c in date_str):  # Check if there are any letters in the string
            # Parse dates like '6AUG24' or '16AUG24'
            return datetime.strptime(date_str, '%d%b%y').strftime('%Y-%m-%d')
        else:
            # Parse dates like '241207'
            return datetime.strptime(date_str, '%y%m%d').strftime('%Y-%m-%d')
    except ValueError:
        return None  # Handle any parsing errors


def process_instruments(df: pd.DataFrame, valuation_day: str):
    """
        input: df[['instrument', ......]]
            instrument example: BTC-25OCT24-84000-C
        output: df[['instrument', 'underlying-expiry', 'underlying', 
            'expiry', 'strike', 'put_call', 'time_to_expiry', ......]]
    """
    # Splitting the 'instrument' column by "-"
    df[['underlying', 'expiry', 'strike', 'put_call']] = df['instrument'].str.split('-', expand=True)
    df['strike'] = df['strike'].astype(float)
    df['put_call'] = df['put_call'].map({'P': 'put', 'C': 'call'})
    # df['expiry'] = df['expiry'].replace('PERPETUAL', None)  # e.g. BTC-PERPETUAL
    df['expiry'] = df['expiry'].apply(parse_date)
    df['expiry'] = pd.to_datetime(df['expiry'], format='%Y-%m-%d')
    df['underlying-expiry'] = df['underlying'] + "-" + df['expiry'].dt.strftime('%-d%b%y')
    # e.g. BTC-8NOV24, use the %-d to avoid zero-padding
    df['underlying-expiry'] = df['underlying-expiry'].apply(lambda x: x.upper() if isinstance(x, str) else x)
    # Calculate time to expiry
    valuation_day = pd.to_datetime(valuation_day)
    df['time_to_expiry'] = (df['expiry'] - valuation_day).dt.days / 365
    # convert expiry to a string
    df['expiry'] = df['expiry'].dt.strftime('%Y-%m-%d')

    return df


def save_excel(df, file_path):
    """
    Saves a Pandas DataFrame to an Excel file, creating the directory if it doesn't exist.
    """
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df.to_excel(file_path, index=False)


def generate_timestamps(start_time='2024-12-01 00:00:00', end_time='2024-12-27 00:00:00', frequency='1h'):
    start_time = pd.to_datetime(start_time).floor('min')
    end_time = pd.to_datetime(end_time).floor('min')
    timestamps = pd.date_range(start=start_time, end=end_time, freq=frequency)
    unix_timestamps_ms = [int(timestamp.timestamp() * 1000) for timestamp in timestamps]
    return unix_timestamps_ms


def net(df, all_flds, group_by):
    df_net = df[all_flds].copy()
    df_net = df_net.groupby(group_by, as_index=False).sum()
    return df_net.reset_index(drop=True)


def get_last_businessdate():
    if date.today().weekday() == 0:
        last_businessdate = date.today() - timedelta(days=3)
    else:
        last_businessdate = date.today() - timedelta(days=1)
    return last_businessdate


def workday(start_date: date, days: int) -> date:
    '''
        Similar to Excel WORKDAY().
    '''
    # start_date.weekday()==0 for Monday (5,6) for weekends
    weekends = (5, 6)
    mod_days = abs(days) % 5
    if days < 0:
        while start_date.weekday() in weekends:
            start_date += timedelta(days=1)
        if start_date.weekday() - mod_days < 0:
            end_date = start_date - timedelta(days=mod_days + 2)
        else:
            end_date = start_date - timedelta(days=mod_days)
        end_date = end_date - timedelta(days=7 * (abs(days) - mod_days) / 5)

    elif days > 0:
        while start_date.weekday() in weekends:
            start_date -= timedelta(days=1)
        if start_date.weekday() + mod_days > 4:
            end_date = start_date + timedelta(days=mod_days + 2)
        else:
            end_date = start_date + timedelta(days=mod_days)
        end_date = end_date + timedelta(days=7 * (abs(days) - mod_days) / 5)

    else:
        end_date = start_date

    return end_date


class CryptoParameters:

    def __init__(self, crypto_shocks: list):
        self.crypto_shocks = crypto_shocks


class VolSurfaceParameters:
    '''
        Manage config files for options models
    '''

    def __init__(self, parameters: dict):
        self.parameters = parameters

    def _search(self, scenario, class_) -> dict:
        return [item for item in self.parameters[f'{scenario}'] if item["Class"] == class_][0]

    def config_bid_ask(self, class_):
        return self._search('BidAsk', class_)

    def config_parallel(self, class_):
        c1 = self._search('Parallel-Shocks', class_)
        c2 = self._search('Vega-Threshold', class_)
        return {**c1, **c2}

    def config_term_structure(self, class_):
        c1 = self._search('Term-Structure-Shocks', class_)
        c2 = self._search('Vega-Threshold', class_)
        return {**c1, **c2}

    def config_skew(self, class_):
        c1 = self._search('Skew-Shocks', class_)
        c2 = self._search('Vega-Threshold', class_)
        return {**c1, **c2}


class EquityParameters:

    def __init__(
        self, equity_shocks: list, macro_scenarios: list, sector_scenarios: list, net_pairs: dict, rv_scenarios: dict,
        dict_country_core: dict, dict_fx_scenarios: dict, dict_country_grouping: dict, concentration_scenarios: dict
    ):
        self.equity_shocks = equity_shocks
        self.macro_scenarios = macro_scenarios
        self.net_pairs = net_pairs
        self.rv_scenarios = rv_scenarios
        self.sector_scenarios = sector_scenarios
        self.dict_country_core = dict_country_core
        self.dict_fx_scenarios = dict_fx_scenarios
        self.dict_country_grouping = dict_country_grouping
        self.concentration_scenarios = concentration_scenarios


class OptParameters:
    '''
        Manage config files for options models
    '''

    def __init__(self, parameters: dict):
        self.parameters = parameters

    def _search(self, scenario, region, class_) -> dict:
        return [item for item in self.parameters[f'{region}-{scenario}'] if item["Class"] == class_][0]

    def config_ir(self, product):
        return [item for item in self.parameters['IR'] if item['Product'] == product][0]

    def config_bid_ask(self, region, class_):
        return self._search('BidAsk', region, class_)

    def config_parallel(self, region, class_):
        c1 = self._search('Parallel-Shocks', region, class_)
        c2 = self._search('Vega-Threshold', region, class_)
        return {**c1, **c2}

    def config_term_structure(self, region, class_):
        c1 = self._search('Term-Structure-Shocks', region, class_)
        c2 = self._search('Vega-Threshold', region, class_)
        return {**c1, **c2}

    def config_skew(self, region, class_):
        c1 = self._search('Skew-Shocks', region, class_)
        c2 = self._search('Vega-Threshold', region, class_)
        return {**c1, **c2}
