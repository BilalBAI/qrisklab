import numpy as np
import pandas as pd
from pandas import DataFrame
from .black_scholes import bsm_pricing, BS_PARAMETERS

COLUMNS = ['spot_shock', 'vol_shock', 'vol_shock_mode', 'quantity', 'multiplier'] + BS_PARAMETERS
COLUMNS2 = ['post_shock_spot', 'post_shock_vol', 'quantity', 'multiplier'] + BS_PARAMETERS


GLOBAL_SHOCK_EXCLUDE_G4_GAINS = False


class StressTest2:
    '''
        Basic Spot and Vol Stress Test
        sticky strike: if spot changes, the implied volatility of an option with a given absolute strike does not change.
        sticky moneyness/sticky delta: if spot changes, the implied volatility of an option with a given moneyness does not change

        time_to_expiry_offset: adjust the timeframe of the shock 
                            Default time_to_expiry_offset = 0 i.e. instantaneous shock
                            e.g. set time_to_expiry_offset = -1/365 indicates the shock happens in a day
                                    this allows to capture the theta impact
    '''

    def __init__(self, time_to_expiry_offset: float = 0):
        self.time_to_expiry_offset = time_to_expiry_offset

    def shock_df(self, data: DataFrame, name):
        if not set(COLUMNS2).issubset(set(data.columns)):
            raise Exception(f'Input data should include columns: {COLUMNS2}')
        data[name] = np.vectorize(self.shock)(**{col: data[col] for col in COLUMNS2})
        return data

    def shock(self,
              post_shock_spot, post_shock_vol, spot, vol, quantity, multiplier, strike, put_call, time_to_expiry, rate,
              cost_of_carry_rate, **kwargs
    ):
        # Pre shock valuation
        pre_shock = bsm_pricing(
            strike=strike,
            time_to_expiry=time_to_expiry,
            spot=spot,
            rate=rate,
            vol=vol,
            put_call=put_call,
            cost_of_carry_rate=cost_of_carry_rate
        )

        post_shock_time_to_expiry = time_to_expiry + self.time_to_expiry_offset

        # Post shock valuation
        post_shock = bsm_pricing(
            strike=strike,
            time_to_expiry=post_shock_time_to_expiry,
            spot=post_shock_spot,
            rate=rate,
            vol=post_shock_vol,
            put_call=put_call,
            cost_of_carry_rate=cost_of_carry_rate
        )
        return (post_shock - pre_shock) * quantity * multiplier


class StressTest:
    '''
        Basic Spot and Vol Stress Test
        This class only supports the "sticky strike" rule.
    '''

    def shock_df(self, data: DataFrame, name):
        if not set(COLUMNS).issubset(set(data.columns)):
            raise Exception(f'Input data should include columns: {COLUMNS}')
        data[name] = np.vectorize(self.shock)(**{col: data[col] for col in COLUMNS})
        return data

    @staticmethod
    def shock(
        spot_shock, vol_shock, spot, quantity, multiplier, vol, strike, put_call, time_to_expiry, rate,
        cost_of_carry_rate, vol_shock_mode: str = 'relative', **kwargs
    ):
        '''
        vol_shock_mode:
            relative: vol = vol * (1 + vol_shock); 
            absolute: vol = vol + vol_shock;
        '''
        # Pre shock valuation
        pre_shock = bsm_pricing(
            strike=strike,
            time_to_expiry=time_to_expiry,
            spot=spot,
            rate=rate,
            vol=vol,
            put_call=put_call,
            cost_of_carry_rate=cost_of_carry_rate
        )
        # Calculate post shock spot and vol
        if vol_shock_mode == 'absolute':
            post_shock_vol = vol + vol_shock
        else:
            post_shock_vol = vol * (1 + vol_shock)
        post_shock_spot = spot * (1 + spot_shock)
        # Post shock valuation
        post_shock = bsm_pricing(
            strike=strike,
            time_to_expiry=time_to_expiry,
            spot=post_shock_spot,
            rate=rate,
            vol=post_shock_vol,
            put_call=put_call,
            cost_of_carry_rate=cost_of_carry_rate
        )
        return (post_shock - pre_shock) * quantity * multiplier


class StressTree:
    '''
    Stress Tree Model for Multilevel Stress Tests
    '''

    def __init__(self, name: str, stress: dict):
        self.name = name
        self.name2 = f'{name} Loss'
        self.stress = stress
        self.up = None
        self.done = None


class MultilevelST:
    '''
    Multilevel Stress Tests based on a stress tree
    Can only apply single relative vol shock at the moment.  Will add absolute vol shock.
    data should include columns ['spot', 'quantity', 'multiplier', 'vol', 'strike', 'put_call', 'time_to_expiry',
                                    'rate', 'cost_of_carry_rate','group']
    scenario defination example    {
            "Global Shock": {
                        "Group 1": -0.15,
                        "Group 2": -0.15,
                        "Group 3": -0.15,
                        "Group 4": -0.15
                    },
            "Disp1": {
                        "Group 1": 0.025,
                        "Group 2": 0.025,
                        "Group 3": 0.075,
                        "Group 4": 0.075
                    },
            "Disp2": {
                        "Group 1": 0.075,
                        "Group 2": 0.075,
                        "Group 3": 0.05,
                        "Group 4": 0.05
                    },
            "Vol Shock": {
                "Relative": 0,
                "Absolute": 0
            }
        }
    '''

    def __init__(self, data, scenario: dict):
        self.scenario = scenario
        self.data = data
        self.root = None

    def map_shocks(self, node: StressTree):
        self.data['spot_shock'] = self.data['group'].map(node.stress).fillna(0)
        self.data[node.name] = self.data.apply(lambda row: row[f"spot {row['spot_shock']} vol 0"], axis=1).values

    def run(self):
        global_shock = self.scenario['Global Shock']
        disp1 = self.scenario['Disp1']
        disp2 = self.scenario['Disp2']
        self.data['vol_shock'] = self.scenario['Vol Shock']['Relative']
        root = StressTree('Global Shock', global_shock)
        root.up = StressTree(
            'Disp1 Up', {key: (value * 100 + disp1[key] * 100) / 100 for key, value in global_shock.items()}
        )
        root.down = StressTree(
            'Disp1 Down', {key: (value * 100 - disp1[key] * 100) / 100 for key, value in global_shock.items()}
        )
        root.up.up = StressTree(
            'Disp1 Up Disp2 Up',
            {key: (value * 100 + disp1[key] * 100 + disp2[key] * 100) / 100 for key, value in global_shock.items()}
        )
        root.up.down = StressTree(
            'Disp1 Up Disp2 Down',
            {key: (value * 100 + disp1[key] * 100 - disp2[key] * 100) / 100 for key, value in global_shock.items()}
        )
        root.down.up = StressTree(
            'Disp1 Down Disp2 Up',
            {key: (value * 100 - disp1[key] * 100 + disp2[key] * 100) / 100 for key, value in global_shock.items()}
        )
        root.down.down = StressTree(
            'Disp1 Down Disp2 Down',
            {key: (value * 100 - disp1[key] * 100 - disp2[key] * 100) / 100 for key, value in global_shock.items()}
        )
        # Apply shocks
        self.map_shocks(root)
        self.map_shocks(root.up)
        self.map_shocks(root.down)
        self.map_shocks(root.up.up)
        self.map_shocks(root.up.down)
        self.map_shocks(root.down.up)
        self.map_shocks(root.down.down)
        # Calc relative losses from the shocks
        data = self.data
        data[root.up.name2] = data[root.up.name] - data[root.name]
        data[root.down.name2] = data[root.down.name] - data[root.name]
        data[root.up.up.name2] = data[root.up.up.name] - data[root.up.name]
        data[root.up.down.name2] = data[root.up.down.name] - data[root.up.name]
        data[root.down.up.name2] = data[root.down.up.name] - data[root.down.name]
        data[root.down.down.name2] = data[root.down.down.name] - data[root.down.name]
        self.data = data
        self.root = root


class EQMultilevelST(MultilevelST):
    '''
        Multilevel Stress Tests for equity macro and sector scenarios
        data should include columns ['spot', 'quantity', 'multiplier', 'vol', 'strike', 'put_call',
                                        'time_to_expiry', 'rate', 'cost_of_carry_rate','group',
                                        'ISSUER_BULK', 'COUNTRY_ISO', 'GICS_INDUSTRY_GROUP_NAME', 'MKT_CAP_USD_M']
        Available scenario types are ['macro', 'sector']
        Scenario defination example: MultilevelST example + {"Scenario Name": "-15%"}
    '''

    def __init__(self, data, scenario, scenario_type: str):
        if scenario_type not in ['macro', 'sector']:
            raise Exception(f"Unknow type {scenario_type}: Available types are ['macro', 'sector']")
        self.scenario_name = scenario['Scenario Name']
        self.scenario_type = scenario_type
        super().__init__(data=data, scenario=scenario)
        super().run()   # get self.data and self.root

    def run(self):
        data = self.data
        root = self.root
        # Group by certain fields and calc total disp1 loss
        if self.scenario_type == 'macro':
            groupby_fld = 'COUNTRY_ISO'
        elif self.scenario_type == 'sector':
            groupby_fld = 'GICS_INDUSTRY_GROUP_NAME'
        # Aggregate Global Loss
        data['Global Loss'] = data[root.name]

        if GLOBAL_SHOCK_EXCLUDE_G4_GAINS:
            # For country groups 1-3, sum all losses and gains for each asset at the global shock level.
            # For country group 4, sum only losses. Sum the Country 1-3 and Country 4 charges at the global level.
            data.loc[(data['group'] == 'Group 4') & (data['Global Loss'] > 0), 'Global Loss'] = 0

        # Aggregate Disp1 Loss and Disp2 Loss
        data['Disp2 Loss Disp1 Up'] = data.apply(
            lambda row: min(row[root.up.up.name2], row[root.up.down.name2]), axis=1
        )
        data['Disp2 Loss Disp1 Down'] = data.apply(
            lambda row: min(row[root.down.up.name2], row[root.down.down.name2]), axis=1
        )
        # Asset dispersion does not apply to companies with market cap greater than 150bn.
        data.loc[data['MKT_CAP_USD_M'] > 150000, 'Disp2 Loss Disp1 Up'] = 0
        data.loc[data['MKT_CAP_USD_M'] > 150000, 'Disp2 Loss Disp1 Down'] = 0
        # Choose the worst loss tree branch
        tt = data[[groupby_fld, root.up.name2, root.down.name2, 'Disp2 Loss Disp1 Up', 'Disp2 Loss Disp1 Down']]
        df_grouped = tt.groupby(by=[groupby_fld], as_index=False).sum()
        df_grouped['Disp1 Direction'] = df_grouped.apply(
            lambda row: 'Up' if row[root.up.name2] + row['Disp2 Loss Disp1 Up'] < row[root.down.name2] + row[
                'Disp2 Loss Disp1 Down'] else 'Down',
            axis=1
        )
        df_grouped['Disp1 Loss'] = df_grouped.apply(
            lambda row: row[root.up.name2] if row['Disp1 Direction'] == 'Up' else row[root.down.name2], axis=1
        )
        data = pd.merge(data, df_grouped[[groupby_fld, 'Disp1 Direction']], on=groupby_fld, how='left')
        data['Disp1 Loss'] = data.apply(
            lambda row: row[root.up.name2] if row['Disp1 Direction'] == 'Up' else row[root.down.name2], axis=1
        )
        data['Disp2 Loss'] = data.apply(
            lambda row: row['Disp2 Loss Disp1 Up'] if row['Disp1 Direction'] == 'Up' else row['Disp2 Loss Disp1 Down'],
            axis=1
        )
        # Total Loss
        self.global_loss = data['Global Loss'].sum()
        self.disp1_loss = -np.sqrt((df_grouped['Disp1 Loss']**2).sum())
        self.disp2_loss = -np.sqrt((data['Disp2 Loss']**2).sum())
        self.total_loss = self.global_loss + self.disp1_loss + self.disp2_loss
        self.data = data
        self.df_grouped = df_grouped
