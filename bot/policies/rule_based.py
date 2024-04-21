import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

MONTHLY_PRICE_DICT = {
    1: {     # Jan
        "high": [2, 7, 8, 9, 17],
        "low": [1, 3, 4, 5, 15, 16, 23, 0]
    },
    2: {     # Feb
        "high": [18, 19, 6],
        "low": [2, 3, 9, 10, 11, 12, 13, 14]
    },
    3: {     # Mar
        "high": [9, 10, 13, 20],
        "low": [11, 1, 2, 3, 15, 16, 17]
    },
    4: {    # Apr
        "high": [7, 8, 13, 20, 21],
        "low": [17, 18, 1, 2, 3, 11, 12]
    },
    5: {    # May
        "high": [21, 22, 7, 8],
        "low": [1, 2, 3, 16, 17, 18]
    },
    6: {    # June
        "high": [7, 8, 21, 22],
        "low": [18, 19, 1, 2, 3]
    },
    7: {    # July
        "high": [7, 8, 21, 22],
        "low": [17, 18, 2, 3, 4]
    },
    8: {    # Aug
        "high": [7, 8, 9, 21, 22],
        "low": [16, 17, 18, 1, 2, 3]
    },
    9: {    # Sep
        "high": [8, 9, 13, 14, 20],
        "low": [2, 3, 4, 11, 12, 17, 18]
    },
    10: {   # Oct
        "high": [8, 9, 19, 20],
        "low": [16, 17, 2, 3, 4]
    },
    11: {   # Nov
        "high": [8, 9, 10, 17, 19, 14],
        "low": [3, 4, 5, 18, 16, 12]
    },
    12: {   # Dec
        "high": [9, 14],
        "low": [22, 23, 0, 11, 12]
    }
}


class RuleBasedPolicy(Policy):
    def __init__(self, window_size=7):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.price_dict = MONTHLY_PRICE_DICT
        self.window_size = window_size
        self.price_history = deque(maxlen=window_size)

    def act(self, external_state, internal_state):
        # timestamp = pd.to_datetime(external_state['timestamp'], format="%Y-%m-%d %H:%M:%S")
        timestamp = pd.to_datetime(external_state['timestamp'], utc=True)
        month = timestamp.month
        hour = timestamp.hour
        market_price = external_state['price']
        self.price_history.append(market_price)
        moving_average = np.mean(self.price_history)

        if external_state["price"] < 0:
            charge_kW = internal_state['max_charge_rate']
            solar_kW_to_battery = external_state["pv_power"]
            return solar_kW_to_battery, charge_kW
        else:
            if hour in self.price_dict[month]["high"]:
                charge_kW = -internal_state['max_charge_rate']
                solar_kW_to_battery = 0
            elif hour in self.price_dict[month]["low"] and float(external_state["pv_power"]) > 0:
                solar_kW_to_battery = external_state["pv_power"]
                charge_kW = internal_state['max_charge_rate'] - external_state["pv_power"]
            elif hour in self.price_dict[month]["low"] and float(external_state["pv_power"]) == 0:
                solar_kW_to_battery = 0
                charge_kW = internal_state['max_charge_rate']
            # elif float(external_state["pv_power"]) > 0 and internal_state["battery_soc"] < 6.5:
            #     solar_kW_to_battery = float(external_state["pv_power"]) / 2
            #     charge_kW = float(external_state["pv_power"]) / 2
            # elif float(external_state["pv_power"]) > 0 and internal_state["battery_soc"] >= 6.5:
            #     solar_kW_to_battery = 0
            #     charge_kW = -internal_state['max_charge_rate']
            else:
                if market_price > moving_average:
                    solar_kW_to_battery = 0
                    charge_kW = -internal_state['max_charge_rate']
                else:
                    solar_kW_to_battery = 0
                    charge_kW = internal_state['max_charge_rate']
                # solar_kW_to_battery = float(external_state["pv_power"]) / 2
                # charge_kW = float(external_state["pv_power"]) / 2

        return solar_kW_to_battery, charge_kW

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)