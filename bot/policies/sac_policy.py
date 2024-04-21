import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
from collections import deque
import random
import time
import json
from policies.policy import Policy
from stable_baselines3 import SAC


# # Function to get action from the policy network
# def get_action(state):
#     state_tensor = torch.from_numpy(np.array([state])).float()
#     action_mean = policy_net(state_tensor)
#     action_dist = torch.distributions.Normal(action_mean, max_charge_kW)
#     action = action_dist.sample().clamp(-max_charge_kW, max_charge_kW).numpy()[0]
#     return action

max_charge_kW = 5 
pd.set_option('future.no_silent_downcasting', True)

class SACPolicy(Policy):
    def __init__(self, state_size=5, action_size=2, weight_path="sac_battery"):
        super().__init__()
        self.state_size = state_size
        self.solar_action_size = action_size
        self.model = SAC.load(weight_path)
        self.price_history = deque(maxlen=7)

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)

    def act(self, external_state, internal_state):
        columns_to_keep = ["price", "demand", "temp_air", "pv_power"]
        filtered_external_states = external_state.loc[columns_to_keep]
        
        nan_indices = filtered_external_states.isna()
        filtered_external_states[nan_indices] = 0
        battery_soc = internal_state["battery_soc"]
        state = filtered_external_states._append(pd.Series([battery_soc]))
        state_numpy = np.array(state, dtype=np.float32)
        # print(state_numpy)

        action, _states = self.model.predict(state_numpy, deterministic=True)
        action_float64 = action.astype(np.float64)   # np.float32 will give eror when dumping to JSON file
        charge_kW = action_float64[1]
        solar_kW_to_battery = action_float64[0]

        return solar_kW_to_battery, charge_kW
