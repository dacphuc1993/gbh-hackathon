import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
os.environ["KERAS_BACKEND"] = "torch"
import pandas as pd
from battery_gym import BatteryEnv
from datetime import datetime
from stable_baselines3 import SAC
from stable_baselines3.common.env_checker import check_env

data_path = 'data/pruned_training.csv'
external_states = pd.read_csv(data_path)
start_step = 0
future_data = external_states.iloc[start_step:]
max_charge_kW = 5  # Maximum charge/discharge rate
capacity_kWh = 13  # Battery capacity
initial_soc = 7.5  # Initial battery charge
initial_profit = 0

env = BatteryEnv(
        data=future_data,
        initial_charge_kWh=initial_soc,
        initial_profit=initial_profit
    )


# check_env(env)

policy_kwargs = {"net_arch":[512, 512, 512]}

# policy = stable_baselines3.sac.policies.SACPolicy(env.observation_space,
#                     env.action_space,
#                     lr_schedule=
#                     net_arch=[512, 512, 512])

model = SAC("MlpPolicy", env, verbose=2, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=10000, log_interval=1, progress_bar=True)
model.save("sac_battery")
