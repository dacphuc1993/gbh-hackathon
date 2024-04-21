import argparse
import yaml
import os
import time
import random
import pandas as pd
# from policies import policy_classes
from environment import BatteryEnv
from datetime import datetime
import numpy as np
import tqdm
import json
from policies import agent_dqn
os.environ["KERAS_BACKEND"] = "torch"


data_path = 'data/imputed_training_v3.csv'
external_states = pd.read_csv(data_path)
batch_size = 1024
num_trials = 10
timestamp = time.strftime('%Y%m%d%H%M')
state_size = len(external_states.columns)
action_size = 3
initial_profit = 0
initial_soc = 7.5
start_step = 0
future_data = external_states.iloc[start_step:]

agent = agent_dqn.DQNAgent(state_size=state_size, action_size=action_size)

total_profits = []
for trial in tqdm.tqdm(range(num_trials)):
    battery_environment = BatteryEnv(
        data=future_data,
        initial_charge_kWh=initial_soc,
        initial_profit=initial_profit
    )
    profits, socs, market_prices, battery_actions, solar_actions, pv_inputs, timestamps = [], [], [], [], [], [], []
    external_state, internal_state = battery_environment.initial_state()
    # external_state = np.reshape(external_state, [1, state_size])  
    while True:
        # state['battery_soc'] = info['battery_soc']
        pv_power = float(external_state["pv_power"])
        solar_kW_to_battery, charge_kW = agent.act(external_state)
        next_external_state, internal_state = battery_environment.step(charge_kW, solar_kW_to_battery, pv_power)
        if next_external_state is None or internal_state["remaining_steps"] == 0:
            break
        reward = internal_state["profit_delta"]
        done = internal_state["remaining_steps"] == 0
        # next_state['battery_soc'] = info['battery_soc']
        # next_external_state = np.reshape(next_external_state, [1, state_size])
        agent.remember(external_state.tolist(), (solar_kW_to_battery, charge_kW), reward, next_external_state.tolist(), done)
        external_state = next_external_state

        battery_actions.append(charge_kW)
        solar_actions.append(solar_kW_to_battery)
        profits.append(internal_state['total_profit'])
        socs.append(internal_state['battery_soc'])
        # market_prices.append(external_state['price'])

        if (len(agent.memory) > batch_size) and (internal_state["remaining_steps"] % batch_size == 0):
            agent.replay(batch_size)
            rem_step = internal_state["remaining_steps"]
            print(f"Epoch: {trial} -- Remaining steps: {rem_step}")

    try:
        if trial % 3 == 0:
            agent.save(f'weights/epoch_{trial}-dqn.weights.h5')
    except:
        pass

    total_profits.extend(profits)

agent.save(f'weights/{timestamp}-dqn.weights.h5')

