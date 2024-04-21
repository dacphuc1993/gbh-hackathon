import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
from collections import deque
import random
import time
import keras
from environment import BatteryEnv
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Set the seed for reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

# Environment parameters
BATTERY_CAPACITY = 13  # kWh
MAX_CHARGE_RATE = 5  # kW
INITIAL_CHARGE = 7.5  # kWh
REPLAY_BUFFER_SIZE = 100000
BATCH_SIZE = 512
GAMMA = 0.99
LEARNING_RATE = 0.001
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
UPDATE_TARGET_EVERY = 1  # Update target network every 5 episodes


# Helper functions
def create_network(
            state_size, 
            action_size,
            n_hidden_layer=5, 
            n_neuron_per_layer=64,
            activation='relu', 
            loss='mse'
        ):
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_neuron_per_layer, input_shape=(state_size,), activation=activation))

    for i in range(n_hidden_layer):
        model.add(keras.layers.Dense(n_neuron_per_layer, activation=activation))

    model.add(keras.layers.Dense(action_size, activation='linear'))
    model.compile(loss=loss, optimizer=keras.optimizers.Adam())
    # model = Sequential()
    # model.add(Dense(64, input_dim=state_size, activation='relu'))
    # model.add(Dense(64, activation='relu'))
    # model.add(Dense(action_size, activation='linear'))
    # model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
    return model


def update_target_network(main_network, target_network):
    target_network.set_weights(main_network.get_weights())


def encode_actions(solar_kW_to_batteries, charge_kWs, max_charge_rate):
    solar_kW_to_batteries = np.clip(solar_kW_to_batteries, -max_charge_rate, max_charge_rate)
    charge_kWs = np.clip(charge_kWs, -max_charge_rate, max_charge_rate)
    return np.column_stack((solar_kW_to_batteries, charge_kWs))

def decode_action(action, max_charge_rate):
    solar_kW_to_battery = np.clip(action[0], -max_charge_rate, max_charge_rate)
    charge_kW = np.clip(action[1], -max_charge_rate, max_charge_rate)
    return solar_kW_to_battery, charge_kW
# def encode_actions(solar_kW_to_batteries, charge_kWs, max_charge_rate):
#     num_actions = 2 * int(2 * max_charge_rate + 1)
#     solar_kW_to_batteries = (solar_kW_to_batteries + max_charge_rate) / (2 * max_charge_rate)
#     charge_kWs = (charge_kWs + max_charge_rate) / (2 * max_charge_rate)
#     actions = solar_kW_to_batteries * (num_actions // 2) + charge_kWs
#     return actions.astype(int)


# def decode_action(action, max_charge_rate):
#     num_actions = 2 * int(2 * max_charge_rate + 1)
#     solar_kW_to_battery = (action // (num_actions // 2)) * (2 * max_charge_rate) - max_charge_rate
#     charge_kW = (action % (num_actions // 2)) * (2 * max_charge_rate) - max_charge_rate
#     return solar_kW_to_battery, charge_kW


data_path = 'data/imputed_training_v3.csv'
external_states = pd.read_csv(data_path)
state_size = len(external_states.columns)
action_size = 2  # (solar_kW_to_battery, charge_kW)
replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
main_network = create_network(state_size, action_size)
target_network = create_network(state_size, action_size)
update_target_network(main_network, target_network)
initial_profit = 0
initial_soc = 7.5
start_step = 0
epsilon = 0.5
num_episodes = 50000
checkpoint_interval = 10000
future_data = external_states.iloc[start_step:]

# Training loop
for episode in range(num_episodes):
    battery_environment = BatteryEnv(
        data=future_data,
        initial_charge_kWh=initial_soc,
        initial_profit=initial_profit
    )
    profits, socs, market_prices, battery_actions, solar_actions, pv_inputs, timestamps = [], [], [], [], [], [], []
    external_state, internal_state = battery_environment.initial_state()
    episode_reward = 0
    while True:
        pv_power = float(external_state["pv_power"])
        # Choose action using epsilon-greedy policy
        if np.random.rand() <= epsilon:
            solar_kW_to_battery = np.random.uniform(-MAX_CHARGE_RATE, MAX_CHARGE_RATE)
            charge_kW = np.random.uniform(-MAX_CHARGE_RATE, MAX_CHARGE_RATE)
        else:
            state_tensor = np.expand_dims(external_state, axis=0)
            # q_values = main_network.predict(state_tensor, verbose=0)
            solar_kW_to_battery, charge_kW = main_network.predict(state_tensor, verbose=0)[0]
            # action = np.argmax(q_values)
            # solar_kW_to_battery, charge_kW = decode_action(q_values[0], MAX_CHARGE_RATE)

        # Take action and observe next state and reward
        next_state, next_info = battery_environment.step(charge_kW, solar_kW_to_battery, pv_power)

        if (next_state is None) or (next_info['remaining_steps'] == 0):
            break

        reward = next_info['profit_delta']
        episode_reward += reward

        # Store experience in replay buffer
        replay_buffer.append((external_state, solar_kW_to_battery, charge_kW, reward, next_state))

        if len(replay_buffer) >= BATCH_SIZE:
            # Sample random batch from replay buffer
            batch = random.sample(replay_buffer, BATCH_SIZE)
            states, solar_kW_to_batteries, charge_kWs, rewards, next_states = map(np.array, zip(*batch))

            # Calculate target Q-values
            next_q_values = target_network.predict(next_states, verbose=0)
            max_next_q_values = np.max(next_q_values, axis=1)
            targets = np.expand_dims(rewards + GAMMA * max_next_q_values, axis=1)

            # Update main network
            q_values = main_network.predict(states, verbose=0)
            # encoded_actions = encode_actions(solar_kW_to_batteries, charge_kWs, MAX_CHARGE_RATE)
            # encoded_actions = encoded_actions.astype(int)
            # row_indices = np.arange(BATCH_SIZE).reshape(-1, 1)
            q_values[np.arange(BATCH_SIZE), :] = targets

            main_network.fit(states, q_values, epochs=1, verbose=0)

        # Update target network
        if episode % UPDATE_TARGET_EVERY == 0:
            update_target_network(main_network, target_network)

        # Update epsilon for exploration
        epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)

        external_state = next_state

        print(f"Episode: {episode} -- Remaining steps: {next_info['remaining_steps']} -- Episode reward: {episode_reward} -- Epsilon: {epsilon}")

    try:
        if (episode + 1) % checkpoint_interval == 0:
            main_network.save_weights(f"weights/main_network_episode_{episode+1}.weights.h5")
        # target_network.save_weights(f"target_network_weights_episode_{episode+1}.weights.h5")
    except:
        pass

# Save final weights after training
main_network.save_weights("weights/main_network_final.weights.h5")
target_network.save_weights("weights/target_network_final.weights.h5")

