import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import os
os.environ["KERAS_BACKEND"] = "torch"
import pandas as pd
from tariff_environment import BatteryEnv
from datetime import datetime

# Environment settings
max_charge_kW = 5  # Maximum charge/discharge rate
capacity_kWh = 13  # Battery capacity
initial_soc = 7.5  # Initial battery charge
initial_profit = 0

# SAC hyperparameters
gamma = 0.99  # Discount factor
alpha = 0.2  # Temperature parameter for entropy regularization
tau = 0.005  # Target network update rate
lr = 3e-4  # Learning rate
batch_size = 64  # Batch size for training
num_episodes = 10000
checkpoint_interval = 100
episode_length = 288
start_step = 0
data_path = 'data/pruned_training.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
external_states = pd.read_csv(data_path)
data_len = len(external_states)
external_states = external_states.drop(columns=[
                    # "timestamp",
                    "pv_power_forecast_1h",
                    "pv_power_forecast_2h",
                    "pv_power_forecast_24h",
                    "pv_power_basic",
                    # "pv_power_basic_forecast_1h",
                    # "pv_power_basic_forecast_2h",
                    # "pv_power_basic_forecast_24h"
                    ]) 

future_data = external_states.iloc[start_step:]

# Create replay buffer
replay_buffer = deque(maxlen=100000)

# Define MLP architecture
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define policy, Q-function, and value function networks
state_dim = 5  # [battery.state_of_charge_kWh, price, demand, pv_power, temperature_air]
action_dim = 2  # [solar_kW_to_battery, charge_kW]
policy_net = MLP(state_dim, action_dim).to(device)
q_net1 = MLP(state_dim + action_dim, 1).to(device)
q_net2 = MLP(state_dim + action_dim, 1).to(device)
value_net = MLP(state_dim, 1).to(device)

# Create target networks
target_policy_net = MLP(state_dim, action_dim).to(device)
target_q_net1 = MLP(state_dim + action_dim, 1).to(device)
target_q_net2 = MLP(state_dim + action_dim, 1).to(device)
target_value_net = MLP(state_dim, 1).to(device)

# Copy weights from main networks to target networks
for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(target_q_net1.parameters(), q_net1.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(target_q_net2.parameters(), q_net2.parameters()):
    target_param.data.copy_(param.data)
for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
    target_param.data.copy_(param.data)

# Define optimizers
policy_optimizer = optim.Adam(policy_net.parameters(), lr=lr)
q_optimizer = optim.Adam(list(q_net1.parameters()) + list(q_net2.parameters()), lr=lr)
value_optimizer = optim.Adam(value_net.parameters(), lr=lr)


last_episode_step = 0
# Training loop
for episode in range(num_episodes):
    if (last_episode_step + episode_length) > data_len:
        last_episode_step = 0
    future_data = external_states.iloc[last_episode_step:]
    battery_environment = BatteryEnv(
        data=future_data,
        initial_charge_kWh=initial_soc,
        initial_profit=initial_profit
    )
    external_state, internal_state = battery_environment.initial_state()
    episode_reward = 0

    for step in range(episode_length):
        last_episode_step += 1
        pv_power = float(external_state["pv_power"])
        battery_soc = internal_state["battery_soc"]
        state = external_state._append(pd.Series([battery_soc]))
        state_numpy = np.array([state[1:].to_numpy()], dtype=np.float32)

        # Select action
        state_tensor = torch.from_numpy(state_numpy).to(device)
        action_mean = policy_net(state_tensor)
        action_dist = torch.distributions.Normal(action_mean, max_charge_kW)
        action = action_dist.sample().clamp(-max_charge_kW, max_charge_kW).cpu().numpy()
        charge_kW, solar_kW_to_battery = action[0]

        # Take action and observe next state and reward
        next_state, next_info = battery_environment.step(charge_kW, solar_kW_to_battery, pv_power)
        reward = next_info['profit_delta']
        done = next_info['remaining_steps'] == 0
        episode_reward += reward

        next_state = next_state[1:]._append(pd.Series(next_info["battery_soc"])).to_numpy(dtype=np.float32)

        # Store transition in replay buffer
        replay_buffer.append((state_numpy, action, reward, next_state, done))

        # Sample batch from replay buffer
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

            # # Convert states, actions, rewards, next_states, and dones to tensors on GPU
            # states_tensor = torch.from_numpy(np.array(states)).float().to(device)
            # actions_tensor = torch.from_numpy(np.array(actions)).float().to(device)
            # rewards_tensor = torch.from_numpy(np.array(rewards)).float().to(device)
            # next_states_tensor = torch.from_numpy(np.array(next_states)).float().to(device)
            # dones_tensor = torch.from_numpy(np.array(dones)).float().to(device)

            # Convert states, actions, rewards, next_states, and dones to tensors on GPU
            states_tensor = torch.from_numpy(states).float().to(device)
            actions_tensor = torch.from_numpy(actions).float().to(device)
            rewards_tensor = torch.from_numpy(rewards).float().to(device)
            next_states_tensor = torch.from_numpy(next_states).float().to(device)
            dones_tensor = torch.from_numpy(dones).float().to(device)

            # Compute target values
            # with torch.no_grad():
            #     next_actions = target_policy_net(torch.from_numpy(next_states).float())
            #     next_q_values1 = target_q_net1(torch.cat([torch.from_numpy(next_states).float(), next_actions], dim=1))
            #     next_q_values2 = target_q_net2(torch.cat([torch.from_numpy(next_states).float(), next_actions], dim=1))
            #     next_q_values = torch.min(next_q_values1, next_q_values2)
            #     next_values = target_value_net(torch.from_numpy(next_states).float())
            #     target_q_values = torch.from_numpy(rewards).float() + gamma * (1 - torch.from_numpy(dones).float()) * (next_values - alpha * next_q_values)
            with torch.no_grad():
                next_actions = target_policy_net(next_states_tensor)
                next_q_values1 = target_q_net1(torch.cat([next_states_tensor, next_actions], dim=1))
                next_q_values2 = target_q_net2(torch.cat([next_states_tensor, next_actions], dim=1))
                next_q_values = torch.min(next_q_values1, next_q_values2)
                next_values = target_value_net(next_states_tensor)
                target_q_values = rewards_tensor + gamma * (1 - dones_tensor) * (next_values - alpha * next_q_values)

            # Update Q-function networks
            # q_values1 = q_net1(torch.cat([torch.from_numpy(states).float(), torch.from_numpy(actions).float()], dim=1))
            # q_values2 = q_net2(torch.cat([torch.from_numpy(states).float(), torch.from_numpy(actions).float()], dim=1))
            q_values1 = q_net1(torch.cat([states_tensor, policy_net(states_tensor)], dim=2))
            q_values2 = q_net2(torch.cat([states_tensor, policy_net(states_tensor)], dim=2))
            q_loss1 = ((q_values1 - target_q_values) ** 2).mean()
            q_loss2 = ((q_values2 - target_q_values) ** 2).mean()

            q_optimizer.zero_grad()
            q_loss1.backward()
            q_loss2.backward()
            q_optimizer.step()

            # Update value function network
            values = value_net(states_tensor)
            q_values1 = q_net1(torch.cat([states_tensor, policy_net(states_tensor)], dim=2))
            q_values2 = q_net2(torch.cat([states_tensor, policy_net(states_tensor)], dim=2))
            q_values = torch.min(q_values1, q_values2)
            value_loss = ((values - q_values) ** 2).mean()

            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()

            # Update policy network
            actions = policy_net(states_tensor)
            q_values1 = q_net1(torch.cat([states_tensor, actions], dim=2))
            q_values2 = q_net2(torch.cat([states_tensor, actions], dim=2))
            q_values = torch.min(q_values1, q_values2)
            policy_loss = (alpha * torch.log(torch.maximum(actions, torch.tensor(1e-6))).sum(dim=1) - q_values).mean()

            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()

            # Update target networks
            for target_param, param in zip(target_policy_net.parameters(), policy_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(target_q_net1.parameters(), q_net1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(target_q_net2.parameters(), q_net2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        state = next_state

    print(f"Episode {episode}: Reward = {episode_reward} --- Data step = {last_episode_step}")

    if (episode + 1) % checkpoint_interval == 0:
        torch.save(policy_net.state_dict(), f'policy_net_{episode+1}.pth')
        # torch.save(q_net1.state_dict(), f'q_net1_{episode+1}.pth')
        # torch.save(q_net2.state_dict(), f'q_net2_{episode+1}.pth')
        # torch.save(value_net.state_dict(), f'value_net_{episode+1}.pth')