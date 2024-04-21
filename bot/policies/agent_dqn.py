import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
import collections
import random
# import policy


# class MLP(keras.Model):
#     def __init__(self, 
#                 state_size, 
#                 action_size,
#                 n_hidden_layer=1, 
#                 n_neuron_per_layer=32,
#                 activation='relu', 
#                 loss='mse'):


def build_MLP(
        n_obs, 
        n_action, 
        n_hidden_layer=5, 
        n_neuron_per_layer=32,
        activation='relu', 
        loss='mse'
    ):

  model = keras.models.Sequential()
  model.add(keras.layers.Dense(n_neuron_per_layer, input_shape=(n_obs,), activation=activation))

  for i in range(n_hidden_layer):
    model.add(keras.layers.Dense(n_neuron_per_layer, activation=activation))

  model.add(keras.layers.Dense(n_action + 1, activation='linear'))
  model.compile(loss=loss, optimizer=keras.optimizers.Adam())

  return model


class DQNAgent(object):
  def __init__(self, state_size, action_size):
    self.state_size = state_size
    self.solar_action_size = action_size
    self.memory = collections.deque(maxlen=2000)
    self.gamma = 0.95  # discount rate
    self.epsilon = 1.0  # exploration rate
    self.epsilon_min = 0.01
    self.epsilon_decay = 0.995
    self.model = build_MLP(state_size, action_size)

  def remember(self, state, action, reward, next_state, done):
    self.memory.append((state, action, reward, next_state, done))

  def _random_action(self):
    # Choose random actions for exploration
    return np.random.randint(self.solar_action_size), np.random.randint(self.charge_action_size)

  def act(self, state):
    if np.random.rand() <= self.epsilon:
        # Random action for exploration
        solar_action = np.random.randint(self.solar_action_size)  # Choose a random solar action
        charge_action = np.random.uniform(-5.0, 5.0)  # Choose a random charge rate between -5 kW and 5 kW
    else:
        state = np.reshape(state, (1, self.state_size))
        # Predict the Q-values for the current state
        q_values = self.model.predict(state, verbose=0)[0]
        # Split the Q-values into solar_kW_to_battery (discrete) and charge_kW (continuous) parts
        solar_q_values = q_values[:self.solar_action_size]
        charge_q_value = q_values[self.solar_action_size]
        # Choose the solar action with the highest Q-value
        solar_action = np.argmax(solar_q_values)
        # Choose the charge rate based on the predicted Q-value
        charge_action = charge_q_value

    return solar_action, charge_action
    # if np.random.rand() <= self.epsilon:
    #   return random.randrange(self.action_size)
    # act_values = self.model.predict(state, verbose=0)
    # return np.argmax(act_values[0])  # returns action

  def replay(self, batch_size=32):
    minibatch = random.sample(self.memory, batch_size)
    
    # Extract elements from minibatch
    states = np.array([experience[0] for experience in minibatch])
    actions = np.array([experience[1] for experience in minibatch])
    rewards = np.array([experience[2] for experience in minibatch])
    next_states = np.array([experience[3] for experience in minibatch])
    dones = np.array([experience[4] for experience in minibatch])

    # Compute targets
    targets = rewards + self.gamma * np.amax(self.model.predict(next_states, verbose=0), axis=1) * (1 - dones)

    # Predict Q-values for current states
    q_values = self.model.predict(states, verbose=0)

    # Update Q-values based on the actions taken
    for idx, action in enumerate(actions):
        solar_action, charge_action = action
        q_values[idx][int(solar_action)] = targets[idx]
        q_values[idx][self.solar_action_size] = targets[idx]  # Update charge_kW Q-value
        # charge_action_idx = action[1]
        # solar_action_idx = 0
        # charge_action_idx = 1
        # q_values[idx][solar_action_idx] = targets[idx]
        # q_values[idx][charge_action_idx] = targets[idx]

    # Train the model using the computed targets
    self.model.fit(states, q_values, epochs=1, verbose=0)

    # Decay exploration rate
    if self.epsilon > self.epsilon_min:
        self.epsilon *= self.epsilon_decay

    # """ vectorized implementation; 30x speed up compared with for loop """
    # minibatch = random.sample(self.memory, batch_size)

    # states = np.array([tup[0] for tup in minibatch])
    # actions = np.array([tup[1] for tup in minibatch])
    # rewards = np.array([tup[2] for tup in minibatch])
    # next_states = np.array([tup[3] for tup in minibatch])
    # done = np.array([tup[4] for tup in minibatch])

    # # Q(s', a)
    # target = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1)
    # # end state target is reward itself (no lookahead)
    # target[done] = rewards[done]

    # # Q(s, a)
    # target_f = self.model.predict(states)
    # # make the agent to approximately map the current state to future discounted reward
    # target_f[range(batch_size), actions] = target

    # self.model.fit(states, target_f, epochs=1, verbose=0)

    # if self.epsilon > self.epsilon_min:
    #   self.epsilon *= self.epsilon_decay

  def load(self, name):
    self.model.load_weights(name)

  def save(self, name):
    self.model.save_weights(name)