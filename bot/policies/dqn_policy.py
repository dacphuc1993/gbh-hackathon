import time
import argparse
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import random
import pandas as pd
from datetime import datetime
import numpy as np
import json
from policies.policy import Policy

def build_MLP(
        n_obs, 
        n_action, 
        n_hidden_layer=10, 
        n_neuron_per_layer=128,
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


class DQNPolicy(Policy):
    def __init__(self, state_size, action_size, weight_path="epoch_3-dqn.weight.h5"):
        """
        Constructor for the Policy class. It can take flexible parameters.
        Contestants are free to maintain internal state and use any auxiliary data or information
        within this class.
        """
        super().__init__()
        self.state_size = state_size
        self.solar_action_size = action_size
        self.model = build_MLP(n_obs=state_size, n_action=action_size)
        self.model.load_weights(weight_path)

    def act(self, external_state, internal_state):
        """
        Method to be called when the policy needs to make a decision.

        :param external_state: A dictionary containing the current market data.
        :param internal_state: A dictionary containing the internal state of the policy.
        :return: A tuple (amount to route from solar panel to battery: int, amount to charge battery from grid: int). Note: any solar power not routed to the battery goes directly to the grid and you earn the current spot price.
        """
        if 'timestamp' in external_state.index:
            # external_state = external_state.drop(columns=["timestamp"])
            external_state = external_state[1:]
        state = np.reshape(external_state.tolist(), (1, self.state_size))
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

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)
