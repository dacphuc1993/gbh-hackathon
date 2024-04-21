import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import pandas as pd
from collections import deque
import random
import time
import keras
import json
from policies.policy import Policy


def create_network(
            state_size, 
            action_size,
            n_hidden_layer=5, 
            n_neuron_per_layer=128,
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


class DDQNPolicy(Policy):
    def __init__(self, state_size, action_size, weight_path="epoch_3-dqn.weight.h5"):
        super().__init__()
        self.state_size = state_size
        self.solar_action_size = action_size
        self.model = create_network(state_size=state_size, action_size=action_size)
        self.model.load_weights(weight_path)

    def act(self, external_state, internal_state):
        """
        Method to be called when the policy needs to make a decision.

        :param external_state: A dictionary containing the current market data.
        :param internal_state: A dictionary containing the internal state of the policy.
        :return: A tuple (amount to route from solar panel to battery: int, amount to charge battery from grid: int). Note: any solar power not routed to the battery goes directly to the grid and you earn the current spot price.
        """
        # columns_to_drop = ["timestamp",
        #             # "pv_power_forecast_1h",
        #             # "pv_power_forecast_2h",
        #             # "pv_power_forecast_24h",
        #             # "pv_power_basic",
        #             # "pv_power_basic_forecast_1h",
        #             # "pv_power_basic_forecast_2h",
        #             # "pv_power_basic_forecast_24h"
        #             ]

        # filtered_state = external_state.copy()

        # # Iterate over the list of columns to drop
        # for column in columns_to_drop:
        #     if column in filtered_state.index:
        #         filtered_state = filtered_state.drop(column)

        if 'timestamp' in external_state.index:
            # external_state = external_state.drop(columns=["timestamp"])
            external_state = external_state[1:]
        # print(external_state)

        state_tensor = np.expand_dims(external_state.tolist(), axis=0)

        # Predict the actions using the main network
        actions = self.model.predict(state_tensor, verbose=0)[0]
        solar_kW_to_battery, charge_kW = actions

        return solar_kW_to_battery, charge_kW
    
    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)

