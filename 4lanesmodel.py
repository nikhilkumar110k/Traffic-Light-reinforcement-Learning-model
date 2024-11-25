import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import random
from collections import deque

class TrafficLightEnv:
    def __init__(self, data, initial_state=None, max_timesteps=100):
        self.data = data
        self.current_step = 0
        self.max_timesteps = max_timesteps
        self.n_actions = 3  
        self.state = initial_state if initial_state is not None else self.data[0]

    def reset(self):
        self.current_step = 0
        self.state = self.data[self.current_step]
        return np.array(self.state)

    def step(self, action):
        reward = 0
        traffic_volumes = self.state[:4]  
        co2_emissions = self.state[4:]  

        if action == 2:  
            if max(traffic_volumes) > 15:
                reward = 1 - 0.01 * sum(co2_emissions)  
            else:
                reward = -0.1
        elif action == 1:  
            if 10 <= max(traffic_volumes) <= 15:
                reward = 0.5 - 0.01 * sum(co2_emissions)
            else:
                reward = -0.1
        elif action == 0:  
            if max(traffic_volumes) < 10:
                reward = 0.4 - 0.01 * sum(co2_emissions)
            else:
                reward = -0.1

        self.current_step += 1
        done = self.current_step >= len(self.data)
        if not done:
            self.state = self.data[self.current_step]

        return np.array(self.state), reward, done, {}

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        inputs = Input(shape=(self.state_size,))
        x = Dense(24, activation='relu')(inputs)
        x = Dense(24, activation='relu')(x)
        outputs = Dense(self.action_size, activation='linear')(x)  
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        state = np.reshape(state, [1, self.state_size])
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = np.reshape(state, [1, self.state_size])
            next_state = np.reshape(next_state, [1, self.state_size])

            target = reward
            if not done:
                target += self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        self.model.save(filename)

def generate_synthetic_traffic_data(num_samples=1000):
    times_of_day = np.random.uniform(0, 24, num_samples)
    traffic_volumes = np.random.randint(0, 50, (num_samples, 4))  
    co2_emissions = traffic_volumes * np.random.uniform(0.05, 0.15, (num_samples, 4))  
    data = np.hstack((traffic_volumes, co2_emissions))  
    return data

if __name__ == "__main__":
    data = generate_synthetic_traffic_data()

    env = TrafficLightEnv(data)
    state_size = env.reset().shape[0]
    action_size = env.n_actions
    agent = DQNAgent(state_size, action_size)

    batch_size = 32
    episodes = 1000
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(batch_size)
        agent.save("saved_model/4lanestraffic_light_agent.h5")
        print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}")
