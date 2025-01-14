import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="4x4", render_mode="rgb_array", is_slippery=False)

# creation of the q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))

# result list
outcomes = []

# hyperparameters
episodes = 1000
epsilon = 1.0
epsilon_decay = 0.001
learning_rate = 0.6
discount_factor = 0.9

# loop
for i in range(episodes):
    state, info = env.reset()
    terminated = False
    truncated = False
    episode_success = False

    while not terminated and not truncated:
        env.render()

        # take action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])
        new_state, reward, terminated, truncated, info = env.step(action)

        # update q-table
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + (discount_factor * np.max(qtable[new_state]) - qtable[state, action])) 
        state = new_state

        if reward > 0:
            episode_success = True

    outcomes.append(1 if episode_success else 0)
    epsilon = max(epsilon - epsilon_decay, 0)

print(outcomes)