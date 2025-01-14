import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", map_name="8x8", render_mode="rgb_array", is_slippery=True) 

# creation of the q-table
qtable = np.zeros((env.observation_space.n, env.action_space.n))

# result list
outcomes = []

episodes = 10000
epsilon = 1.0
epsilon_decay = 0.0001
learning_rate = 0.4
discount_factor = 0.9

for i in range(episodes):
    state, info = env.reset()
    terminated, truncated = False, False
    success = False
    while not terminated and not truncated:
        env.render()

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(qtable[state])

        new_state, reward, terminated, truncated, _ = env.step(action)

        qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_factor * np.max(qtable[new_state]) - qtable[state, action])
        state = new_state

        if reward > 0:
            success = True

    outcomes.append(1 if success else 0)
    epsilon = max(epsilon - epsilon_decay, 0)

print(outcomes)



        
