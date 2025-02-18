import numpy as np
import gymnasium as gym

env = gym.make("CliffWalking-v0", render_mode="human")

q_table = np.load("cliff_walking/q_table/cliffwalking_q_table.npy")

episodes = 1000
epsilon = 1
epsilon_decay = 0.001
learning_rate = 0.5
discount_factor = 0.9
scores = []

for i in range(episodes):
    state, info = env.reset()
    terminated, truncated = False, False
    score = 0
    while not terminated and not truncated:
        action = np.argmax(q_table[state])

        state_, reward, terminated, truncated, _ = env.step(action)

        action_ = np.argmax(q_table[state_])
        q_table[state, action] += learning_rate * (reward + discount_factor * q_table[state_, action_] - q_table[state, action]) 
        state = state_
        score+=reward
    scores.append(score)
    epsilon = max(epsilon - epsilon_decay, 0)

print(scores)
