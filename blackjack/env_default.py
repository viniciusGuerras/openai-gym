import gymnasium as gym
import numpy as np  
import matplotlib.pyplot as plt

env = gym.make("Blackjack-v1", render_mode="rgb_array")

player_sum_range  = range(0,22)
dealer_card_range = range(0,11)
ace_range         = [True, False]

q_table = np.zeros((len(player_sum_range), len(dealer_card_range), len(ace_range), env.action_space.n))

episodes = 35000
learning_rate = 0.85
discount_factor = 0.9

epsilon = 1.0
epsilon_decay = 0.00005

scores=[]


def get_state(state):
    player_sum = state[0]
    dealer_card = state[1] - 1
    ace = 1 if state[2] else 0

    return (player_sum, dealer_card, ace)

for i in range(episodes):
    state, info = env.reset()
    state = get_state(state)
    terminated, truncated = False, False
    score=0

    while not terminated and not truncated:

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state[0], state[1], state[2]])

        state_, reward, terminated, truncated, _ = env.step(action)
        state_ = get_state(state_)

        if state_[0] > 21:
            terminated = True
            break

        action_ = np.argmax(q_table[state_[0], state_[1], int(state_[2])])

        q_table[state[0], state[1], int(state[2]), action] += \
        learning_rate * (reward + discount_factor * q_table[state_[0], state_[1], int(state_[2]), action_] - \
        q_table[state[0], state[1], int(state[2]), action])

        score+=reward
        state = state_
    scores.append(score)
    epsilon = max(epsilon - epsilon_decay, 0) 

print(scores)