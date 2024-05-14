import re
import time
import numpy as np
from SnakeGymEnv.SnakeEnv import Snake
from agents.DQN import DQNAgent

env = Snake(render=False)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
print(state_size, action_size)
batch_size = 64
EPISODES = 120
reward_list_training = []
apple_list_training = []
timestep_list_training = []
eps_list = []
max_apple_count = 1
reached = False

# %%
for e in range(EPISODES):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    episode_reward = 0
    timestep = 0
    
    while(not done):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward
        timestep+=1
        if timestep % 100 == 0:
            print(f"{timestep} steps taken in episode {e}")
        if timestep > 4000:
            done = True
        if done:
            if e % 5000 == 0:
                print("Episode: {}/{}, Episode Reward: {}, Epsilon: {:.2}, Episode Apple Count: {}, Timestep: {}"
                      .format(e, EPISODES, episode_reward, agent.epsilon, env.apple_count, timestep))
            reward_list_training.append(episode_reward)
            apple_list_training.append(env.apple_count)
            timestep_list_training.append(timestep)
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
    eps_list.append(agent.epsilon)
    if len(agent.memory) > 100000:
        if not reached: # THIS WILL RUN ONLY ONCE
            train_start_at = e
            reached = True
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon = agent.epsilon*((agent.epsilon_min/1)**(1/(EPISODES-train_start_at)))

env.close()
