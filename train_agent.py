from collections import deque
from agent_definition import Agent
import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

def train(n_episodes=1000):
    scores = []
    scores_window = deque(maxlen=100)
    
    for i_episode in range(n_episodes):
        state, _ = env.reset()
        score = 0
        while True:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.step(state, action, reward, next_state, done, truncated)
            state = next_state
            score += reward
            if done or truncated:
                break
        
        agent.end_episode()  
        scores_window.append(score)
        scores.append(score)
        
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
train()