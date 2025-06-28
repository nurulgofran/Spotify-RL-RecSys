import torch
from collections import deque
import numpy as np

from src.recommender.environment import SongRecommenderEnvironment
from src.recommender.agent import Agent


def train(n_episodes=2000, max_t=100):
    print("Training started...")
    env = SongRecommenderEnvironment()
    agent = Agent(state_size=env.state_space_size, action_size=env.action_space_size)

    scores = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0

        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        scores_window.append(score)
        scores.append(score)

        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        if np.mean(scores_window) >= 15.0:
            print(
                f"\nSolved in {i_episode - 100:d} episodes! Average Score: {np.mean(scores_window):.2f}"
            )
            break

    torch.save(agent.q_network.state_dict(), "q_network_checkpoint.pth")
    print("Training finished. Model saved.")
    return scores
