import torch
from collections import deque
import numpy as np

from src.recommender.environment import SongRecommenderEnvironment
from src.recommender.agent import Agent


def train(n_episodes=2000, max_t=100):
    print("--- Starting Training ---")
    env = SongRecommenderEnvironment()
    agent = Agent(state_size=env.state_space_size, action_size=env.action_space_size)

    print(f"Agent and Environment initialized. Training for {n_episodes} episodes.")

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

        print(
            f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        if np.mean(scores_window) >= 15.0:
            print(
                f"\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
            )
            break

    torch.save(agent.q_network.state_dict(), "q_network_checkpoint.pth")
    print("\nTraining finished. Model saved to q_network_checkpoint.pth")
    return scores
