# src/recommender/train.py

import torch
from collections import deque
import numpy as np

# Import our custom classes
from src.recommender.environment import SongRecommenderEnvironment
from src.recommender.agent import Agent


def train(n_episodes=2000, max_t=100):
    """
    Main training loop for the DQN agent.

    Args:
        n_episodes (int): Maximum number of training episodes.
        max_t (int): Maximum number of timesteps per episode.
    """
    # 1. Initialize Environment and Agent
    print("--- Starting Training ---")
    env = SongRecommenderEnvironment()
    agent = Agent(state_size=env.state_space_size, action_size=env.action_space_size)

    print(f"Agent and Environment initialized. Training for {n_episodes} episodes.")

    # 2. Training Loop
    scores = []  # List containing scores from each episode
    scores_window = deque(maxlen=100)  # Last 100 scores for averaging

    for i_episode in range(1, n_episodes + 1):
        # Reset environment and get initial state
        state = env.reset()
        score = 0

        for t in range(max_t):
            # Agent chooses an action
            action = agent.act(state)

            # Environment performs the action
            next_state, reward, done = env.step(action)

            # Agent learns from the experience
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += reward

            if done:
                break

        # Save score and update scores window
        scores_window.append(score)
        scores.append(score)

        # Decay epsilon after each episode
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Print progress
        print(
            f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}",
            end="",
        )
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}")

        # Optional: Add a condition to stop if the agent solves the environment
        if (
            np.mean(scores_window) >= 15.0
        ):  # A high score target, e.g., 15 cumulative reward
            print(
                f"\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}"
            )
            break

    # 3. Save the trained model weights
    torch.save(agent.q_network.state_dict(), "q_network_checkpoint.pth")
    print(f"\nTraining finished. Model saved to q_network_checkpoint.pth")
    return scores
