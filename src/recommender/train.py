"""
Training module for DQN music recommendation.
"""

import numpy as np
from collections import deque
import logging

from src.recommender.environment import SongRecommenderEnvironment
from src.recommender.agent import Agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(n_episodes=1000, max_t=100, print_every=100):
    """Train the DQN agent."""
    logger.info("Starting DQN training...")

    env = SongRecommenderEnvironment()
    agent = Agent(
        state_size=env.state_space_size,
        action_size=env.action_space_size,
    )

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

        # Decay epsilon
        agent.epsilon = max(0.01, agent.epsilon * 0.995)

        if i_episode % print_every == 0:
            avg = np.mean(scores_window)
            logger.info(f"Episode {i_episode} | Avg Score: {avg:.2f} | Epsilon: {agent.epsilon:.3f}")

    logger.info("Training complete!")
    return scores


if __name__ == "__main__":
    scores = train(n_episodes=10)
    print(f"Done. Episodes: {len(scores)}")
