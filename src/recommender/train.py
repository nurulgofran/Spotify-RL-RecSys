"""
Training module for the DQN music recommendation agent.

This module handles the training loop, progress tracking, and model persistence
for the reinforcement learning music recommendation system.
"""

from collections import deque
import numpy as np
import logging
from datetime import datetime
from typing import List, Optional
import os

from src.recommender.environment import SongRecommenderEnvironment
from src.recommender.agent import Agent
from src.config import TrainingConfig, MODEL_CHECKPOINT_PATH, LOGS_DIR

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_training_logger() -> logging.Logger:
    """Set up a dedicated logger for training with file output."""
    train_logger = logging.getLogger("training")
    train_logger.setLevel(logging.INFO)

    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"training_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add handler to logger
    train_logger.addHandler(file_handler)

    return train_logger


def train(
    n_episodes: int = TrainingConfig.N_EPISODES,
    max_t: int = TrainingConfig.MAX_STEPS_PER_EPISODE,
    print_every: int = TrainingConfig.PRINT_EVERY,
    solve_score: float = TrainingConfig.SOLVE_SCORE,
    save_model: bool = True,
    model_path: Optional[str] = None,
) -> List[float]:
    """
    Train the DQN agent using the music recommendation environment.

    Args:
        n_episodes: Number of training episodes
        max_t: Maximum steps per episode
        print_every: Frequency of progress updates
        solve_score: Target average score to consider solved
        save_model: Whether to save the trained model
        model_path: Custom path for saving the model

    Returns:
        List of scores for each episode
    """
    # Setup logging
    train_logger = setup_training_logger()

    logger.info("🎵 Starting DQN training for music recommendation...")
    logger.info("Training parameters:")
    logger.info(f"  - Episodes: {n_episodes}")
    logger.info(f"  - Max steps per episode: {max_t}")
    logger.info(f"  - Target score: {solve_score}")

    # Initialize environment and agent
    env = SongRecommenderEnvironment()
    agent = Agent(
        state_size=env.state_space_size,
        action_size=env.action_space_size,
        buffer_size=TrainingConfig.BUFFER_SIZE,
        batch_size=TrainingConfig.BATCH_SIZE,
        gamma=TrainingConfig.GAMMA,
        tau=TrainingConfig.TAU,
        lr=TrainingConfig.LEARNING_RATE,
    )

    # Training tracking
    scores = []
    scores_window = deque(maxlen=100)
    best_avg_score = -np.inf
    solved = False

    logger.info(
        f"Environment: {env.action_space_size:,} songs, {env.state_space_size} features"
    )
    logger.info("Starting training loop...")

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

        # Store score and update epsilon
        scores_window.append(score)
        scores.append(score)

        # Decay epsilon
        agent.epsilon = max(
            TrainingConfig.EPSILON_MIN, agent.epsilon * TrainingConfig.EPSILON_DECAY
        )

        # Calculate average score
        avg_score = np.mean(scores_window)

        # Log progress
        if i_episode % print_every == 0:
            logger.info(
                f"Episode {i_episode:4d} | "
                f"Average Score: {avg_score:7.2f} | "
                f"Current Score: {score:7.2f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

            train_logger.info(
                f"Episode {i_episode} - Avg: {avg_score:.2f}, "
                f"Current: {score:.2f}, Epsilon: {agent.epsilon:.3f}"
            )

        # Check if we've improved
        if avg_score > best_avg_score:
            best_avg_score = avg_score

        # Check if solved
        if avg_score >= solve_score and not solved:
            logger.info(f"🎉 Environment solved in {i_episode - 100} episodes!")
            logger.info(f"Average Score: {avg_score:.2f}")
            solved = True

            if save_model:
                save_path = model_path or MODEL_CHECKPOINT_PATH
                agent.save_model(str(save_path))
                logger.info(f"Model saved to {save_path}")

            break

    # Final model save if not already saved
    if save_model and not solved:
        save_path = model_path or MODEL_CHECKPOINT_PATH
        agent.save_model(str(save_path))
        logger.info(f"Training completed. Final model saved to {save_path}")

    # Training summary
    logger.info("🏁 Training Summary:")
    logger.info(f"  - Episodes completed: {len(scores)}")
    logger.info(f"  - Best average score: {best_avg_score:.2f}")
    logger.info(f"  - Final epsilon: {agent.epsilon:.3f}")
    logger.info(f"  - Solved: {'Yes' if solved else 'No'}")

    return scores


def evaluate_agent(
    agent: Agent, env: SongRecommenderEnvironment, n_episodes: int = 10
) -> List[float]:
    """
    Evaluate a trained agent's performance.

    Args:
        agent: Trained DQN agent
        env: Environment to evaluate on
        n_episodes: Number of evaluation episodes

    Returns:
        List of evaluation scores
    """
    logger.info(f"Evaluating agent for {n_episodes} episodes...")

    eval_scores = []

    for episode in range(n_episodes):
        state = env.reset()
        score = 0

        while True:
            # Use greedy policy (no exploration)
            action = agent.act(state, add_noise=False)
            next_state, reward, done = env.step(action)

            state = next_state
            score += reward

            if done:
                break

        eval_scores.append(score)
        logger.info(f"Evaluation Episode {episode + 1}: Score = {score:.2f}")

    avg_score = np.mean(eval_scores)
    logger.info(f"Average evaluation score: {avg_score:.2f}")

    return eval_scores


def load_and_evaluate(model_path: str, n_episodes: int = 10) -> List[float]:
    """
    Load a trained model and evaluate its performance.

    Args:
        model_path: Path to the saved model
        n_episodes: Number of evaluation episodes

    Returns:
        List of evaluation scores
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Initialize environment and agent
    env = SongRecommenderEnvironment()
    agent = Agent(state_size=env.state_space_size, action_size=env.action_space_size)

    # Load the trained model
    agent.load_model(model_path)
    logger.info(f"Model loaded from {model_path}")

    # Evaluate
    return evaluate_agent(agent, env, n_episodes)


def main() -> None:
    """Main training entry point."""
    try:
        scores = train()
        logger.info("Training completed successfully!")

        # Optional: Quick evaluation
        if len(scores) > 0:
            logger.info(f"Final 10 episodes average: {np.mean(scores[-10:]):.2f}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
