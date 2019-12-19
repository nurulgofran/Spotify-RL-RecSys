"""
Reinforcement Learning Environment for Music Recommendation.

This module implements a custom RL environment that simulates user interactions
with a music recommendation system. The environment manages state transitions
and reward calculations based on music similarity.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import deque
import logging
from typing import Tuple, Optional

from src.config import PROCESSED_FEATURES_PATH, TRACK_IDS_PATH, EnvironmentConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SongRecommenderEnvironment:
    """
    Custom RL environment for music recommendation.

    The environment simulates a user listening session where:
    - State: Average audio features of recently played songs
    - Action: Recommending a song from the dataset
    - Reward: Similarity between recommended song and user's current taste
    """

    def __init__(self, history_length: int = EnvironmentConfig.HISTORY_LENGTH) -> None:
        """
        Initialize the music recommendation environment.

        Args:
            history_length: Number of recent songs to consider for state

        Raises:
            FileNotFoundError: If processed data files are not found
        """
        self.history_length = history_length

        # Load processed data
        self._load_data()

        # Environment properties
        self.action_space_size = len(self.track_ids)
        self.state_space_size = self.song_features.shape[1]

        # Episode state
        self.song_history: deque = deque(maxlen=self.history_length)
        self.current_state: Optional[np.ndarray] = None
        self.episode_step_counter = 0

        logger.info("Environment initialized:")
        logger.info(f"  - Songs in dataset: {self.action_space_size:,}")
        logger.info(f"  - Feature dimensions: {self.state_space_size}")
        logger.info(f"  - History length: {self.history_length}")

    def _load_data(self) -> None:
        """Load preprocessed song features and track IDs."""
        try:
            self.song_features = np.load(PROCESSED_FEATURES_PATH)
            track_ids_df = pd.read_csv(TRACK_IDS_PATH)
            self.track_ids = track_ids_df["track_id"].tolist()

            # Create mapping for efficient lookups
            self.track_id_to_index = {
                track_id: i for i, track_id in enumerate(self.track_ids)
            }

            logger.info("Processed data loaded successfully")

        except FileNotFoundError:
            logger.error("Could not find processed data files")
            logger.error("Please run the preprocessing script first:")
            logger.error("python -m src.recommender.preprocess")
            raise

    def _get_state(self) -> np.ndarray:
        """
        Calculate current state as average of recent song features.

        Returns:
            State vector representing user's current musical taste
        """
        if len(self.song_history) == 0:
            return np.zeros(self.state_space_size)
        return np.mean(list(self.song_history), axis=0)

    def _calculate_reward(self, action_index: int) -> float:
        """
        Calculate reward for a given action based on similarity to current state.

        Args:
            action_index: Index of the recommended song

        Returns:
            Reward value (cosine similarity - baseline)
        """
        action_song_features = self.song_features[action_index].reshape(1, -1)
        current_state_reshaped = self.current_state.reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(current_state_reshaped, action_song_features)[0][
            0
        ]

        # Center reward around 0 to encourage meaningful improvement
        reward = similarity - EnvironmentConfig.REWARD_BASELINE

        return reward

    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode.

        Returns:
            Initial state vector
        """
        # Clear history
        self.song_history.clear()

        # Initialize with random songs to create diverse starting states
        for _ in range(self.history_length):
            random_song_index = np.random.randint(0, self.action_space_size)
            initial_song_features = self.song_features[random_song_index]
            self.song_history.append(initial_song_features)

        # Calculate initial state
        self.current_state = self._get_state()
        self.episode_step_counter = 0

        return self.current_state.copy()

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool]:
        """
        Take an action in the environment.

        Args:
            action_index: Index of the song to recommend

        Returns:
            Tuple of (next_state, reward, done)

        Raises:
            RuntimeError: If step() is called before reset()
            ValueError: If action_index is invalid
        """
        if self.current_state is None:
            raise RuntimeError("Cannot call step() before calling reset()")

        if not (0 <= action_index < self.action_space_size):
            raise ValueError(
                f"Invalid action index: {action_index}. Must be in [0, {self.action_space_size})"
            )

        self.episode_step_counter += 1

        # Calculate reward
        reward = self._calculate_reward(action_index)

        # Update state with new song
        recommended_song_features = self.song_features[action_index]
        self.song_history.append(recommended_song_features)
        next_state = self._get_state()

        # Update current state
        self.current_state = next_state

        # Check if episode is done
        done = self.episode_step_counter >= EnvironmentConfig.EPISODE_LENGTH

        return next_state.copy(), reward, done

    def get_song_info(self, action_index: int) -> str:
        """
        Get track ID for a given action index.

        Args:
            action_index: Index of the song

        Returns:
            Track ID string
        """
        if not (0 <= action_index < self.action_space_size):
            raise ValueError(f"Invalid action index: {action_index}")

        return self.track_ids[action_index]

    def get_state_info(self) -> dict:
        """
        Get information about the current state.

        Returns:
            Dictionary with state information
        """
        return {
            "episode_step": self.episode_step_counter,
            "history_length": len(self.song_history),
            "state_shape": self.current_state.shape
            if self.current_state is not None
            else None,
            "recent_songs": [
                self.track_ids[i] for i in range(min(5, len(self.track_ids)))
            ],
        }


def main() -> None:
    """Test the environment implementation."""
    logger.info("Testing SongRecommenderEnvironment...")

    try:
        # Initialize environment
        env = SongRecommenderEnvironment()

        # Test reset
        initial_state = env.reset()
        assert initial_state.shape == (env.state_space_size,), (
            f"Invalid state shape: {initial_state.shape}"
        )
        logger.info(f"✅ Reset test passed. State shape: {initial_state.shape}")

        # Test step
        random_action = np.random.randint(0, env.action_space_size)
        next_state, reward, done = env.step(random_action)

        # Validate outputs
        assert next_state.shape == (env.state_space_size,), (
            f"Invalid next state shape: {next_state.shape}"
        )
        assert isinstance(reward, (int, float)), f"Invalid reward type: {type(reward)}"
        assert isinstance(done, bool), f"Invalid done type: {type(done)}"

        logger.info("✅ Step test passed:")
        logger.info(f"   - Action: {random_action}")
        logger.info(f"   - Reward: {reward:.4f}")
        logger.info(f"   - Done: {done}")
        logger.info(f"   - Track ID: {env.get_song_info(random_action)}")

        # Test multiple steps
        total_reward = reward
        for _ in range(5):
            action = np.random.randint(0, env.action_space_size)
            next_state, reward, done = env.step(action)
            total_reward += reward
            if done:
                break

        logger.info(f"✅ Multi-step test passed. Total reward: {total_reward:.4f}")
        logger.info("🎵 Environment test completed successfully!")

    except FileNotFoundError:
        logger.error("❌ Test failed: Processed data files not found")
        logger.error("Run preprocessing first: python -m src.recommender.preprocess")
    except Exception as e:
        logger.error(f"❌ Test failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
