"""
Reinforcement Learning Environment for Music Recommendation.
"""

import numpy as np
import pandas as pd
from collections import deque
import logging

from src.config import PROCESSED_FEATURES_PATH, TRACK_IDS_PATH, EnvironmentConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SongRecommenderEnvironment:
    """Custom RL environment for song recommendation."""

    def __init__(self, history_length=EnvironmentConfig.HISTORY_LENGTH):
        self.history_length = history_length
        self._load_data()

        self.action_space_size = len(self.track_ids)
        self.state_space_size = self.song_features.shape[1]
        self.song_history = deque(maxlen=self.history_length)
        self.current_state = None
        self.episode_step_counter = 0

        logger.info(f"Environment initialized: {self.action_space_size} songs, {self.state_space_size} features")

    def _load_data(self):
        """Load preprocessed song features and track IDs."""
        try:
            self.song_features = np.load(PROCESSED_FEATURES_PATH)
            track_ids_df = pd.read_csv(TRACK_IDS_PATH)
            self.track_ids = track_ids_df["track_id"].tolist()
            logger.info("Processed data loaded successfully")
        except FileNotFoundError:
            logger.error("Processed data not found. Run preprocessing first.")
            raise

    def reset(self):
        """Reset the environment for a new episode."""
        self.song_history.clear()
        for _ in range(self.history_length):
            idx = np.random.randint(0, self.action_space_size)
            self.song_history.append(self.song_features[idx])

        self.current_state = np.mean(list(self.song_history), axis=0)
        self.episode_step_counter = 0
        return self.current_state.copy()

    def step(self, action_index):
        """Take an action in the environment."""
        if self.current_state is None:
            raise RuntimeError("Call reset() before step()")

        action_index = min(action_index, self.action_space_size - 1)
        self.episode_step_counter += 1

        # Simple reward based on feature average vs baseline
        song = self.song_features[action_index]
        reward = float(np.mean(song) - EnvironmentConfig.REWARD_BASELINE)

        self.song_history.append(song)
        self.current_state = np.mean(list(self.song_history), axis=0)

        done = self.episode_step_counter >= EnvironmentConfig.EPISODE_LENGTH
        return self.current_state.copy(), reward, done


if __name__ == "__main__":
    try:
        env = SongRecommenderEnvironment()
        state = env.reset()
        print(f"State shape: {state.shape}")

        action = np.random.randint(0, env.action_space_size)
        next_state, reward, done = env.step(action)
        print(f"Reward: {reward:.4f}, Done: {done}")
        print("Environment test passed!")
    except FileNotFoundError:
        print("Skipping test - no processed data available")
