# src/recommender/environment.py

import numpy as np
import pandas as pd
from src.config import PROCESSED_FEATURES_PATH, TRACK_IDS_PATH


class SongRecommenderEnvironment:
    """
    A custom Reinforcement Learning Environment for a Song Recommender System.

    This environment simulates a user's listening session. The agent's goal
    is to recommend songs that match the user's current taste (state).
    """

    def __init__(self):
        """
        Initializes the environment by loading the processed song data.
        """
        print("Initializing the recommendation environment...")

        # 1. Load the processed song feature matrix and track IDs
        try:
            self.song_features = np.load(PROCESSED_FEATURES_PATH)
            track_ids_df = pd.read_csv(TRACK_IDS_PATH)
            self.track_ids = track_ids_df["track_id"].tolist()
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find processed data files.")
            print(f"  Please run the preprocessing script first.")
            print(f"  Details: {e}")
            raise

        # 2. Create a mapping from track_id to its index in the feature matrix
        # This allows for quick lookups: given a track_id, find its features.
        self.track_id_to_index = {
            track_id: i for i, track_id in enumerate(self.track_ids)
        }

        # 3. Define the dimensions of our state and action spaces
        # The number of possible actions is the total number of songs.
        self.action_space_size = len(self.track_ids)
        # The state is represented by the song features (e.g., danceability, energy, etc.).
        # The state space size is the number of features.
        self.state_space_size = self.song_features.shape[1]

        print("✅ Environment initialized successfully.")
        print(f"   - Number of songs (actions): {self.action_space_size}")
        print(f"   - Number of features (state size): {self.state_space_size}")


# You can add this temporary block to test if the file runs correctly
if __name__ == "__main__":
    try:
        env = SongRecommenderEnvironment()
    except FileNotFoundError:
        print("\nTest failed. Make sure your processed data exists.")
