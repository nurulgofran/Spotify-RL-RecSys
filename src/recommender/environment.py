# src/recommender/environment.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_FEATURES_PATH, TRACK_IDS_PATH
from collections import deque


class SongRecommenderEnvironment:
    """
    A custom Reinforcement Learning Environment for a Song Recommender System.

    This environment simulates a user's listening session. The agent's goal
    is to recommend songs that match the user's current taste (state).
    """

    def __init__(self, history_length=5):
        """
        Initializes the environment by loading the processed song data.

        Args:
            history_length (int): The number of past songs to consider for the state.
        """
        print("Initializing the recommendation environment...")
        self.history_length = history_length

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

        # NEW STATE REPRESENTATION
        # Use a deque to store the feature vectors of the last few songs
        self.song_history = deque(maxlen=self.history_length)
        self.current_state = None
        # Add a counter for the episode length
        self.episode_step_counter = 0

        print("✅ Environment initialized successfully.")
        print(f"   - Number of songs (actions): {self.action_space_size}")
        print(f"   - Number of features (state size): {self.state_space_size}")

    def _get_state(self):
        """Calculates the state as the mean of the song history."""
        # The state is the average of the features of the songs in the history
        return np.mean(list(self.song_history), axis=0)

    def reset(self):
        """
        Resets the environment to start a new user session (episode).

        Returns:
            np.ndarray: The initial state vector.
        """
        print("\n🔄 Resetting environment for new episode...")

        # Clear the history
        self.song_history.clear()

        # To start, we need to fill the history. We'll do this by picking one
        # random song and duplicating its features `history_length` times.
        random_song_index = np.random.randint(0, self.action_space_size)
        initial_song_features = self.song_features[random_song_index]
        for _ in range(self.history_length):
            self.song_history.append(initial_song_features)

        self.current_state = self._get_state()
        self.episode_step_counter = 0

        random_track_id = self.track_ids[random_song_index]
        print(f"   - Starting song ID: {random_track_id}")
        print("   - Initial state set.")

        # Return the initial state to the agent
        return self.current_state

    def step(self, action_index):
        """
        Processes one step in the environment.

        Args:
            action_index (int): The index of the song recommended by the agent.

        Returns:
            tuple: A tuple containing (next_state, reward, done).
        """
        if self.current_state is None:
            raise RuntimeError("Cannot call step() before calling reset().")

        self.episode_step_counter += 1

        # 1. Get the feature vector for the song the agent chose (the action)
        action_song_features = self.song_features[action_index].reshape(1, -1)

        # 2. Calculate the reward
        # The current state is the mean of the history
        current_state_reshaped = self.current_state.reshape(1, -1)
        reward = cosine_similarity(current_state_reshaped, action_song_features)[0][0]

        # 3. UPDATE THE STATE HISTORY
        # Add the new song's features to the history
        self.song_history.append(self.song_features[action_index])
        # The new state is the updated average of the history
        next_state = self._get_state()

        # Update the environment's current state
        self.current_state = next_state

        # 4. Check if the episode is done
        # For simplicity, we'll end the episode after 20 recommendations.
        done = self.episode_step_counter >= 20

        return next_state, reward, done


# Update the test block at the bottom of the file
if __name__ == "__main__":
    try:
        env = SongRecommenderEnvironment()
        initial_state = env.reset()

        print("\n--- Testing Step ---")
        # Simulate taking a random action
        random_action = np.random.randint(0, env.action_space_size)
        print(f"Simulating action: Recommend song with index {random_action}")

        # Take a step
        next_state, reward, done = env.step(random_action)

        print(f"Next State (first 5 features): {next_state[:5]}")
        print(f"Reward received: {reward:.4f}")
        print(f"Episode done: {done}")

        # Verify shapes and types
        assert next_state.shape == (env.state_space_size,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        print("\n✅ Test passed: step() returned the correct data types and shapes.")

    except FileNotFoundError:
        print("\nTest failed. Make sure your processed data exists.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
