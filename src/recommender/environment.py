# src/recommender/environment.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
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

        # Add a placeholder for the current state
        self.current_state = None
        # Add a counter for the episode length
        self.episode_step_counter = 0

        print("✅ Environment initialized successfully.")
        print(f"   - Number of songs (actions): {self.action_space_size}")
        print(f"   - Number of features (state size): {self.state_space_size}")

    def reset(self):
        """
        Resets the environment to start a new user session (episode).

        It simulates a user starting their listening by picking a random song.
        The features of this song become the initial state.

        Returns:
            np.ndarray: The initial state vector.
        """
        print("\n🔄 Resetting environment for new episode...")

        # 1. Choose a random song index to start the session
        random_song_index = np.random.randint(0, self.action_space_size)

        # 2. Set the initial state to the features of that random song
        self.current_state = self.song_features[random_song_index]
        # Reset the step counter
        self.episode_step_counter = 0

        random_track_id = self.track_ids[random_song_index]
        print(f"   - Starting song ID: {random_track_id}")
        print("   - Initial state set.")

        # 3. Return the initial state to the agent
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
        # The reward is the cosine similarity between the current state and the chosen song.
        # We reshape the current_state to be a 2D array for the function.
        current_state_reshaped = self.current_state.reshape(1, -1)
        reward = cosine_similarity(current_state_reshaped, action_song_features)[0][0]

        # 3. Determine the next state
        # The next state is simply the features of the song just recommended.
        next_state = self.song_features[action_index]

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
