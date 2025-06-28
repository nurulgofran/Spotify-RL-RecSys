import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from src.config import PROCESSED_FEATURES_PATH, TRACK_IDS_PATH
from collections import deque


class SongRecommenderEnvironment:
    def __init__(self, history_length=5):
        self.history_length = history_length

        try:
            self.song_features = np.load(PROCESSED_FEATURES_PATH)
            track_ids_df = pd.read_csv(TRACK_IDS_PATH)
            self.track_ids = track_ids_df["track_id"].tolist()
        except FileNotFoundError as e:
            print("❌ Error: Could not find processed data files.")
            print("  Please run the preprocessing script first.")
            raise

        self.track_id_to_index = {
            track_id: i for i, track_id in enumerate(self.track_ids)
        }

        self.action_space_size = len(self.track_ids)
        self.state_space_size = self.song_features.shape[1]

        self.song_history = deque(maxlen=self.history_length)
        self.current_state = None
        self.episode_step_counter = 0

    def _get_state(self):
        return np.mean(list(self.song_history), axis=0)

    def reset(self):
        self.song_history.clear()

        random_song_index = np.random.randint(0, self.action_space_size)
        initial_song_features = self.song_features[random_song_index]
        for _ in range(self.history_length):
            self.song_history.append(initial_song_features)

        self.current_state = self._get_state()
        self.episode_step_counter = 0

        return self.current_state

    def step(self, action_index):
        if self.current_state is None:
            raise RuntimeError("Cannot call step() before calling reset().")

        self.episode_step_counter += 1

        action_song_features = self.song_features[action_index].reshape(1, -1)

        current_state_reshaped = self.current_state.reshape(1, -1)
        REWARD_BASELINE = 0.5
        reward = (
            cosine_similarity(current_state_reshaped, action_song_features)[0][0]
            - REWARD_BASELINE
        )

        self.song_history.append(self.song_features[action_index])
        next_state = self._get_state()

        self.current_state = next_state

        done = self.episode_step_counter >= 20

        return next_state, reward, done


if __name__ == "__main__":
    try:
        env = SongRecommenderEnvironment()
        initial_state = env.reset()
        random_action = np.random.randint(0, env.action_space_size)
        next_state, reward, done = env.step(random_action)

        assert next_state.shape == (env.state_space_size,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        print("✅ Environment test passed")

    except FileNotFoundError:
        print("\nTest failed. Make sure your processed data exists.")
    except Exception as e:
        print(f"\nAn error occurred during testing: {e}")
