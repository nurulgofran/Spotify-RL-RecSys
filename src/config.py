"""
Configuration file for the Spotify Reinforcement Learning Recommender.

This module contains all configuration parameters and file paths used throughout
the project. Centralizing configuration makes the codebase more maintainable
and allows for easy parameter tuning.
"""

from pathlib import Path
from typing import List

# Project structure
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"
PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "song_features.npy"
TRACK_IDS_PATH = PROCESSED_DATA_DIR / "track_ids.csv"

# Model file paths
MODEL_CHECKPOINT_PATH = MODELS_DIR / "q_network_checkpoint.pth"

# Audio features to use for state representation
AUDIO_FEATURES: List[str] = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence",
]


# Training hyperparameters
class TrainingConfig:
    """Training configuration parameters."""

    N_EPISODES = 2000
    MAX_STEPS_PER_EPISODE = 100
    PRINT_EVERY = 100
    SOLVE_SCORE = 15.0

    # Agent hyperparameters
    BUFFER_SIZE = int(1e5)
    BATCH_SIZE = 64
    GAMMA = 0.99  # Discount factor
    TAU = 1e-3  # Soft update parameter
    LEARNING_RATE = 5e-4
    EPSILON_START = 1.0
    EPSILON_MIN = 0.01
    EPSILON_DECAY = 0.995


# Environment configuration
class EnvironmentConfig:
    """Environment configuration parameters."""

    HISTORY_LENGTH = 5
    EPISODE_LENGTH = 20
    REWARD_BASELINE = 0.5


# Data processing configuration
class DataConfig:
    """Data processing configuration parameters."""

    RANDOM_SEED = 42
    TEST_SIZE = 0.2
    MIN_TRACK_POPULARITY = 20  # Minimum popularity threshold for songs
