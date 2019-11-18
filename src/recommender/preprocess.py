"""
Data preprocessing module for Spotify music recommendation system.

This module handles loading, cleaning, and preprocessing the raw Spotify dataset
for use in the reinforcement learning recommendation system.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
from typing import Tuple

from src.config import (
    RAW_DATA_PATH,
    PROCESSED_FEATURES_PATH,
    TRACK_IDS_PATH,
    AUDIO_FEATURES,
    DataConfig,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_clean_data() -> pd.DataFrame:
    """
    Load and perform initial cleaning of the raw dataset.

    Returns:
        Cleaned pandas DataFrame

    Raises:
        FileNotFoundError: If the raw data file is not found
        ValueError: If the data format is invalid
    """
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Raw data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        logger.error(f"Raw data file not found at '{RAW_DATA_PATH}'")
        logger.error(
            "Please download the dataset and place it in the data/raw/ directory"
        )
        raise

    # Initial data validation
    required_columns = ["track_id"] + AUDIO_FEATURES
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Remove rows with missing track_id
    initial_count = len(df)
    df = df.dropna(subset=["track_id"])
    logger.info(f"Removed {initial_count - len(df)} rows with missing track_id")

    # Remove duplicate tracks
    initial_count = len(df)
    df = df.drop_duplicates(subset=["track_id"])
    logger.info(f"Removed {initial_count - len(df)} duplicate tracks")

    # Filter by popularity if available
    if "popularity" in df.columns:
        initial_count = len(df)
        df = df[df["popularity"] >= DataConfig.MIN_TRACK_POPULARITY]
        logger.info(
            f"Filtered to {len(df)} tracks with popularity >= {DataConfig.MIN_TRACK_POPULARITY}"
        )

    return df


def normalize_features(df: pd.DataFrame) -> Tuple[np.ndarray, pd.Series]:
    """
    Normalize audio features and extract relevant data.

    Args:
        df: Input DataFrame with audio features

    Returns:
        Tuple of (normalized_features_array, track_ids_series)
    """
    # Set random seed for reproducibility
    np.random.seed(DataConfig.RANDOM_SEED)

    # Shuffle the dataset
    df_shuffled = df.sample(frac=1, random_state=DataConfig.RANDOM_SEED).reset_index(
        drop=True
    )
    logger.info("Dataset shuffled for randomization")

    # Extract features and track IDs
    df_features = df_shuffled[["track_id"] + AUDIO_FEATURES].copy()

    # Check for missing values in audio features
    missing_counts = df_features[AUDIO_FEATURES].isnull().sum()
    if missing_counts.any():
        logger.warning(
            f"Missing values found in features: {missing_counts[missing_counts > 0].to_dict()}"
        )
        df_features = df_features.dropna(subset=AUDIO_FEATURES)
        logger.info(
            f"Dropped rows with missing features. Remaining: {len(df_features)}"
        )

    # Normalize features to [0, 1] range
    scaler = MinMaxScaler()
    df_features[AUDIO_FEATURES] = scaler.fit_transform(df_features[AUDIO_FEATURES])
    logger.info("Audio features normalized using MinMaxScaler")

    # Convert to numpy array and series
    song_feature_matrix = df_features[AUDIO_FEATURES].to_numpy()
    track_ids = df_features["track_id"]

    logger.info(f"Final dataset shape: {song_feature_matrix.shape}")
    logger.info(f"Features used: {AUDIO_FEATURES}")

    return song_feature_matrix, track_ids


def save_processed_data(song_features: np.ndarray, track_ids: pd.Series) -> None:
    """
    Save processed data to disk.

    Args:
        song_features: Normalized feature matrix
        track_ids: Series of track IDs
    """
    # Save feature matrix
    np.save(PROCESSED_FEATURES_PATH, song_features)
    logger.info(f"Song features saved to {PROCESSED_FEATURES_PATH}")

    # Save track IDs
    track_ids.to_csv(TRACK_IDS_PATH, index=False, header=["track_id"])
    logger.info(f"Track IDs saved to {TRACK_IDS_PATH}")


def preprocess_spotify_data() -> None:
    """
    Main preprocessing pipeline.

    Loads raw data, cleans it, normalizes features, and saves processed data.
    """
    logger.info("Starting data preprocessing pipeline...")

    try:
        # Load and clean data
        df = load_and_clean_data()

        # Normalize features
        song_features, track_ids = normalize_features(df)

        # Save processed data
        save_processed_data(song_features, track_ids)

        logger.info("✅ Data preprocessing completed successfully!")
        logger.info(f"   - Total songs: {len(track_ids):,}")
        logger.info(f"   - Feature dimensions: {song_features.shape[1]}")

    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        raise


def main() -> None:
    """Entry point for standalone execution."""
    preprocess_spotify_data()


if __name__ == "__main__":
    main()
