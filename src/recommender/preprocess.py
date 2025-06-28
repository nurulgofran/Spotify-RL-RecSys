# src/recommender/preprocess.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Import the configuration from our new config file
from src.config import RAW_DATA_PATH, PROCESSED_FEATURES_PATH, TRACK_IDS_PATH


def preprocess_spotify_data():
    """
    Loads the raw Spotify dataset, selects and normalizes audio features,
    and saves the resulting feature matrix and track IDs.
    """
    print("Starting data preprocessing...")

    # 1. Load the dataset using the path from config
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df.dropna(subset=["track_id"], inplace=True)
        print(f"✅ Raw data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Raw data file not found at '{RAW_DATA_PATH}'.")
        return

    # Shuffle the DataFrame to randomize the data
    df = df.sample(frac=1).reset_index(drop=True)
    print("✅ DataFrame has been shuffled.")

    # 2. Select the numerical audio features
    feature_columns = [
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
    df_features = df[["track_id"] + feature_columns].copy()

    # 3. Normalize the features
    scaler = MinMaxScaler()
    df_features[feature_columns] = scaler.fit_transform(df_features[feature_columns])
    print("✅ Features have been normalized to a [0, 1] scale.")

    # 4. Save the processed data
    song_feature_matrix = df_features[feature_columns].to_numpy()
    track_ids = df_features["track_id"].tolist()

    np.save(PROCESSED_FEATURES_PATH, song_feature_matrix)
    pd.Series(track_ids).to_csv(TRACK_IDS_PATH, index=False, header=["track_id"])

    print(f"\n✅ Preprocessing complete!")
    print(f"Processed feature matrix saved to: {PROCESSED_FEATURES_PATH}")
    print(f"Track ID mapping saved to: {TRACK_IDS_PATH}")


if __name__ == "__main__":
    # This part allows you to run the script directly for testing
    # Note: You might need to adjust your VS Code run configuration
    # or run from the root directory as `python -m src.recommender.preprocess`
    preprocess_spotify_data()
