import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def preprocess_spotify_data(
    raw_data_path="dataset.csv", output_path="song_features.npy"
):
    """
    Loads the raw Spotify dataset, selects and normalizes audio features,
    and saves the resulting feature matrix.

    Args:
        raw_data_path (str): Path to the raw dataset.csv file.
        output_path (str): Path to save the processed .npy file.
    """
    print("Starting data preprocessing...")

    # 1. Load the dataset
    try:
        df = pd.read_csv(raw_data_path)
        # Drop rows with missing track_id, as they are unusable
        df.dropna(subset=["track_id"], inplace=True)
        print(f"✅ Raw data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Raw data file not found at '{raw_data_path}'.")
        return

    # 2. Select the numerical audio features for our model
    # These will define our state space and be used for reward calculation.
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

    print(f"\nSelected feature columns: {feature_columns}")

    # Create a new DataFrame with only the essential columns
    df_features = df[["track_id"] + feature_columns].copy()

    # 3. Normalize the features
    # We use MinMaxScaler to scale features to a range of [0, 1].
    # This is essential for fair similarity calculations and for neural networks.
    scaler = MinMaxScaler()

    # Fit and transform the feature columns
    df_features[feature_columns] = scaler.fit_transform(df_features[feature_columns])

    print("\n✅ Features have been normalized to a [0, 1] scale.")
    print("Sample of normalized data:")
    print(df_features.head())

    # 4. Save the processed data for later use
    # We save the features as a NumPy array for efficient loading.
    # We also save the corresponding track_ids.

    # Convert features to a NumPy array
    song_feature_matrix = df_features[feature_columns].to_numpy()

    # Get the list of track_ids in the same order
    track_ids = df_features["track_id"].tolist()

    # Save the feature matrix
    np.save(output_path, song_feature_matrix)

    # Save the track_ids list
    pd.Series(track_ids).to_csv("track_ids.csv", index=False, header=["track_id"])

    print(f"\n✅ Preprocessing complete!")
    print(f"Processed feature matrix saved to: {output_path}")
    print(f"Track ID mapping saved to: track_ids.csv")


if __name__ == "__main__":
    preprocess_spotify_data()
