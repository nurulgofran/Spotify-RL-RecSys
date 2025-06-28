import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from src.config import RAW_DATA_PATH, PROCESSED_FEATURES_PATH, TRACK_IDS_PATH


def preprocess_spotify_data():
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        df.dropna(subset=["track_id"], inplace=True)
        print(f"Raw data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"❌ Error: Raw data file not found at '{RAW_DATA_PATH}'.")
        return

    df = df.sample(frac=1).reset_index(drop=True)

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

    scaler = MinMaxScaler()
    df_features[feature_columns] = scaler.fit_transform(df_features[feature_columns])

    song_feature_matrix = df_features[feature_columns].to_numpy()
    track_ids = df_features["track_id"].tolist()

    np.save(PROCESSED_FEATURES_PATH, song_feature_matrix)
    pd.Series(track_ids).to_csv(TRACK_IDS_PATH, index=False, header=["track_id"])

    print("Preprocessing complete!")


if __name__ == "__main__":
    preprocess_spotify_data()
