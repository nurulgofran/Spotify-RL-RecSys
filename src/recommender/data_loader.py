import pandas as pd


def load_spotify_data(path="dataset.csv"):
    try:
        df = pd.read_csv(path)
        print("✅ Dataset loaded successfully!")
        print("-" * 30)
        return df
    except FileNotFoundError:
        print(
            f"❌ Error: File not found at '{path}'. Please ensure the dataset is in the correct directory."
        )
        return None


if __name__ == "__main__":
    spotify_df = load_spotify_data()

    if spotify_df is not None:
        print(f"Shape of the dataset (rows, columns): {spotify_df.shape}")
        print("\nFirst 5 rows of the dataset:")
        print(spotify_df.head())

        print("\nInformation about the dataset columns and data types:")
        # Using .info() gives a concise summary of the DataFrame
        spotify_df.info()
