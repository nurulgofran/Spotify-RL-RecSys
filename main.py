"""Main script for Spotify RL Recommendation System."""

from src.recommender.preprocess import preprocess_spotify_data
from src.recommender.train import train


def main():
    print("Spotify RL Recommendation System")
    print("=" * 40)

    print("\nPreprocessing data...")
    preprocess_spotify_data()

    print("\nTraining agent...")
    scores = train()

    print(f"\nDone! Trained for {len(scores)} episodes.")
    if scores:
        avg = sum(scores[-10:]) / min(len(scores), 10)
        print(f"Final avg score: {avg:.2f}")


if __name__ == "__main__":
    main()
