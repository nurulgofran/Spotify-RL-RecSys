from src.recommender.preprocess import preprocess_spotify_data
from src.recommender.train import train


def main():
    print("--- Starting Music Recommender Pipeline ---")
    print("\n[Phase 2/2] Training Agent...")
    scores = train()
    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()
