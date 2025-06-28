# main.py

from src.recommender.preprocess import preprocess_spotify_data
from src.recommender.train import train


def main():
    """
    Main function to run the ML pipeline.
    """
    print("--- Starting Music Recommender Pipeline ---")

    # Step 1: Preprocess the data (only needs to be run once)
    # print("\n[Phase 1/2] Preprocessing Data...")
    # preprocess_spotify_data()

    # Step 2: Train the RL Agent
    print("\n[Phase 2/2] Training Agent...")
    scores = train()

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()
