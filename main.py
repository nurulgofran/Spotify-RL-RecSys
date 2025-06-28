# main.py

from src.recommender.preprocess import preprocess_spotify_data


def main():
    """
    Main function to run the ML pipeline.
    """
    print("--- Starting Music Recommender Pipeline ---")

    # Step 1: Preprocess the data
    preprocess_spotify_data()

    # Step 2: Build the RL Environment (coming next)
    # build_environment()

    # Step 3: Train the Agent (coming later)
    # train_agent()

    print("\n--- Pipeline Finished ---")


if __name__ == "__main__":
    main()
