from src.recommender.preprocess import preprocess_spotify_data
from src.recommender.train import train


def main():
    print("Starting Music Recommender Pipeline...")
    scores = train()
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
