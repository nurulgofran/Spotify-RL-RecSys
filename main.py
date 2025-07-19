"""
Main entry point for the Spotify Reinforcement Learning Recommender.

This script orchestrates the complete pipeline from data preprocessing
to model training for the music recommendation system.
"""

import logging
import sys
import argparse

from src.recommender.preprocess import preprocess_spotify_data
from src.recommender.train import train, load_and_evaluate
from src.config import RAW_DATA_PATH, MODEL_CHECKPOINT_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_data_exists() -> bool:
    """Check if the raw data file exists."""
    if not RAW_DATA_PATH.exists():
        logger.error(f"Raw data file not found at {RAW_DATA_PATH}")
        logger.error(
            "Please download the Spotify dataset and place it in data/raw/dataset.csv"
        )
        logger.error(
            "Dataset URL: https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db"
        )
        return False
    return True


def run_preprocessing() -> bool:
    """Run the data preprocessing pipeline."""
    logger.info("🔄 Starting data preprocessing...")
    try:
        preprocess_spotify_data()
        logger.info("✅ Data preprocessing completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {str(e)}")
        return False


def run_training() -> bool:
    """Run the model training pipeline."""
    logger.info("🤖 Starting model training...")
    try:
        scores = train()
        logger.info("✅ Model training completed successfully")
        logger.info(f"Training episodes: {len(scores)}")
        if len(scores) > 10:
            logger.info(
                f"Final 10 episodes average score: {sum(scores[-10:]) / 10:.2f}"
            )
        return True
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return False


def run_evaluation() -> bool:
    """Run model evaluation."""
    if not MODEL_CHECKPOINT_PATH.exists():
        logger.error(f"No trained model found at {MODEL_CHECKPOINT_PATH}")
        logger.error("Please train a model first using --train or --full")
        return False

    logger.info("📊 Starting model evaluation...")
    try:
        eval_scores = load_and_evaluate(str(MODEL_CHECKPOINT_PATH))
        avg_score = sum(eval_scores) / len(eval_scores)
        logger.info(f"✅ Evaluation completed. Average score: {avg_score:.2f}")
        return True
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {str(e)}")
        return False


def main() -> int:
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Spotify Reinforcement Learning Music Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline (preprocess + train)
  python main.py --preprocess-only  # Only preprocess data
  python main.py --train-only       # Only train model (requires preprocessed data)
  python main.py --evaluate         # Evaluate existing model
        """,
    )

    parser.add_argument(
        "--preprocess-only", action="store_true", help="Only run data preprocessing"
    )

    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run model training (requires preprocessed data)",
    )

    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate an existing trained model"
    )

    parser.add_argument(
        "--skip-data-check",
        action="store_true",
        help="Skip checking for raw data file (useful for testing)",
    )

    args = parser.parse_args()

    logger.info("🎵 Spotify Reinforcement Learning Music Recommender")
    logger.info("=" * 60)

    # Check for data file unless specifically skipped
    if not args.skip_data_check and not args.evaluate and not check_data_exists():
        return 1

    success = True

    if args.evaluate:
        # Only run evaluation
        success = run_evaluation()

    elif args.preprocess_only:
        # Only run preprocessing
        success = run_preprocessing()

    elif args.train_only:
        # Only run training
        success = run_training()

    else:
        # Run full pipeline (default)
        logger.info("Running full pipeline: preprocessing + training")

        # Step 1: Preprocessing
        if not run_preprocessing():
            return 1

        # Step 2: Training
        if not run_training():
            return 1

        logger.info("🎉 Full pipeline completed successfully!")

    if success:
        logger.info("=" * 60)
        logger.info("✅ All operations completed successfully!")
        return 0
    else:
        logger.error("❌ Some operations failed. Check the logs above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
