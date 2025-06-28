from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

RAW_DATA_PATH = RAW_DATA_DIR / "dataset.csv"
PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "song_features.npy"
TRACK_IDS_PATH = PROCESSED_DATA_DIR / "track_ids.csv"
