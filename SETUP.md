# Setup and Installation Guide

This guide will help you set up the Spotify Reinforcement Learning Music Recommender on your local machine.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Git
- ~2GB of free disk space (for dataset and models)

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/nurulgofran/spotify-reinforcement-learning-recommendations-.git
cd spotify-reinforcement-learning-recommendations-
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

**Using conda:**
```bash
conda create -n spotify-rl python=3.9
conda activate spotify-rl
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

1. Go to [Kaggle - Ultimate Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
2. Download the dataset (you may need to create a Kaggle account)
3. Extract the CSV file and rename it to `dataset.csv`
4. Place it in the `data/raw/` directory

Your directory structure should look like:
```
spotify-reinforcement-learning-recommendations-/
├── data/
│   └── raw/
│       └── dataset.csv  # <-- Place the downloaded file here
├── src/
├── main.py
└── requirements.txt
```

### 5. Run the Project

**Full pipeline (preprocessing + training):**
```bash
python main.py
```

**Step by step:**
```bash
# 1. Preprocess the data
python main.py --preprocess-only

# 2. Train the model
python main.py --train-only

# 3. Evaluate the model
python main.py --evaluate
```

## Installation Options

### Option 1: CPU-only PyTorch (Default)

The default `requirements.txt` installs CPU-only PyTorch, which works on all systems but is slower for training.

### Option 2: GPU-accelerated PyTorch (CUDA)

For faster training with NVIDIA GPUs:

1. Install CUDA toolkit from [NVIDIA](https://developer.nvidia.com/cuda-toolkit)
2. Replace PyTorch installation:
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

### Option 3: Apple Silicon (M1/M2 Macs)

For optimized performance on Apple Silicon:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

## Troubleshooting

### Common Issues

**1. "Raw data file not found" error:**
- Ensure `dataset.csv` is placed in `data/raw/` directory
- Check that the file is named exactly `dataset.csv`

**2. Memory errors during training:**
- Reduce batch size in `src/config.py`
- Use CPU-only PyTorch if you have limited GPU memory

**3. Import errors:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`

**4. Slow training:**
- Consider using GPU-accelerated PyTorch
- Reduce the number of training episodes in `src/config.py`

### Dataset Issues

If you can't access Kaggle or prefer an alternative:

1. Any Spotify dataset with audio features will work
2. Required columns: `track_id`, `acousticness`, `danceability`, `energy`, `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`
3. Place the CSV file as `data/raw/dataset.csv`

### Performance Optimization

**For faster training:**
- Use GPU acceleration if available
- Increase batch size (if you have enough memory)
- Reduce the size of the dataset in preprocessing

**For lower memory usage:**
- Reduce buffer size in agent configuration
- Use smaller neural network architecture
- Process data in smaller chunks

## Development Setup

For contributors or advanced users:

```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
python -m pytest tests/

# Format code
black src/ main.py

# Check code quality
flake8 src/ main.py
mypy src/ main.py
```

## Verification

To verify your installation is working:

```bash
# Test individual components
python -m src.recommender.agent
python -m src.recommender.environment
python -m src.recommender.preprocess

# Run a quick training test (5 episodes)
python -c "from src.recommender.train import train; train(n_episodes=5)"
```

You should see:
- ✅ Agent test passed
- ✅ Environment test passed  
- ✅ Data preprocessing completed
- Training progress output

## Next Steps

Once installed:

1. **Read the README.md** for project overview
2. **Check out the code** in `src/` directory
3. **Experiment with hyperparameters** in `src/config.py`
4. **Run your first training** with `python main.py`

## Getting Help

If you encounter issues:

1. Check this setup guide
2. Look at the troubleshooting section
3. Search existing [GitHub issues](https://github.com/nurulgofran/spotify-reinforcement-learning-recommendations-/issues)
4. Create a new issue with:
   - Your operating system
   - Python version (`python --version`)
   - Error message and stack trace
   - Steps to reproduce

Happy coding! 🎵
