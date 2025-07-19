# Spotify Reinforcement Learning Music Recommender

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> A sophisticated music recommendation system powered by Deep Reinforcement Learning that learns user preferences in real-time.

## 🎯 Overview

This project implements a novel approach to music recommendation using **Deep Q-Networks (DQN)** to create an intelligent agent that learns from user interactions. Unlike traditional collaborative filtering or content-based methods, this system adapts dynamically to user preferences, capturing their current "musical mood" through reinforcement learning.

### Key Innovation

Instead of static recommendation algorithms, we frame music recommendation as a **sequential decision-making problem** where an AI agent learns the optimal policy for song recommendations through trial and error, just like how a human DJ learns to read the room.

## ✨ Features

- **🤖 Deep Reinforcement Learning**: Uses DQN with experience replay and target networks
- **🎵 Dynamic Adaptation**: Learns user preferences in real-time during listening sessions  
- **📊 Rich State Representation**: Leverages audio features (danceability, energy, valence, etc.)
- **⚡ Efficient Training**: Offline training with simulated user sessions
- **🔧 Modular Architecture**: Clean, extensible codebase with professional structure
- **📈 Comprehensive Logging**: Detailed training metrics and evaluation tools
- **🎛️ Configurable Hyperparameters**: Easy experimentation with different settings

## 🧠 How It Works

| Component | Description |
|-----------|-------------|
| **Agent** | Deep Q-Network that learns optimal song recommendation policy |
| **Environment** | Simulated user listening session with reward feedback |
| **State** | Average audio features of recently played songs (user's current taste) |
| **Action** | Recommending a specific song from the dataset |
| **Reward** | Cosine similarity between recommended song and user's current preferences |

## 🛠️ Technical Stack

- **Core**: Python 3.9+, NumPy, Pandas
- **ML/DL**: PyTorch, scikit-learn
- **Data**: Spotify Dataset (100k+ songs with audio features)
- **Development**: Type hints, comprehensive logging, modular design

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- ~2GB free disk space
- [Spotify Dataset](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)

### Installation

```bash
# Clone the repository
git clone https://github.com/nurulgofran/spotify-reinforcement-learning-recommendations-.git
cd spotify-reinforcement-learning-recommendations-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download dataset and place as data/raw/dataset.csv
```

### Usage

```bash
# Run complete pipeline (preprocessing + training)
python main.py

# Or run individual steps
python main.py --preprocess-only
python main.py --train-only
python main.py --evaluate
```

For detailed setup instructions, see [SETUP.md](SETUP.md).

## 📁 Project Structure

```
spotify-reinforcement-learning-recommendations-/
│
├── src/
│   ├── config.py              # Configuration and hyperparameters
│   └── recommender/
│       ├── agent.py           # DQN Agent implementation
│       ├── environment.py     # RL Environment
│       ├── preprocess.py      # Data preprocessing
│       └── train.py           # Training loop and evaluation
│
├── data/
│   ├── raw/                   # Raw dataset
│   └── processed/             # Processed features and track IDs
│
├── models/                    # Trained model checkpoints
├── logs/                      # Training logs
├── main.py                    # Main entry point
├── requirements.txt           # Python dependencies
├── SETUP.md                   # Detailed setup guide
├── CONTRIBUTING.md            # Contribution guidelines
└── README.md                  # This file
```

## � The Reinforcement Learning Approach

### State Space
- **Dimension**: 9 audio features
- **Representation**: Mean of last 5 songs' features
- **Features**: acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence

### Action Space
- **Size**: Number of songs in dataset (~100k)
- **Action**: Select a song index to recommend

### Reward Function
- **Metric**: Cosine similarity between recommended song and current state
- **Baseline**: Centered around 0.5 to encourage improvement
- **Goal**: Maximize similarity to user's current musical taste

### Algorithm Details
- **Algorithm**: Deep Q-Network (DQN)
- **Neural Network**: 2-layer feedforward (128 hidden units each)
- **Experience Replay**: Buffer size 100k, batch size 64
- **Target Network**: Soft updates with τ=0.001
- **Exploration**: ε-greedy with decay (1.0 → 0.01)

## 📊 Results and Performance

The agent typically achieves:
- **Convergence**: ~1000-1500 episodes
- **Target Score**: 15.0 average reward over 100 episodes
- **Training Time**: 30-60 minutes on modern CPU
- **Memory Usage**: ~2-4GB during training

## 🎛️ Configuration

Key hyperparameters in `src/config.py`:

```python
class TrainingConfig:
    N_EPISODES = 2000           # Training episodes
    BATCH_SIZE = 64             # Mini-batch size
    LEARNING_RATE = 5e-4        # Adam learning rate
    GAMMA = 0.99                # Discount factor
    EPSILON_DECAY = 0.995       # Exploration decay
```

## 🔮 Future Enhancements

- **Advanced Algorithms**: Double DQN, Dueling DQN, Rainbow DQN
- **User Personalization**: Multi-user support with user embeddings
- **Real-time Integration**: Web interface for live recommendations
- **Enhanced Features**: Genre, artist, and temporal information
- **Evaluation Metrics**: A/B testing framework with human feedback
- **Scalability**: Distributed training and serving infrastructure

## 🧪 Experimentation

### Testing Individual Components
```bash
# Test agent
python -m src.recommender.agent

# Test environment  
python -m src.recommender.environment

# Test preprocessing
python -m src.recommender.preprocess
```

### Hyperparameter Tuning
Modify values in `src/config.py` and retrain:
```bash
python main.py --train-only
```

### Custom Datasets
Replace `data/raw/dataset.csv` with any Spotify-format dataset containing the required audio features.

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup
```bash
# Install development dependencies
pip install pytest black flake8 mypy

# Run tests
python -m pytest

# Format code
black src/ main.py

# Type checking
mypy src/
```

## � License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Zaheenhamidani's Ultimate Spotify Tracks DB](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
- **Inspiration**: DeepMind's DQN paper and OpenAI's reinforcement learning research
- **Framework**: PyTorch team for the excellent deep learning framework

## 📖 References

- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) - Original DQN paper
- [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) - Nature DQN paper
- [Deep Reinforcement Learning for Recommender Systems](https://arxiv.org/abs/1801.00209) - RL for recommendations

---

**Star ⭐ this repository if you find it helpful!**

For questions or issues, please [open an issue](https://github.com/nurulgofran/spotify-reinforcement-learning-recommendations-/issues) or reach out to the maintainers.