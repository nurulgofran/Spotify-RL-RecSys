# Spotify RL Recommendation System

## Bachelor's Thesis Project

**Deep Reinforcement Learning Based Music Recommendation System**

Gujarat Technological University (GTU), 2020

### About

This is my bachelor's thesis project where I built a music recommendation system using
reinforcement learning. The idea is to treat song recommendation as a sequential
decision-making problem and use a Deep Q-Network (DQN) to learn what songs to recommend
based on a user's listening history.

Instead of using traditional approaches like collaborative filtering, this system learns
to adapt its recommendations over time by observing how similar each recommended song is
to what the user has been listening to recently.

### How it Works

1. **Data**: Uses Spotify audio features (danceability, energy, tempo, etc.) from the
   Kaggle Ultimate Spotify Tracks DB
2. **Environment**: Simulates a listening session where the agent recommends songs and
   gets feedback based on how well they match the user's current taste
3. **Agent**: A DQN agent that learns which songs to recommend using experience replay
   and epsilon-greedy exploration
4. **Reward**: Based on cosine similarity between the recommended song's features and
   the user's recent listening history

### Project Structure

```
src/
├── config.py                 # Configuration and hyperparameters
└── recommender/
    ├── agent.py              # DQN agent (Q-Network, replay buffer, agent)
    ├── data_loader.py        # Basic data loading utility
    ├── environment.py        # RL environment for song recommendation
    ├── preprocess.py         # Data preprocessing pipeline
    └── train.py              # Training and evaluation
main.py                       # Entry point with CLI
```

### Setup

1. Clone the repo
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
   and place it at `data/raw/dataset.csv`

### Usage

```bash
# Run full pipeline (preprocess + train)
python main.py

# Preprocess only
python main.py --preprocess-only

# Train only (requires preprocessed data)
python main.py --train-only

# Evaluate a trained model
python main.py --evaluate
```

### Tech Stack

- Python 3.8+
- PyTorch (Deep Q-Network)
- pandas, NumPy, scikit-learn (data processing)
- matplotlib (visualization)

### Thesis Details

- **Student**: Md Norul Gofran
- **Enrollment No.**: 160470107067
- **Supervisors**: Prof. Viraj Daxini, Dr. Tejas Patalia
- **Department**: Computer Engineering, GTU

### References

1. Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
2. Mnih et al., "Human-level control through deep reinforcement learning", Nature (2015)
3. Zheng et al., "DRN: A Deep Reinforcement Learning Framework for News Recommendation"

### License

MIT License - see [LICENSE](LICENSE) for details.
