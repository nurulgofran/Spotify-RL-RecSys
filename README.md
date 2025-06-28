# A Deep Reinforcement Learning Music Recommender

![Project Banner](https://i.imgur.com/a2e3F1F.png)

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Built with: scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

This is a cutting-edge music recommendation system that learns a user's taste in real-time. Instead of traditional methods, this project uses **Deep Reinforcement Learning (DRL)** to build an intelligent agent that recommends songs. The agent learns from its decisions, continuously refining its policy to create the perfect personalized playlist.

This repository contains the full end-to-end implementation, from raw data processing to training a sophisticated DQN agent, using a public Spotify dataset.

## ✨ Key Features

- **Dynamic Learning:** The agent adapts its recommendations based on the user's immediate listening history, capturing their current "vibe."
- **DRL-Powered Engine:** Built on a Deep Q-Network (DQN), a state-of-the-art algorithm from the field of reinforcement learning.
- **Sophisticated State Representation:** The agent's understanding of the user's taste is based on an averaged history of song audio features, creating a rich and nuanced state.
- **Modular & Scalable Codebase:** The project is structured professionally with clear separation of concerns, making it easy to understand, maintain, and extend.
- **Offline Training:** The entire system is trained offline using a large public dataset, simulating thousands of user sessions.

## 🤖 The Reinforcement Learning Approach

We frame the recommendation task as a Reinforcement Learning problem, where our agent learns to interact with a simulated user environment.

| Component     | Definition                                                                                             |
| :------------ | :----------------------------------------------------------------------------------------------------- |
| **Agent** | A Deep Q-Network (DQN) that learns a policy to recommend songs.                                        |
| **Environment** | A simulated user listening session built from the Spotify dataset.                                     |
| **State** | The **average audio features** (`danceability`, `energy`, etc.) of the last 5 songs the user has heard.  |
| **Action** | Recommending a single song from the entire dataset.                                                    |
| **Reward** | The **cosine similarity** between the recommended song's features and the current state (the user's vibe), centered around 0 to encourage meaningful improvement. |

## 🛠️ Tech Stack

- **Core Libraries:** Python 3.9+, NumPy, Pandas
- **Machine Learning:** PyTorch, scikit-learn
- **Data:** A public [Spotify dataset of 100k+ songs](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db).
- **Development:** VS Code, Git, GitHub

## 📁 Project Structure

The repository is organized to be clean and scalable:

```
spotify-rl-recommender/
│
├── data/
│   ├── raw/            # Raw dataset.csv
│   └── processed/      # Processed song features and track IDs
│
├── src/
│   └── recommender/
│       ├── agent.py          # The DQN Agent and Replay Buffer
│       ├── environment.py    # The custom RL environment
│       ├── preprocess.py     # Data preprocessing and shuffling script
│       └── train.py          # The main training loop
│
├── .gitignore
├── main.py             # Main entry point to run the pipeline
├── README.md
└── requirements.txt
```

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.9 or higher
- Git for version control

### 2. Clone the Repository

```bash
git clone [https://github.com/your-username/spotify-rl-recommender.git](https://github.com/your-username/spotify-rl-recommender.git)
cd spotify-rl-recommender
```

### 3. Set Up a Virtual Environment

It's highly recommended to use a virtual environment.

```bash
# Create the virtual environment
python -m venv venv

# Activate it
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 4. Install Dependencies

Install all the necessary libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Download the Data

- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db).
- Rename the file to `dataset.csv`.
- Place it inside the `data/raw/` directory.

## ⚙️ Usage

The project pipeline is run from `main.py`. It handles both data preprocessing and agent training.

**To run the entire pipeline:**

```bash
python main.py
```

This will:
1.  **Preprocess the data:** Shuffle the dataset, select audio features, normalize them, and save the processed files in `data/processed/`.
2.  **Train the agent:** Launch the training loop for 2000 episodes. Progress will be printed to the console.
3.  **Save the model:** Once training is complete, the learned model weights will be saved as `q_network_checkpoint.pth` in the root directory.

## 🔮 Future Work

This project provides a solid foundation. Here are some ideas for future improvements:

- **Hyperparameter Tuning:** Experiment with different learning rates, network architectures, and `epsilon` decay schedules.
- **Advanced State Representation:** Incorporate user listening history over a longer term or include genre information in the state.
- **Evaluation Module:** Build a script to load the trained model and evaluate its recommendation quality on a hold-out test set.
- **Web Interface:** Create a simple web application using Flask or Streamlit to interact with the trained agent in real-time.
- **Explore Other RL Algorithms:** Implement more advanced algorithms like Double DQN or Dueling DQN to compare performance.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.