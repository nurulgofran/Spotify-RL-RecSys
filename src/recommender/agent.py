"""
Deep Q-Network (DQN) Agent for Music Recommendation.

This module implements a DQN agent that learns to recommend songs based on user preferences
represented as audio feature vectors. The agent uses experience replay and a target network
for stable learning.

Classes:
    ReplayBuffer: Stores and samples experiences for training
    QNetwork: Neural network architecture for Q-value estimation
    Agent: Main DQN agent implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
from typing import Tuple


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples for experience replay.

    The replay buffer stores transitions and provides random sampling to break
    correlation between consecutive experiences during training.
    """

    def __init__(self, buffer_size: int, batch_size: int) -> None:
        """
        Initialize the replay buffer.

        Args:
            buffer_size: Maximum number of experiences to store
            batch_size: Number of experiences to sample for training
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a new experience to memory.

        Args:
            state: Current state (audio features)
            action: Action taken (song index)
            reward: Reward received
            next_state: Next state after action
            done: Whether episode is finished
        """
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly sample a batch of experiences from memory.

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float()
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long()
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float()
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float()
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.memory)


class QNetwork(nn.Module):
    """
    Deep Q-Network for estimating Q-values.

    A simple feedforward neural network that maps states (audio features)
    to Q-values for all possible actions (songs).
    """

    def __init__(
        self, state_size: int, action_size: int, hidden_size: int = 128
    ) -> None:
        """
        Initialize the Q-Network.

        Args:
            state_size: Dimension of each state (number of audio features)
            action_size: Dimension of each action (number of songs in dataset)
            hidden_size: Number of nodes in hidden layers
        """
        super(QNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            state: Input state tensor

        Returns:
            Q-values for all actions
        """
        return self.layers(state)


class Agent:
    """
    Deep Q-Network (DQN) agent for music recommendation.

    Implements the DQN algorithm with experience replay and target networks
    to learn optimal song recommendations based on user listening patterns.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        buffer_size: int = int(1e5),
        batch_size: int = 64,
        gamma: float = 0.99,
        tau: float = 1e-3,
        lr: float = 5e-4,
        device: str = "cpu",
    ) -> None:
        """
        Initialize the DQN Agent.

        Args:
            state_size: Dimension of each state (audio features)
            action_size: Dimension of each action (number of songs)
            buffer_size: Replay buffer size
            batch_size: Minibatch size for training
            gamma: Discount factor for future rewards
            tau: Soft update parameter for target network
            lr: Learning rate for optimizer
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.device = device

        # Q-Networks
        self.q_network = QNetwork(state_size, action_size).to(device)
        self.target_q_network = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay memory
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Epsilon-greedy parameters
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state: np.ndarray, add_noise: bool = True) -> int:
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state: Current state (audio features)
            add_noise: Whether to use epsilon-greedy exploration

        Returns:
            Action index (song to recommend)
        """
        if add_noise and random.random() <= self.epsilon:
            return random.choice(np.arange(self.action_size))

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.q_network.eval()
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        self.q_network.train()

        return np.argmax(action_values.cpu().data.numpy())

    def step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Save experience in replay memory and trigger learning if enough samples are available.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences: Tuple[torch.Tensor, ...]) -> None:
        """
        Update value parameters using given batch of experience tuples.

        Args:
            experiences: Tuple of (s, a, r, s', done) tensors
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values for next states from target model
        q_targets_next = (
            self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        )

        # Compute Q targets for current states
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.q_network(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(q_expected, q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.soft_update(self.q_network, self.target_q_network)

    def soft_update(self, local_model: nn.Module, target_model: nn.Module) -> None:
        """
        Soft update model parameters using Polyak averaging.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
        """
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        torch.save(
            {
                "q_network_state_dict": self.q_network.state_dict(),
                "target_q_network_state_dict": self.target_q_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
            },
            filepath,
        )

    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]


def main() -> None:
    """Test the Agent implementation with dummy data."""
    print("Testing DQN Agent...")

    # Test parameters
    STATE_SIZE_TEST = 9
    ACTION_SIZE_TEST = 1000
    BATCH_SIZE_TEST = 10

    # Initialize agent
    agent = Agent(
        state_size=STATE_SIZE_TEST,
        action_size=ACTION_SIZE_TEST,
        batch_size=BATCH_SIZE_TEST,
    )

    # Add some experiences to the replay buffer
    for _ in range(BATCH_SIZE_TEST + 5):
        dummy_state = np.random.rand(STATE_SIZE_TEST)
        dummy_action = np.random.randint(ACTION_SIZE_TEST)
        dummy_reward = np.random.rand()
        dummy_next_state = np.random.rand(STATE_SIZE_TEST)
        dummy_done = random.choice([True, False])

        agent.step(
            dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done
        )

    # Test action selection
    test_state = np.random.rand(STATE_SIZE_TEST)
    action = agent.act(test_state)
    assert 0 <= action < ACTION_SIZE_TEST, f"Invalid action: {action}"

    print("✅ Agent test passed")
    print(f"   - Buffer size: {len(agent.memory)}")
    print(f"   - Epsilon: {agent.epsilon:.3f}")
    print(f"   - Sample action: {action}")


if __name__ == "__main__":
    main()
