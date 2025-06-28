# src/recommender/agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): Maximum size of buffer.
            batch_size (int): Size of each training batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # Convert the batch of experiences to PyTorch tensors
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

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class QNetwork(nn.Module):
    """
    The neural network that will act as the agent's brain.
    It learns to map a state to the Q-values of all possible actions.
    """

    def __init__(self, state_size, action_size):
        """
        Initializes the network layers.

        Args:
            state_size (int): The number of features in the state vector (e.g., 9).
            action_size (int): The number of possible actions (total number of songs).
        """
        super(QNetwork, self).__init__()

        print("Initializing Q-Network...")

        # Define the network architecture
        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),  # Input layer
            nn.ReLU(),  # Activation function
            nn.Linear(128, 128),  # Hidden layer
            nn.ReLU(),  # Activation function
            nn.Linear(128, action_size),  # Output layer
        )

        print("✅ Q-Network initialized successfully.")
        print(f"   - Input size: {state_size}")
        print(f"   - Output size: {action_size}")

    def forward(self, state):
        """
        Defines the forward pass of the network.
        It takes a state and returns the Q-values for all actions.
        """
        return self.layers(state)


class Agent:
    """
    The RL Agent that interacts with and learns from the environment.
    """

    def __init__(
        self,
        state_size,
        action_size,
        buffer_size=int(1e5),
        batch_size=64,
        gamma=0.99,
        tau=1e-3,
        lr=5e-4,
    ):
        """
        Initializes the Agent object.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            buffer_size (int): Maximum size of replay buffer.
            batch_size (int): Size of each training batch.
            gamma (float): Discount factor.
            tau (float): For soft update of target network.
            lr (float): Learning rate.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor
        self.tau = tau  # For soft update of target network

        # Q-Network (The "brain" that gets trained)
        self.q_network = QNetwork(state_size, action_size)
        # Target Network (A stable copy for calculating targets)
        self.target_q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Initialize replay buffer
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Epsilon-greedy action selection parameters
        self.epsilon = 1.0  # Starting value of epsilon
        self.epsilon_min = 0.01  # Minimum value of epsilon
        self.epsilon_decay = 0.995  # Rate at which epsilon decays after each episode

    def act(self, state):
        """
        Returns an action for a given state using the epsilon-greedy policy.

        Args:
            state (np.ndarray): The current state from the environment.

        Returns:
            int: The index of the action (song) to take.
        """
        # --- Epsilon-Greedy Action Selection ---
        if random.random() > self.epsilon:
            # Exploit: Choose the best action from the Q-network
            state = (
                torch.from_numpy(state).float().unsqueeze(0)
            )  # Convert state to PyTorch tensor

            # Set the network to evaluation mode (important for inference)
            self.q_network.eval()
            with (
                torch.no_grad()
            ):  # We don't need to calculate gradients for action selection
                action_values = self.q_network(state)
            # Set it back to training mode
            self.q_network.train()

            # Choose the action with the highest Q-value
            return np.argmax(action_values.cpu().data.numpy())
        else:
            # Explore: Choose a random action
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # 1. Save the experience
        self.memory.add(state, action, reward, next_state, done)

        # 2. Learn if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        """
        Update value parameters using given batch of experience tuples.
        """
        states, actions, rewards, next_states, dones = experiences

        # 1. Get max predicted Q-values for next states from the target network
        Q_targets_next = (
            self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        )

        # 2. Compute Q targets for current states (Bellman equation)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # 3. Get expected Q values from the local network
        Q_expected = self.q_network(states).gather(1, actions)

        # 4. Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # 5. Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 6. Update target network (soft update)
        self.soft_update(self.q_network, self.target_q_network)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters: θ_target = τ*θ_local + (1 - τ)*θ_target"""
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


# --- Update the Test Block ---
if __name__ == "__main__":
    print("--- Testing Agent Learning ---")

    STATE_SIZE_TEST = 9
    ACTION_SIZE_TEST = 1000  # Using a smaller action space for a faster test
    BATCH_SIZE_TEST = 10

    # 1. Instantiate the agent
    agent = Agent(
        state_size=STATE_SIZE_TEST,
        action_size=ACTION_SIZE_TEST,
        batch_size=BATCH_SIZE_TEST,
    )
    print("✅ Agent initialized.")

    # 2. Add enough dummy experiences to the buffer to trigger learning
    print("Populating buffer with dummy experiences...")
    for _ in range(BATCH_SIZE_TEST + 5):
        dummy_state = np.random.rand(STATE_SIZE_TEST)
        dummy_action = np.random.randint(ACTION_SIZE_TEST)
        dummy_reward = np.random.rand()
        dummy_next_state = np.random.rand(STATE_SIZE_TEST)
        dummy_done = random.choice([True, False])
        # Use the new step method
        agent.step(
            dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done
        )

    print(f"Buffer size is now {len(agent.memory)}.")
    print("✅ Test passed: Agent can store experiences and trigger learning.")
