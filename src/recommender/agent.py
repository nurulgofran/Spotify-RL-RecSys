# src/recommender/agent.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random


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

    def __init__(self, state_size, action_size):
        """
        Initializes the Agent object.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
        """
        self.state_size = state_size
        self.action_size = action_size

        # 1. Create the Q-Network
        # This is the "brain" that the agent will train.
        self.q_network = QNetwork(state_size, action_size)

        # 2. Define the Optimizer
        # Adam is a popular choice for optimizing the network's weights.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-4)

        # 3. Epsilon-greedy action selection parameters
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


# --- Update the Test Block ---
if __name__ == "__main__":
    print("--- Testing Agent ---")

    STATE_SIZE_TEST = 9
    ACTION_SIZE_TEST = 114000

    # 1. Instantiate the agent
    agent = Agent(state_size=STATE_SIZE_TEST, action_size=ACTION_SIZE_TEST)
    print("✅ Agent initialized successfully.")

    # 2. Create a dummy state from the environment
    dummy_state = np.random.rand(STATE_SIZE_TEST)

    # 3. Test the act() method
    print("\nTesting act() method...")

    # --- Test Exploration (epsilon = 1.0) ---
    agent.epsilon = 1.0
    print(f"Epsilon is {agent.epsilon}, so action should be random.")
    action = agent.act(dummy_state)
    print(f"Agent chose action (song index): {action}")
    assert isinstance(action, int) or isinstance(action, np.int64)

    # --- Test Exploitation (epsilon = 0.0) ---
    agent.epsilon = 0.0
    print(f"\nEpsilon is {agent.epsilon}, so action should be from the network.")
    action = agent.act(dummy_state)
    print(f"Agent chose action (song index): {action}")
    assert isinstance(action, int) or isinstance(action, np.int64)

    print("\n✅ Test passed: Agent can be created and can choose an action.")
