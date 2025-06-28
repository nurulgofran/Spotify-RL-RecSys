# src/recommender/agent.py

import torch
import torch.nn as nn
import torch.optim as optim


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


# --- Test Block ---
# This allows us to test the file directly.
if __name__ == "__main__":
    print("--- Testing QNetwork ---")

    # Define dummy sizes for testing purposes
    STATE_SIZE_TEST = 9  # Our state has 9 features
    ACTION_SIZE_TEST = 114000  # Our dataset has ~114k songs

    # Create the network
    net = QNetwork(state_size=STATE_SIZE_TEST, action_size=ACTION_SIZE_TEST)
    print("\nNetwork Architecture:")
    print(net)

    # Create a dummy state tensor to feed into the network
    # The state should be a PyTorch tensor
    dummy_state = torch.randn(STATE_SIZE_TEST)
    print(f"\nShape of dummy state: {dummy_state.shape}")

    # Get the output from the network
    # We must add a "batch dimension" using unsqueeze(0) for the network to accept it
    q_values = net(dummy_state.unsqueeze(0))

    print(f"Shape of output Q-values: {q_values.shape}")

    # Verify that the output shape is correct
    assert q_values.shape == (1, ACTION_SIZE_TEST)

    print("\n✅ Test passed: Network produces output with the correct shape.")
