import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
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

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

    def forward(self, state):
        return self.layers(state)


class Agent:
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
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.q_network = QNetwork(state_size, action_size)
        self.target_q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if random.random() > self.epsilon:
            state = torch.from_numpy(state).float().unsqueeze(0)

            self.q_network.eval()
            with torch.no_grad():
                action_values = self.q_network(state)
            self.q_network.train()

            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        Q_targets_next = (
            self.target_q_network(next_states).detach().max(1)[0].unsqueeze(1)
        )

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected = self.q_network(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.q_network, self.target_q_network)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )


if __name__ == "__main__":
    STATE_SIZE_TEST = 9
    ACTION_SIZE_TEST = 1000
    BATCH_SIZE_TEST = 10

    agent = Agent(
        state_size=STATE_SIZE_TEST,
        action_size=ACTION_SIZE_TEST,
        batch_size=BATCH_SIZE_TEST,
    )

    for _ in range(BATCH_SIZE_TEST + 5):
        dummy_state = np.random.rand(STATE_SIZE_TEST)
        dummy_action = np.random.randint(ACTION_SIZE_TEST)
        dummy_reward = np.random.rand()
        dummy_next_state = np.random.rand(STATE_SIZE_TEST)
        dummy_done = random.choice([True, False])
        agent.step(
            dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done
        )

    print("✅ Agent test passed")
