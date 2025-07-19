#!/usr/bin/env python3
"""
Quick verification script for the Spotify RL Recommender.

This script tests the key components to ensure everything is working correctly
without requiring the full dataset or long training times.
"""

import sys
import numpy as np
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")

    try:
        # Test imports by importing and checking they exist
        import src.config
        import src.recommender.agent
        import src.recommender.environment
        import src.recommender.preprocess
        import src.recommender.train

        # Check key classes exist
        assert hasattr(src.config, "TrainingConfig")
        assert hasattr(src.config, "AUDIO_FEATURES")
        assert hasattr(src.recommender.agent, "Agent")
        assert hasattr(src.recommender.agent, "QNetwork")
        assert hasattr(src.recommender.agent, "ReplayBuffer")

        logger.info("✅ All imports successful")
        return True
    except (ImportError, AssertionError) as e:
        logger.error(f"❌ Import failed: {e}")
        return False


def test_config():
    """Test configuration values."""
    logger.info("Testing configuration...")

    try:
        from src.config import TrainingConfig, AUDIO_FEATURES

        assert len(AUDIO_FEATURES) == 9, (
            f"Expected 9 audio features, got {len(AUDIO_FEATURES)}"
        )
        assert TrainingConfig.N_EPISODES > 0, "N_EPISODES must be positive"
        assert 0 < TrainingConfig.GAMMA <= 1, "GAMMA must be in (0, 1]"
        assert TrainingConfig.BATCH_SIZE > 0, "BATCH_SIZE must be positive"

        logger.info("✅ Configuration validation passed")
        return True
    except (AssertionError, ImportError) as e:
        logger.error(f"❌ Configuration test failed: {e}")
        return False


def test_agent():
    """Test agent functionality with dummy data."""
    logger.info("Testing agent...")

    try:
        from src.recommender.agent import Agent

        state_size = 9
        action_size = 100
        batch_size = 10

        agent = Agent(
            state_size=state_size,
            action_size=action_size,
            batch_size=batch_size,
        )

        # Test action selection
        dummy_state = np.random.rand(state_size)
        action = agent.act(dummy_state)
        assert 0 <= action < action_size, f"Invalid action: {action}"

        # Test experience storage and learning
        for _ in range(batch_size + 2):
            state = np.random.rand(state_size)
            action = np.random.randint(action_size)
            reward = np.random.rand()
            next_state = np.random.rand(state_size)
            done = np.random.choice([True, False])

            agent.step(state, action, reward, next_state, done)

        logger.info("✅ Agent test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Agent test failed: {e}")
        return False


def test_network():
    """Test neural network forward pass."""
    logger.info("Testing neural network...")

    try:
        import torch
        from src.recommender.agent import QNetwork

        state_size = 9
        action_size = 100
        batch_size = 5

        network = QNetwork(state_size, action_size)

        # Test forward pass
        dummy_input = torch.randn(batch_size, state_size)
        output = network(dummy_input)

        assert output.shape == (batch_size, action_size), (
            f"Invalid output shape: {output.shape}"
        )
        assert not torch.isnan(output).any(), "Network output contains NaN values"

        logger.info("✅ Neural network test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Neural network test failed: {e}")
        return False


def test_replay_buffer():
    """Test replay buffer functionality."""
    logger.info("Testing replay buffer...")

    try:
        from src.recommender.agent import ReplayBuffer

        buffer_size = 100
        batch_size = 10
        state_size = 9

        buffer = ReplayBuffer(buffer_size, batch_size)

        # Add experiences
        for i in range(batch_size + 5):
            state = np.random.rand(state_size)
            action = i % 10
            reward = np.random.rand()
            next_state = np.random.rand(state_size)
            done = i % 3 == 0

            buffer.add(state, action, reward, next_state, done)

        assert len(buffer) == batch_size + 5, f"Invalid buffer length: {len(buffer)}"

        # Test sampling
        experiences = buffer.sample()
        states, actions, rewards, next_states, dones = experiences

        assert states.shape[0] == batch_size, (
            f"Invalid states batch size: {states.shape[0]}"
        )
        assert actions.shape[0] == batch_size, (
            f"Invalid actions batch size: {actions.shape[0]}"
        )

        logger.info("✅ Replay buffer test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Replay buffer test failed: {e}")
        return False


def test_mock_environment():
    """Test environment with mock data."""
    logger.info("Testing mock environment...")

    try:
        import numpy as np
        from collections import deque

        # Create a minimal mock environment
        class MockEnvironment:
            def __init__(self, n_songs=1000, n_features=9, history_length=5):
                self.song_features = np.random.rand(n_songs, n_features)
                self.action_space_size = n_songs
                self.state_space_size = n_features
                self.history_length = history_length
                self.song_history = deque(maxlen=history_length)
                self.episode_step_counter = 0

            def reset(self):
                self.song_history.clear()
                for _ in range(self.history_length):
                    idx = np.random.randint(0, self.action_space_size)
                    self.song_history.append(self.song_features[idx])
                self.episode_step_counter = 0
                return np.mean(list(self.song_history), axis=0)

            def step(self, action_index):
                self.episode_step_counter += 1
                reward = np.random.rand() - 0.5  # Centered around 0
                self.song_history.append(self.song_features[action_index])
                next_state = np.mean(list(self.song_history), axis=0)
                done = self.episode_step_counter >= 10
                return next_state, reward, done

        env = MockEnvironment()
        state = env.reset()

        assert state.shape == (env.state_space_size,), (
            f"Invalid state shape: {state.shape}"
        )

        action = np.random.randint(0, env.action_space_size)
        next_state, reward, done = env.step(action)

        assert next_state.shape == (env.state_space_size,), (
            f"Invalid next state shape: {next_state.shape}"
        )
        assert isinstance(reward, (int, float)), f"Invalid reward type: {type(reward)}"
        assert isinstance(done, bool), f"Invalid done type: {type(done)}"

        logger.info("✅ Mock environment test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Mock environment test failed: {e}")
        return False


def test_mini_training():
    """Test a mini training loop with mock data."""
    logger.info("Testing mini training loop...")

    try:
        from collections import deque
        from src.recommender.agent import Agent

        # Mock environment (same as above)
        class MockEnvironment:
            def __init__(self, n_songs=100, n_features=9, history_length=5):
                self.song_features = np.random.rand(n_songs, n_features)
                self.action_space_size = n_songs
                self.state_space_size = n_features
                self.history_length = history_length
                self.song_history = deque(maxlen=history_length)
                self.episode_step_counter = 0

            def reset(self):
                self.song_history.clear()
                for _ in range(self.history_length):
                    idx = np.random.randint(0, self.action_space_size)
                    self.song_history.append(self.song_features[idx])
                self.episode_step_counter = 0
                return np.mean(list(self.song_history), axis=0)

            def step(self, action_index):
                self.episode_step_counter += 1
                reward = np.random.rand() - 0.5
                self.song_history.append(self.song_features[action_index])
                next_state = np.mean(list(self.song_history), axis=0)
                done = self.episode_step_counter >= 5
                return next_state, reward, done

        env = MockEnvironment()
        agent = Agent(
            state_size=env.state_space_size,
            action_size=env.action_space_size,
            batch_size=8,
        )

        # Run a few episodes
        for episode in range(3):
            state = env.reset()
            episode_reward = 0

            for step in range(10):
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                agent.step(state, action, reward, next_state, done)

                state = next_state
                episode_reward += reward

                if done:
                    break

        logger.info("✅ Mini training loop test passed")
        return True
    except Exception as e:
        logger.error(f"❌ Mini training loop test failed: {e}")
        return False


def main():
    """Run all verification tests."""
    logger.info("🧪 Running Spotify RL Recommender verification tests...")
    logger.info("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Agent", test_agent),
        ("Neural Network", test_network),
        ("Replay Buffer", test_replay_buffer),
        ("Mock Environment", test_mock_environment),
        ("Mini Training", test_mini_training),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        if test_func():
            passed += 1
        else:
            logger.error(f"Test '{test_name}' failed!")

    logger.info("\n" + "=" * 60)
    logger.info(f"Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! Your setup is working correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Download the Spotify dataset to data/raw/dataset.csv")
        logger.info("2. Run: python main.py")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
