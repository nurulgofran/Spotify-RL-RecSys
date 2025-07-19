# Contributing to Spotify Reinforcement Learning Recommender

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Spotify Reinforcement Learning Music Recommender.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nurulgofran/spotify-reinforcement-learning-recommendations-.git
   cd spotify-reinforcement-learning-recommendations-
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download from [Kaggle](https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db)
   - Place as `data/raw/dataset.csv`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for function parameters and return values
- Write docstrings for all classes and functions
- Use meaningful variable and function names
- Keep functions focused and small

## Testing

Run tests before submitting any changes:

```bash
# Test individual modules
python -m src.recommender.agent
python -m src.recommender.environment
python -m src.recommender.preprocess

# Run the full pipeline
python main.py --preprocess-only
python main.py --train-only
```

## Submitting Changes

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following the code style guidelines
4. **Test your changes** thoroughly
5. **Commit your changes** with clear, descriptive messages
   ```bash
   git commit -m "Add feature: clear description of what you added"
   ```
6. **Push to your fork** and submit a pull request

## Pull Request Guidelines

- Provide a clear description of the changes
- Include any relevant issue numbers
- Ensure all tests pass
- Update documentation if necessary
- Keep changes focused and atomic

## Issue Reporting

When reporting issues, please include:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Error messages and stack traces

## Feature Requests

For feature requests, please describe:

- The use case or problem being solved
- Proposed solution or approach
- Any alternatives considered
- Potential impact on existing functionality

## Code Review Process

All submissions require review. Please be patient and responsive to feedback. The maintainers will:

- Review code for correctness and style
- Suggest improvements where appropriate
- Merge approved changes

## Questions?

Feel free to open an issue for questions about contributing or reach out to the maintainers.

Thank you for contributing!
