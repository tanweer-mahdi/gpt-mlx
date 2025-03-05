# GPT-MLX: Decoder-Only Transformer Implementation

A PyTorch-style implementation of a decoder-only GPT model using Apple's MLX framework, designed to stress test M4 series GPU cores through transformer-based language model training.

## Overview

This project implements a decoder-only transformer architecture (GPT-style) using MLX, Apple's machine learning framework optimized for Apple Silicon. The implementation includes:

- Character-level tokenization
- Multi-head self-attention
- Position embeddings
- Configurable model architecture
- Training and inference pipelines

## Project Structure

```
gpt-mlx/
├── src/
│   ├── datasets/           # Dataset handling and loading
│   ├── tokenizer/         # Character-level tokenizer implementation
│   ├── utils/            # Utility functions
│   ├── train_gpt.py      # Main training script for GPT
│   ├── train_bigram.py   # Bigram model implementation
│   ├── train_config.json # Model and training configuration
│   └── scratchpad.py     # Experimental code and testing
├── tests/                # Test suite
└── setup.py             # Package setup file
```

## Configuration

Model and training parameters can be configured through `train_config.json`:

```json
{
    "data": {
        "dataset_path": "datasets/tiny_shakespeare.txt",
        "train_split": 0.8
    },
    "model": {
        "context_window": 256,
        "embedding_dim": 128,
        "batch_size": 32,
        "num_heads": 8,
        "num_blocks": 3,
        "dropout": 0.1
    },
    "training": {
        "n_eval_steps": 200,
        "n_train_steps": 10000,
        "eval_interval": 500,
        "learning_rate": 0.001
    }
}
```

## Setup

### Using uv (Recommended)

This project uses `uv` for package management, which provides faster dependency resolution and installation:

1. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
```

3. Install dependencies:
```bash
uv pip install -e .
```

### Alternative Installation Methods

#### Using pip with pyproject.toml

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -e .
```

#### Using Poetry

1. Install Poetry:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies:
```bash
poetry install
```


## License

This project is licensed under the MIT License.
