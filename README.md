# Audio Caption Framework

A modular framework for audio-to-text captioning with easy model swapping capabilities.

## Features

- **Plugin Architecture**: Easily swap audio encoders, language models, and projection layers
- **Model Registry**: Simple registration system for adding new models
- **Configuration-Based**: YAML configuration files for easy experimentation
- **Modular Design**: Clean separation of concerns

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

1. Prepare your dataset (should be saved with `datasets.save_to_disk()`)

2. Create a configuration file (see `configs/default.yaml` for example)

3. Train the model:

```bash
# With GPT-2
python scripts/train.py \
    --config configs/default.yaml \
    --dataset path/to/your/dataset

# With T5
python scripts/train.py \
    --config configs/t5.yaml \
    --dataset path/to/your/dataset
```

## Project Structure

```
audiocaption/
├── core/              # Base classes and registry
├── models/            # Model implementations
│   ├── audio_encoders/
│   ├── language_models/
│   └── projections/
├── data/              # Dataset loading
├── training/          # Training pipeline
└── utils/             # Utilities

scripts/
└── train.py           # Training script

configs/
└── default.yaml       # Example configuration
```

## Adding New Models

### Adding a New Audio Encoder

```python
from audiocaption.core.base import AudioEncoder
from audiocaption.core.registry import ModelRegistry

@ModelRegistry.register_audio_encoder("my_encoder")
class MyAudioEncoder(AudioEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
    
    def get_embedding_dim(self):
        return 512  # Your embedding dimension
    
    def encode_audio(self, audio):
        # Your encoding logic
        return embeddings
```

### Adding a New Language Model

```python
from audiocaption.core.base import LanguageModel
from audiocaption.core.registry import ModelRegistry

@ModelRegistry.register_language_model("my_lm")
class MyLanguageModel(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
    
    def get_embedding_dim(self):
        return 768  # Your embedding dimension
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def forward_with_prefix(self, prefix_embeddings, text_embeddings, labels=None):
        # Your forward logic
        return {"loss": loss}
```

### Adding a New Projection

```python
from audiocaption.core.base import Projection
from audiocaption.core.registry import ModelRegistry

@ModelRegistry.register_projection("my_projection")
class MyProjection(Projection):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your projection
    
    def forward(self, audio_embeddings):
        # Your projection logic
        return projected_embeddings
    
    def get_output_shape(self, prefix_len, lm_embedding_dim):
        return (prefix_len, lm_embedding_dim)
```

## Configuration

Configuration files are YAML format. See `configs/default.yaml` for an example.

Key sections:
- `audio_encoder`: Audio encoder configuration
- `language_model`: Language model configuration  
- `projection`: Projection layer configuration
- `training`: Training hyperparameters

## Available Models

### Audio Encoders
- `clap`: CLAP (Contrastive Language-Audio Pretraining)

### Language Models
- `gpt2`: GPT-2
- `t5`: T5 (Text-to-Text Transfer Transformer) - supports t5-small, t5-base, t5-large, etc.

### Projections
- `sequential`: Sequential layers with configurable architecture

## License

MIT
