# Quick Start Guide

## Framework Overview

This framework provides a plugin-based architecture for audio captioning. You can easily swap:
- **Audio Encoders**: CLAP, or add your own
- **Language Models**: GPT-2, or add your own  
- **Projection Layers**: Sequential, or add your own

## Installation

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Prepare Your Dataset

Your dataset should be saved using `datasets.save_to_disk()`:

```python
from datasets import load_dataset, save_to_disk

dataset = load_dataset("your_dataset")
save_to_disk(dataset, "path/to/save")
```

### 2. Create Configuration

Copy `configs/default.yaml` and modify as needed:

```yaml
audio_encoder:
  name: clap
  model_name: laion/clap-htsat-unfused
  freeze: true

language_model:
  name: gpt2  # Options: gpt2, t5
  model_name: gpt2  # For T5: t5-small, t5-base, t5-large, etc.
  freeze: true

projection:
  name: sequential
  prefix_len: 8
  layers:
    - type: linear
      out_features: 3072
    - type: gelu
    - type: dropout
      p: 0.3
    - type: linear
      out_features: 6144

training:
  optimizer:
    type: AdamW
    lr: 5e-4
    weight_decay: 0.1
  checkpoint_dir: checkpoints

epochs: 30
prefix_len: 8
```

### 3. Train

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

## Adding New Models

### Example: Adding a New Audio Encoder

```python
# audiocaption/models/audio_encoders/my_encoder.py
from audiocaption.core.base import AudioEncoder
from audiocaption.core.registry import ModelRegistry

@ModelRegistry.register_audio_encoder("my_encoder")
class MyAudioEncoder(AudioEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model here
        self.model = YourModel()
        if self.freeze:
            self.freeze_weights()
    
    def get_embedding_dim(self):
        return 512  # Your embedding dimension
    
    def encode_audio(self, audio):
        # audio is a dict with 'array' and 'sampling_rate'
        audio_array = audio["array"]
        # Your encoding logic
        embeddings = self.model.encode(audio_array)
        return embeddings
```

Then import it in `audiocaption/models/__init__.py`:

```python
from .audio_encoders.my_encoder import MyAudioEncoder
```

### Example: Adding a New Language Model

```python
# audiocaption/models/language_models/my_lm.py
from audiocaption.core.base import LanguageModel
from audiocaption.core.registry import ModelRegistry

@ModelRegistry.register_language_model("my_lm")
class MyLanguageModel(LanguageModel):
    def __init__(self, config):
        super().__init__(config)
        # Initialize your model
        self.model = YourLanguageModel()
        self.tokenizer = YourTokenizer()
        if self.freeze:
            self.freeze_weights()
    
    def get_embedding_dim(self):
        return 768
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def forward_with_prefix(self, prefix_embeddings, text_embeddings, labels=None):
        # Concatenate prefix and text
        inputs = torch.cat([prefix_embeddings, text_embeddings], dim=1)
        # Forward pass
        outputs = self.model(inputs, labels=labels)
        return {"loss": outputs.loss} if labels else {}
```

### Example: Adding a New Projection

```python
# audiocaption/models/projections/my_projection.py
from audiocaption.core.base import Projection
from audiocaption.core.registry import ModelRegistry
import torch.nn as nn

@ModelRegistry.register_projection("my_projection")
class MyProjection(Projection):
    def __init__(self, config):
        super().__init__(config)
        audio_dim = config["audio_embedding_dim"]
        lm_dim = config["language_model_embedding_dim"]
        prefix_len = config["prefix_len"]
        
        # Your custom architecture
        self.network = nn.Sequential(
            nn.Linear(audio_dim, lm_dim * 2),
            nn.ReLU(),
            nn.Linear(lm_dim * 2, prefix_len * lm_dim)
        ).to(self.device)
    
    def forward(self, audio_embeddings):
        return self.network(audio_embeddings)
    
    def get_output_shape(self, prefix_len, lm_embedding_dim):
        return (prefix_len, lm_embedding_dim)
```

## Programmatic Usage

```python
import audiocaption.models  # Import to register models
from audiocaption.core.registry import ModelRegistry
from audiocaption.core.config import Config

# Load config
config = Config.from_yaml("configs/default.yaml")
config_dict = config.to_dict()

# Create models
audio_encoder = ModelRegistry.get_audio_encoder(
    "clap", 
    config_dict["audio_encoder"]
)

language_model = ModelRegistry.get_language_model(
    "gpt2",
    config_dict["language_model"]
)

# Get dimensions and create projection
audio_dim = audio_encoder.get_embedding_dim()
lm_dim = language_model.get_embedding_dim()

projection_config = config_dict["projection"].copy()
projection_config["audio_embedding_dim"] = audio_dim
projection_config["language_model_embedding_dim"] = lm_dim

projection = ModelRegistry.get_projection(
    "sequential",
    projection_config
)
```

## Project Structure

```
audiocaption/
├── core/                    # Base classes and registry
│   ├── base.py             # Abstract base classes
│   ├── registry.py          # Model registry
│   └── config.py           # Configuration management
│
├── models/                  # Model implementations
│   ├── audio_encoders/     # Audio encoder implementations
│   ├── language_models/    # Language model implementations
│   └── projections/        # Projection implementations
│
├── data/                    # Data loading
│   ├── dataset.py          # Dataset wrapper
│   └── preprocessing.py   # Audio preprocessing
│
├── training/                # Training pipeline
│   ├── trainer.py          # Main trainer
│   └── metrics.py          # Evaluation metrics
│
└── utils/                   # Utilities
    └── device.py           # Device detection

scripts/
├── train.py                # Training script
└── evaluate.py            # Evaluation script

configs/
└── default.yaml            # Example configuration
```

## Key Features

1. **Plugin Architecture**: Easy model swapping via decorators
2. **Registry System**: Automatic model discovery and instantiation
3. **Configuration-Based**: YAML configs for easy experimentation
4. **Modular Design**: Clean separation of concerns
5. **Extensible**: Add new models without modifying existing code

## Next Steps

- Check `README.md` for detailed documentation
- See `example_usage.py` for code examples
- Modify `configs/default.yaml` for your experiments
- Add your own models following the examples above
