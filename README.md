# Audio Captioning: Language Model Ablation Study

Systematic comparison of four language models for CLAP-based audio captioning on MusicCaps.

## Models

| Model | Type | Parameters |
|-------|------|-----------|
| GPT-2 | Decoder-only | 124M |
| T5-base | Encoder-decoder | 220M |
| OPT-350M | Decoder-only | 350M |
| LLaMA-3.2-1B | Decoder-only | 1B |

All models use a shared CLAP audio encoder (frozen) and a learned MLP projection layer. Training is two-stage: (1) projection only, (2) projection + LM fine-tuning.

## Reproducing Results

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU with sufficient VRAM (batch_size=8 for GPT-2/T5/OPT, batch_size=4 for LLaMA)

### 1. Install dependencies

```bash
uv sync
```

### 2. Prepare data

Download and preprocess MusicCaps, then precompute CLAP embeddings:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/prepare_data.py --precompute-clap
```

### 3. Run the pipeline (step by step)

Each script takes model names as arguments and skips already-completed runs.

```bash
# Step 1: LR search (re-search if ceiling/flooring detected)
./scripts/run_1_lr_search.sh gpt2 t5 opt llama

# Step 2: Train and evaluate baselines
./scripts/run_2_baselines.sh gpt2 t5 opt llama

# Step 3: Architectural ablations
./scripts/run_3_arch_ablations.sh gpt2 t5 opt llama

# Step 4: Decoding ablations (eval-only, on baseline checkpoint)
./scripts/run_4_decoding_ablations.sh gpt2 t5 opt llama
```

### 4. Evaluate a single model

```bash
uv run python scripts/evaluate.py --model gpt2 --stage 2
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| Data prep | `scripts/prepare_data.py` | Downloads MusicCaps, extracts audio, saves HF dataset |
| CLAP embeddings | `scripts/prepare_data.py --precompute-clap` | Caches CLAP audio embeddings to `data/clap_embeddings.pt` |
| LR search | `scripts/lr_search.py --model <name>` | Optuna-based hyperparameter search per model |
| Training | `scripts/train_<model>.py` | Per-model training (stage 1 + stage 2) |
| Evaluation | `scripts/evaluate.py` | Generates captions and computes metrics |

## Ablations

**Architectural** (varied per model):
- Prefix length: 4, 8, 16
- Dropout: 0.1, 0.3, 0.5
- Projection depth: 1, 2, 3

**Decoding** (applied to baseline checkpoint):
- Repetition penalty
- Beam search + length penalty
- N-gram blocking
- Nucleus sampling (top-p)

## Metrics

BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr, FENSE

## Project Structure

```
configs/              Per-model YAML configs (hyperparams, paths)
scripts/
  train_gpt2.py       Self-contained training scripts (one per model)
  train_t5.py
  train_opt.py
  train_llama.py
  lr_search.py         Optuna LR search for all models
  evaluate.py          Shared evaluation (metrics + generation)
  prepare_data.py      Data download and CLAP precomputation
  dataset.py           Dataloader with CLAP caching
  projection.py        Shared MLP projection module
  trainer.py           Shared training loop + early stopping
  utils.py             Seed setting
  run_1_lr_search.sh       LR search per model
  run_2_baselines.sh       Train + eval baselines
  run_3_arch_ablations.sh  Train + eval arch ablations
  run_4_decoding_ablations.sh  Eval-only decoding ablations
data/                 Dataset and cached embeddings
checkpoints/          Model checkpoints (per tag)
results/              JSON results (per model, per ablation)
```

## Reproducibility

- All experiments use `seed: 42`
- Learning rates are tuned per model via Optuna (results cached, not re-run if found)
- Pipeline scripts skip already-completed runs (checks for result JSON files)
- Per-epoch loss history is saved in all result JSONs
- Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` if running without internet
