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
- A JRE (for SPICE / CIDEr-D, computed via the `aac-metrics` Java backend)

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

# Step 2b: Noise floor — retrain the baseline with extra seeds to measure
#          run-to-run variance (GPT-2 x3 primary, T5 x2 cross-check).
#          Reuses the seed-42 baseline above; only the training RNG varies.
./scripts/run_2b_noise_floor.sh gpt2 t5

# Step 3: Architectural ablations
./scripts/run_3_arch_ablations.sh gpt2 t5 llama

# Step 4: Decoding ablations (eval-only, on baseline checkpoint)
./scripts/run_4_decoding_ablations.sh gpt2 t5 llama
```

> **Scope:** full ablations (steps 3-4) target GPT-2 / T5 / LLaMA — three distinct
> architecture families. OPT is the same family as GPT-2, so it is run baseline-only
> (steps 1-2) as a scale data point. The noise floor (step 2b) is measured on the two
> cheapest models and applied as an indicative scale to the rest.

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
| Training | `scripts/train_<model>.py` | Per-model training (stage 1 + stage 2); `--seed N` for noise-floor runs |
| Noise floor | `scripts/run_2b_noise_floor.sh` | Retrains baselines with extra seeds; floor = FENSE spread across seeds |
| Evaluation | `scripts/evaluate.py` | Generates captions and computes metrics |

## Ablations

**Architectural** (varied per model):
- Prefix length: 4, 8, 16
- Dropout: 0.1, 0.3, 0.5
- Projection depth: 1, 2, 3

**Decoding** (applied to baseline checkpoint, eval-only):
- Repetition penalty
- Beam search + length penalty
- N-gram blocking
- Max new tokens
- Nucleus sampling (top-p)
- Sentence trimming (opt-in via `--trim-incomplete`)

## Metrics

Computed via the [`aac-metrics`](https://github.com/Labbeti/aac-metrics) suite (canonical DCASE implementations) plus the official FENSE:

- BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr-D, SPICE, SPIDEr
- FENSE — primary metric (audio-aware, reference-robust)

SPICE and CIDEr-D are Java-backed, so a JRE is required (see Prerequisites). FENSE is reported as primary because the single-reference-per-clip nature of MusicCaps makes n-gram metrics noisy; those are reported as supporting.

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
  run_2b_noise_floor.sh    Retrain baselines with extra seeds (variance/noise floor)
  run_3_arch_ablations.sh  Train + eval arch ablations
  run_4_decoding_ablations.sh  Eval-only decoding ablations
data/                 Dataset and cached embeddings
checkpoints/          Model checkpoints (per tag)
results/              JSON results (per model, per ablation)
```

## Reproducibility

- All experiments use `seed: 42` by default
- **Two independent seeds** (the key mental model — same data throughout, only the training luck changes):
  - **Split seed** — fixed at `cfg["seed"]` (42) for every run and every model. Controls *what data* the model sees (the train/val/test partition), so all runs evaluate on one identical ~550-clip test set and their scores are comparable.
  - **Train seed** — set via `--seed N` (42/43/44...). Controls *the random luck of training* on that fixed data: weight initialization, batch shuffle order, and dropout masks. This is the only thing noise-floor runs vary, so they measure training variance, not data-partition variance.
- Learning rates are tuned per model via Optuna (results cached, not re-run if found); noise-floor seeds reuse the tuned LR (no re-search)
- Pipeline scripts skip already-completed runs (checks for result JSON files)
- Per-epoch loss history is saved in all result JSONs
- Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` if running without internet
