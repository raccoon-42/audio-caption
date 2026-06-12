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
- JDK 11 (aac-metrics requires Java 8-13; Java 21+ will not work): `sudo apt install openjdk-11-jre`

### 1. Install dependencies

```bash
uv sync
uv run aac-metrics-download
```

### 2. Prepare data

Download and preprocess MusicCaps, then precompute CLAP embeddings:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/prepare_data.py --precompute-clap
```

### 3. Run the pipeline (per model)

LR search, baselines, noise floor, and evaluation are run manually.
Ablation scripts automate the many configs.

Example for GPT-2:

```bash
# LR search (check ceiling/flooring after each)
uv run python scripts/lr_search.py --model gpt2
uv run python scripts/lr_search.py --model gpt2 --proj-depth 1
uv run python scripts/lr_search.py --model gpt2 --proj-depth 3

# Baseline + noise floor training
uv run python scripts/train_gpt2.py
uv run python scripts/train_gpt2.py --seed 43
uv run python scripts/train_gpt2.py --seed 44

# Evaluate all seeds
uv run python scripts/evaluate.py --stage 2 --model gpt2 --ckpt-tag gpt2
uv run python scripts/evaluate.py --stage 2 --model gpt2 --ckpt-tag gpt2_seed43
uv run python scripts/evaluate.py --stage 2 --model gpt2 --ckpt-tag gpt2_seed44

# Arch ablations (6 configs, trains + evals each)
./scripts/run_arch_ablations.sh gpt2

# Decoding ablations (eval-only, ~23 configs)
./scripts/run_decoding_ablations.sh gpt2
```

> **Scope:** full ablations target GPT-2 / T5 / LLaMA — three distinct architecture
> families. OPT is the same family as GPT-2, so it is run baseline-only as a scale
> data point. The noise floor is measured on the two cheapest models (GPT-2 x3 seeds,
> T5 x2 seeds) and applied as an indicative scale to the rest.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| Data prep | `scripts/prepare_data.py` | Downloads MusicCaps, extracts audio, saves HF dataset |
| CLAP embeddings | `scripts/prepare_data.py --precompute-clap` | Caches CLAP audio embeddings to `data/embeddings/clap_embeddings.pt` |
| LR search | `scripts/lr_search.py --model <name>` | Optuna-based hyperparameter search per model; `--proj-depth N` for non-baseline depths |
| Training | `scripts/train_<model>.py` | Per-model training (stage 1 + stage 2); `--seed N` for noise-floor runs |
| Evaluation | `scripts/evaluate.py` | Generates captions and computes metrics |
| Arch ablations | `scripts/run_arch_ablations.sh` | Trains + evals 6 arch configs per model |
| Decoding ablations | `scripts/run_decoding_ablations.sh` | Eval-only, ~23 decoding configs per model |
| Report artifacts | `scripts/reporting/*.py` | LaTeX table fragments and figures generated from `results/` |

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

Computed via [`aac-metrics`](https://github.com/Labbeti/aac-metrics) (10 metrics from two composite calls: `spider` + `fense`):

- **Legacy:** BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr-D, SPICE, SPIDEr
- **AAC-specific:** SBERT-sim, FER, FENSE

SPICE and CIDEr-D are Java-backed (JDK 11 required, see Prerequisites). FENSE is reported as primary because the single-reference-per-clip nature of MusicCaps makes n-gram metrics noisy.

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
  utils.py                 Seed setting, depth-LR auto-loader
  run_arch_ablations.sh    Train + eval arch ablations (6 configs/model)
  run_decoding_ablations.sh  Eval-only decoding ablations (~23 configs/model)
  noise_floor.py           Compute noise floor stats from seed results
  reporting/               Report artifact generators (run from repo root)
    thesis_tables.py         LaTeX table fragments for the thesis report
    paper_tables.py          Lean table fragments for the two-column paper
    figures.py               Result figures (vector PDF)
    vocal_bias.py            Vocalist-bias statistics + tables
data/                 Dataset and cached embeddings
checkpoints/          Model checkpoints (per tag)
reports/              LaTeX sources; generated tables/ and figures/
results/              Structured results (per model):
  lr_search/            Optuna DBs, LR search JSONs, S1 projection .pt
  training/             Per-stage training history JSONs
  predictions/          Cached generation outputs
  ablations/arch/       Arch ablation result JSONs
  ablations/decoding/   Decoding ablation result JSONs
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
