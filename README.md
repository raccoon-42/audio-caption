# Audio Captioning: Ablation Study

Systematic ablations for CLAP-based music captioning on MusicCaps, along two axes:

- **Language model** — GPT-2 vs T5-base, with a fixed audio encoder.
- **Audio encoder** — general CLAP vs music-specialized encoders, with a fixed
  language model (GPT-2).

The architecture is the audio analogue of ClipCap: a frozen audio encoder, a learned
MLP projection to a prefix of pseudo-tokens, and a language-model decoder. Training is
two-stage: (1) projection only (frozen LM), (2) projection + LM fine-tuning.

## Language Models

| Model | Type | Parameters |
|-------|------|-----------|
| GPT-2 | Decoder-only | 124M |
| T5-base | Encoder-decoder | 220M |

> OPT-350M and LLaMA-3.2-1B were explored earlier as additional decoder-only models.
> Their training scripts (`train_opt.py`, `train_llama.py`) and configs (`opt.yaml`,
> `llama.yaml`) remain in the repo but are **out of the current scope**.

## Audio Encoders

The encoder ablation fixes the language model (GPT-2) and swaps the frozen audio
encoder. The general-audio CLAP is the baseline; the music-specialized encoders are
the comparison rows. Embeddings are precomputed once per encoder; the projection input
dim auto-derives, so the rest of the pipeline is unchanged.

| `--encoder` | HF repo | Type | Dim | SR |
|-------------|---------|------|-----|----|
| `clap` | `laion/clap-htsat-unfused` | General audio (baseline) | 512 | 48k |
| `clap-music` | `laion/larger_clap_music` | Music CLAP | 512 | 48k |
| `clap-music-speech` | `laion/larger_clap_music_and_speech` | Music+speech CLAP | 512 | 48k |
| `mert-95m` | `m-a-p/MERT-v1-95M` | Music SSL (frame-level) | 768 | 24k |
| `mert-330m` | `m-a-p/MERT-v1-330M` | Music SSL (frame-level) | 1024 | 24k |
| `musicfm` | `minzwon/MusicFM` | BEST-RQ conformer (frame-level) | 1024 | 24k |
| `muq` | `OpenMuQ/MuQ-large-msd-iter` | Music SSL (frame-level) | 1024 | 24k |
| `muq-mulan` | `OpenMuQ/MuQ-MuLan-large` | Joint audio-text (pooled) | 512 | 24k |

Pooled encoders (`clap`, `muq-mulan`) return a clip vector directly; frame-level
encoders are mean-pooled over time. The encoder comparison reports **stage-1
(frozen-LM)** numbers, since stage-2 full fine-tuning tends to flatten upstream
differences.

## Reference Captioners

Large zero-shot audio-LMs caption raw audio directly, bypassing the
CLAP→projection→LM pipeline. They are scored on the **same test split and the same
metrics** as a separate reference comparison (not as controlled ablation rows), to
contextualize the lightweight pipeline against off-the-shelf models.

| Model | HF repo / slug | Backend | Input |
|-------|----------------|---------|-------|
| Audio Flamingo | `nvidia/audio-flamingo-next-captioner-hf` | `flamingo` | 16 kHz wav path (local) |
| Qwen2-Audio | `Qwen/Qwen2-Audio-7B-Instruct` | `qwen` | 16 kHz array (local) |
| Qwen3-Omni Captioner | `qwen3-omni-30b-a3b-captioner` | `dashscope` | base64 WAV via DashScope API |

The first two run locally (BF16 on GPU). The 30B Qwen3-Omni captioner is too large to
run locally, so the `dashscope` backend calls the DashScope OpenAI-compatible API
(`openai` SDK + base URL); set `DASHSCOPE_API_KEY` in `.env` (auto-loaded), and
`DASHSCOPE_BASE_URL` only for non-Singapore regions.

Decoding is kept close to the pipeline (greedy, no repetition penalty) so the
comparison is fair; `max_new_tokens` is higher (128) to fit the captioners' natural
output length. The Qwen3-Omni captioner is **prompt-less and long-form** (it ignores
text prompts), so it is generated then **sentence-aware trimmed to ~50 words**
(`--trim-words`, the MusicCaps reference length) at scoring time — a normalization
applied to that row only, since unlike the others it can't be prompted for brevity.
Caveats reported alongside the numbers: these models may have seen MusicCaps in
pre-training (treat as a **reference upper bound**, not a clean held-out score), and
**FENSE** is the fair comparator since verbose output penalizes lexical n-gram metrics
on a style mismatch.

## Reproducing Results

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU with sufficient VRAM (batch_size=8)
- JDK 11 (aac-metrics requires Java 8-13; Java 21+ will not work): `sudo apt install openjdk-11-jre`
- Encoder-swap extras: `uv add muq einops torchaudio` (MERT may need `nnAudio`);
  MusicFM is not pip-installable and needs its repo cloned (pass `--musicfm-repo <dir>`).

### 1. Install dependencies

```bash
uv sync
uv run aac-metrics-download
```

### 2. Prepare data

Download and preprocess MusicCaps, then precompute the baseline CLAP embeddings:

```bash
uv run python scripts/prepare_data.py
uv run python scripts/prepare_data.py --precompute-clap
```

### 3a. Language-model ablation

LR search, baselines, noise floor, and evaluation are run manually.
Ablation scripts automate the many configs.

Example for GPT-2 (T5 is analogous):

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

> **Noise floor:** run-to-run variance is measured per model (GPT-2 x3 seeds,
> T5 x3 seeds), and each model is judged against its own floor (it does not transfer
> across architectures — T5 is ~2.5x noisier than GPT-2).

### 3b. Audio-encoder ablation (GPT-2 fixed)

```bash
# 1. Precompute embeddings per encoder (downloads weights on first load)
uv run python scripts/precompute_embeddings.py --encoder clap-music
uv run python scripts/precompute_embeddings.py --encoder mert-330m
uv run python scripts/precompute_embeddings.py --encoder musicfm --musicfm-repo ~/musicfm
# ... one per encoder in the table above

# 2. LR search for every encoder config (resume-safe)
./scripts/run_lr_search_encoders.sh

# 3. Stage-1 train + eval for every encoder (baseline CLAP is eval-only)
./scripts/run_train_eval_encoders.sh
```

Stage-2 can be run later from the saved `stage1_best.pt`:
`train_gpt2.py --stage 2 --config configs/gpt2_<encoder>.yaml`.

### 3c. Reference captioners (zero-shot, separate comparison)

```bash
# Smoke test first (prints REF vs HYP for a few clips)
uv run python scripts/try_audio_flamingo.py --n 5
uv run python scripts/try_qwen_audio.py --n 5

# Full generate + score on the shared test split (backend auto-detected)
uv run python scripts/eval_reference_captioner.py \
    --model-id nvidia/audio-flamingo-next-captioner-hf --name audio_flamingo
uv run python scripts/eval_reference_captioner.py \
    --model-id Qwen/Qwen2-Audio-7B-Instruct --name qwen2_audio

# Qwen3-Omni captioner via DashScope API (needs DASHSCOPE_API_KEY in .env);
# prompt-less long-form -> trim to ~50 words at scoring time
uv run python scripts/eval_reference_captioner.py \
    --model-id qwen3-omni-30b-a3b-captioner --name qwen3_omni_captioner \
    --max-new-tokens 256 --trim-words 50

# Comparison figure + table vs GPT-2 S1/S2
uv run python scripts/reporting/reference_comparison.py
```

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| Data prep | `scripts/prepare_data.py` | Downloads MusicCaps, extracts audio, saves HF dataset |
| Baseline embeddings | `scripts/prepare_data.py --precompute-clap` | Caches baseline CLAP embeddings to `data/embeddings/clap_embeddings.pt` |
| Encoder embeddings | `scripts/precompute_embeddings.py --encoder <name>` | Caches encoder embeddings to `data/embeddings/<encoder>_embeddings.pt` |
| LR search | `scripts/lr_search.py --model <name>` | Optuna LR search; `--config` for encoder swaps, `--proj-depth N` for non-baseline depths |
| Training | `scripts/train_<model>.py` | Per-model training; `--seed N` for noise-floor runs, `--stage {1,2,both}` for staged runs |
| Evaluation | `scripts/evaluate.py` | Generates captions and computes metrics |
| Arch ablations | `scripts/run_arch_ablations.sh` | Trains + evals 6 arch configs per model |
| Decoding ablations | `scripts/run_decoding_ablations.sh` | Eval-only, ~23 decoding configs per model |
| Encoder LR search | `scripts/run_lr_search_encoders.sh` | LR search across all encoder configs |
| Encoder train+eval | `scripts/run_train_eval_encoders.sh` | Stage-1 train + eval per encoder (baseline eval-only) |
| Reference captioner | `scripts/eval_reference_captioner.py` | Generate + score a zero-shot audio-LM (`--model-id`, `--backend`); reuses `compute_metrics` |
| Captioner smoke tests | `scripts/try_audio_flamingo.py`, `scripts/try_qwen_audio.py` | Print REF vs HYP for a few clips before a full run |
| Report artifacts | `scripts/reporting/*.py` | LaTeX table fragments and figures generated from `results/` |

## Ablations

**Architectural** (per language model):
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

**Audio encoder** (GPT-2 fixed, stage-1 reported):
- General CLAP (baseline) vs music-CLAP, MERT, MusicFM, MuQ / MuQ-MuLan

## Metrics

Computed via [`aac-metrics`](https://github.com/Labbeti/aac-metrics) (10 metrics from two composite calls: `spider` + `fense`):

- **Legacy:** BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr-D, SPICE, SPIDEr
- **AAC-specific:** SBERT-sim, FER, FENSE

SPICE and CIDEr-D are Java-backed (JDK 11 required, see Prerequisites). FENSE is reported as primary because the single-reference-per-clip nature of MusicCaps makes n-gram metrics noisy.

## Project Structure

```
configs/              Per-model + per-encoder YAML configs (hyperparams, paths)
  gpt2.yaml           Baseline CLAP language-model configs
  t5.yaml
  gpt2_<encoder>.yaml Encoder-swap configs (GPT-2 fixed, one per encoder)
  opt.yaml            Retained, out of current scope
  llama.yaml
scripts/
  train_gpt2.py       Self-contained training scripts (one per model)
  train_t5.py
  train_opt.py        Retained, out of current scope
  train_llama.py
  lr_search.py             Optuna LR search (--config for encoder swaps)
  evaluate.py              Shared evaluation (metrics + generation)
  prepare_data.py          Data download and baseline CLAP precomputation
  precompute_embeddings.py Encoder-swap embedding extraction (--encoder <name>)
  eval_reference_captioner.py Zero-shot audio-LM generate + score (multi-backend)
  try_audio_flamingo.py    Audio Flamingo smoke test
  try_qwen_audio.py        Qwen2-Audio smoke test
  dataset.py               Dataloader with embedding caching
  projection.py            Shared MLP projection module (input dim auto-derived)
  trainer.py               Shared training loop + early stopping
  utils.py                 Seed setting, depth-LR auto-loader
  run_arch_ablations.sh      Train + eval arch ablations (6 configs/model)
  run_decoding_ablations.sh  Eval-only decoding ablations (~23 configs/model)
  run_lr_search_encoders.sh  LR search across all encoder configs
  run_train_eval_encoders.sh Stage-1 train + eval per encoder
  noise_floor.py             Compute noise floor stats from seed results
  reporting/                 Report artifact generators (run from repo root)
    thesis_tables.py           LaTeX table fragments for the thesis report
    paper_tables.py            Lean table fragments for the two-column paper
    figures.py                 Result figures (vector PDF)
    reference_comparison.py    Pipeline vs zero-shot captioner figure + table
    vocal_bias.py              Vocalist-bias statistics + tables
data/                 Dataset and cached embeddings
  embeddings/         Precomputed per-encoder embedding tensors
checkpoints/          Model checkpoints (per tag; encoder swaps under encoder_swap/)
reports/              LaTeX sources; generated tables/ and figures/
results/              Structured results (per model / per encoder):
  lr_search/            Optuna DBs, LR search JSONs, S1 projection .pt
  training/             Per-stage training history JSONs
  predictions/          Cached generation outputs
  ablations/arch/       Arch ablation result JSONs
  ablations/decoding/   Decoding ablation result JSONs
  reference/<name>/     Zero-shot captioner predictions + metrics JSONs
```

## Reproducibility

- All experiments use `seed: 42` by default
- **Two independent seeds** (the key mental model — same data throughout, only the training luck changes):
  - **Split seed** — fixed at `cfg["seed"]` (42) for every run and every model. Controls *what data* the model sees (the train/val/test partition), so all runs evaluate on one identical ~550-clip test set and their scores are comparable.
  - **Train seed** — set via `--seed N` (42/43/44...). Controls *the random luck of training* on that fixed data: weight initialization, batch shuffle order, and dropout masks. This is the only thing noise-floor runs vary, so they measure training variance, not data-partition variance.
- Learning rates are tuned per model/encoder via Optuna (results cached, not re-run if found); noise-floor seeds reuse the tuned LR (no re-search)
- Pipeline scripts skip already-completed runs (checks for result JSON files)
- Per-epoch loss history is saved in all result JSONs
- Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` if running without internet
