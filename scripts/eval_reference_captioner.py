"""Score a zero-shot reference captioner on the thesis test split.

Reference captioners (Audio Flamingo, Qwen2-Audio, ...) caption raw audio
directly, bypassing the CLAP->projection->LM pipeline, so evaluate.py cannot
run them (its --model is restricted to gpt2/t5/opt/llama). Instead we generate
here and reuse evaluate.compute_metrics, scoring on the IDENTICAL test rows as
the pipeline models -> drops into a separate reference-comparison table.

Default model is nvidia/audio-flamingo-next-captioner-hf; --model-id generalizes
to any HF chat-style audio-LM with the same processor/apply_chat_template API.

Run on the Ubuntu GPU box (8B, BF16). Predictions are cached, so re-running
re-scores without regenerating; pass a fresh --tag to force a new generation.
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
import torchaudio
import yaml
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from evaluate import compute_metrics

# Prompt mirrors MusicCaps reference style (~3-4 sentences, ~49 words, covering
# instruments/tempo/mood/vocals/genre/quality), NOT the card's verbose long-form
# default (max_new_tokens=2048) which craters lexical metrics on a style
# mismatch. Report FENSE as the fair comparator. Prompt is a CLI flag.
DEFAULT_PROMPT = (
    "Describe this music in 2-3 concise sentences. Mention the instruments, the "
    "tempo and rhythm, the mood, any vocals, the genre or likely use, and the "
    "audio/recording quality."
)
TARGET_SR = 16000


def decode_to_16k_mono(audio_decoder):
    """torchcodec AudioDecoder -> (1, samples) float32 mono at 16 kHz."""
    samples = audio_decoder.get_all_samples()
    wav = samples.data  # (channels, n)
    sr = samples.sample_rate
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:  # downmix to mono
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.to(torch.float32)


def generate_caption(model, processor, wav, prompt, gen_kwargs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        torchaudio.save(tmp.name, wav, TARGET_SR)
        conversation = [[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "path": tmp.name},
            ],
        }]]
        batch = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(model.dtype)

        with torch.no_grad():
            generated = model.generate(**batch, **gen_kwargs)

    prompt_len = batch["input_ids"].shape[1]
    return processor.batch_decode(
        generated[:, prompt_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2.yaml",
                        help="Only used for the split seed (must match evaluate.py).")
    parser.add_argument("--model-id", default="nvidia/audio-flamingo-next-captioner-hf")
    parser.add_argument("--name", default="audio_flamingo",
                        help="Short name for output dir/filenames.")
    parser.add_argument("--tag", default=None,
                        help="Output tag (defaults to --name). Predictions cache key.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="1.0 = none, matching evaluate.py stage-1 greedy. "
                             "Raise only if outputs degenerate/loop.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap rows (debug); default = full test split.")
    args = parser.parse_args()

    # Fold --limit into the tag so a partial debug run never poisons the
    # full-run prediction cache (the cache key doesn't encode row count).
    tag = args.tag or args.name
    if args.limit and not args.tag:
        tag = f"{tag}_n{args.limit}"
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]

    base_dir = Path(cfg["results_dir"]) / "reference" / args.name
    pred_dir = base_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{args.name}_predictions_{tag}.json"

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    if pred_path.exists():
        # Cached-predictions shortcut (mirrors evaluate.py): re-score without
        # reloading the 8B model or regenerating.
        print(f"Found cached predictions: {pred_path}")
        with open(pred_path) as f:
            pred_data = json.load(f)
        references = [p["reference"] for p in pred_data["predictions"]]
        hypotheses = [p["hypothesis"] for p in pred_data["predictions"]]
        predictions = pred_data["predictions"]
        failed = pred_data["num_failed"]
        gen_kwargs = pred_data["gen_kwargs"]
        prompt = pred_data.get("prompt", args.prompt)
    else:
        prompt = args.prompt
        # Reproduce the EXACT test split evaluate.py uses.
        ds = load_from_disk(str(Path(cfg["data_dir"])))
        test_data = ds.train_test_split(test_size=0.1, seed=seed)["test"]
        if args.limit:
            test_data = test_data.select(range(min(args.limit, len(test_data))))
        print(f"Test split: {len(test_data)} rows (seed {seed}).")

        print(f"Loading {args.model_id} ...")
        processor = AutoProcessor.from_pretrained(args.model_id)
        model = AutoModel.from_pretrained(
            args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
        ).eval()
        print("Loaded. dtype:", model.dtype, "device:", model.device)
        print(f"Prompt: {prompt!r}")
        print(f"Gen kwargs: {gen_kwargs}")

        # Caption/ytid from an audio-free view so reading them never triggers
        # the audio decode -- a few MusicCaps rows have empty/corrupt clips that
        # blow up torchcodec, and indexing the full row is what decodes audio.
        # Decode audio lazily inside the try so a bad clip skips only that row.
        meta = test_data.remove_columns(["audio"])
        references, hypotheses = [], []
        failed = 0
        for i in tqdm(range(len(test_data)), desc="Generating"):
            try:
                wav = decode_to_16k_mono(test_data[i]["audio"])
                hyp = generate_caption(model, processor, wav, prompt, gen_kwargs)
                references.append(meta[i]["caption"])
                hypotheses.append(hyp)
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Sample {i} ({meta[i].get('ytid')}) failed: {e}")

        print(f"Generated {len(hypotheses)}/{len(test_data)} captions ({failed} failed)")
        predictions = [
            {"reference": r, "hypothesis": h} for r, h in zip(references, hypotheses)
        ]
        with open(pred_path, "w") as f:
            json.dump({
                "model": args.model_id,
                "stage": "reference",
                "seed": seed,
                "prompt": prompt,
                "num_samples": len(hypotheses),
                "num_failed": failed,
                "gen_kwargs": gen_kwargs,
                "predictions": predictions,
            }, f, indent=2)
        print(f"Predictions cached to {pred_path}")

    print("Computing metrics...")
    metrics = compute_metrics(references, hypotheses)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    results = {
        "model": args.model_id,
        "stage": "reference",
        "seed": seed,
        "prompt": prompt,
        "num_samples": len(hypotheses),
        "num_failed": failed,
        "gen_kwargs": gen_kwargs,
        "metrics": metrics,
        "predictions": predictions,
    }
    out_path = base_dir / f"{args.name}_metrics_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
