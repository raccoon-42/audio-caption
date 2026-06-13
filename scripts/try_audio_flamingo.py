"""Smoke test: can Audio Flamingo caption our MusicCaps test clips?

Loads nvidia/audio-flamingo-next-captioner-hf, takes the first N rows of the
EXACT thesis test split, resamples each clip to mono 16 kHz, and generates a
caption. Prints reference vs hypothesis so we can eyeball whether this is a
usable reference captioner before wiring up full metric scoring.

Run on the Ubuntu GPU box (8B, BF16). Not a metric run -- no scoring here.
"""

import argparse
import tempfile
from pathlib import Path

import torch
import torchaudio
import yaml
from datasets import load_from_disk
from transformers import AutoModel, AutoProcessor

# Prompt mirrors MusicCaps reference style (~3-4 sentences, ~49 words, covering
# instruments/tempo/mood/vocals/genre/quality), NOT the card's verbose long-form
# default. Keep this in sync with eval_reference_captioner.py.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2.yaml",
                        help="Only used for the split seed (must match evaluate.py).")
    parser.add_argument("--model-id", default="nvidia/audio-flamingo-next-captioner-hf")
    parser.add_argument("--n", type=int, default=3, help="How many test clips to try.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]

    # Reproduce the EXACT test split evaluate.py uses, so any later metric run
    # scores on the identical rows as the pipeline models.
    ds = load_from_disk(str(Path(cfg["data_dir"])))
    test_data = ds.train_test_split(test_size=0.1, seed=seed)["test"]
    n = min(args.n, len(test_data))
    print(f"Test split: {len(test_data)} rows (seed {seed}); trying first {n}.")

    print(f"Loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    print("Loaded. dtype:", model.dtype, "device:", model.device)

    for i in range(n):
        row = test_data[i]
        wav = decode_to_16k_mono(row["audio"])

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            torchaudio.save(tmp.name, wav, TARGET_SR)
            conversation = [[{
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
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
                generated = model.generate(
                    **batch,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                )

        prompt_len = batch["input_ids"].shape[1]
        hyp = processor.batch_decode(
            generated[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print("\n" + "=" * 70)
        print(f"[{i}] ytid={row['ytid']}")
        print(f"REF: {row['caption']}")
        print(f"HYP: {hyp}")

    print("\n" + "=" * 70)
    print("Smoke test done. If captions look on-topic, proceed to full")
    print("generate-then-compute_metrics scoring (see project-captioner-eval).")


if __name__ == "__main__":
    main()
