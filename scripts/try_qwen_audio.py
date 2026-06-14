"""Smoke test: can Qwen2-Audio caption our MusicCaps test clips?

Mirror of try_audio_flamingo.py for Qwen/Qwen2-Audio-7B-Instruct. Qwen's API
differs: Qwen2AudioForConditionalGeneration, audio passed as in-memory arrays
via processor(audios=...) at 16 kHz (NOT a file path), ChatML template.

Takes the first N rows of the EXACT thesis test split, decodes each clip to
mono 16 kHz, and prints reference vs hypothesis. Run on the Ubuntu GPU box.
Not a metric run -- scoring is done by eval_reference_captioner.py.

If loading raises KeyError: 'qwen2-audio', the installed transformers predates
Qwen2-Audio support -- bump transformers.
"""

import argparse
from pathlib import Path

import torch
import torchaudio
import yaml
from datasets import load_from_disk
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# Same MusicCaps-style prompt as the Flamingo smoke test (keep in sync).
DEFAULT_PROMPT = (
    "Describe this music in 2-3 concise sentences. Mention the instruments, the "
    "tempo and rhythm, the mood, any vocals, the genre or likely use, and the "
    "audio/recording quality."
)
TARGET_SR = 16000


def decode_to_16k_mono(audio_decoder):
    """torchcodec AudioDecoder -> 1-D float32 numpy array, mono at 16 kHz."""
    samples = audio_decoder.get_all_samples()
    wav = samples.data  # (channels, n)
    sr = samples.sample_rate
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).to(torch.float32).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2.yaml",
                        help="Only used for the split seed (must match evaluate.py).")
    parser.add_argument("--model-id", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--n", type=int, default=3, help="How many test clips to try.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["seed"]

    ds = load_from_disk(str(Path(cfg["data_dir"])))
    test_data = ds.train_test_split(test_size=0.1, seed=seed)["test"]
    meta = test_data.remove_columns(["audio"])  # caption/ytid without decoding audio
    n = min(args.n, len(test_data))
    print(f"Test split: {len(test_data)} rows (seed {seed}); trying first {n}.")

    print(f"Loading {args.model_id} ...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()
    print("Loaded. dtype:", model.dtype, "device:", model.device)
    sr = processor.feature_extractor.sampling_rate
    assert sr == TARGET_SR, f"processor expects {sr} Hz, decode targets {TARGET_SR}"

    for i in range(n):
        audio = decode_to_16k_mono(test_data[i]["audio"])
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "clip"},  # marker; array supplied below
                {"type": "text", "text": args.prompt},
            ],
        }]
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False)
        # transformers 5.x processor takes `audio=` (singular); the old `audios=`
        # is silently ignored -> model sees no audio and hallucinates from text.
        inputs = processor(
            text=text, audio=[audio], sampling_rate=TARGET_SR,
            return_tensors="pt", padding=True,
        ).to(model.device)

        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        hyp = processor.batch_decode(
            generate_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        print("\n" + "=" * 70)
        print(f"[{i}] ytid={meta[i]['ytid']}")
        print(f"REF: {meta[i]['caption']}")
        print(f"HYP: {hyp}")

    print("\n" + "=" * 70)
    print("Smoke test done. If captions look on-topic, score the full split with")
    print("eval_reference_captioner.py --model-id Qwen/Qwen2-Audio-7B-Instruct "
          "--name qwen2_audio (needs the Qwen processor path wired in).")


if __name__ == "__main__":
    main()
