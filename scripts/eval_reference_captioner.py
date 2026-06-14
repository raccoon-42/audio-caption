"""Score a zero-shot reference captioner on the thesis test split.

Reference captioners (Audio Flamingo, Qwen2-Audio, ...) caption raw audio
directly, bypassing the CLAP->projection->LM pipeline, so evaluate.py cannot
run them (its --model is restricted to gpt2/t5/opt/llama). Instead we generate
here and reuse evaluate.compute_metrics, scoring on the IDENTICAL test rows as
the pipeline models -> drops into a separate reference-comparison table.

Default model is nvidia/audio-flamingo-next-captioner-hf. --backend selects the
generation API (each audio-LM family differs); auto-detected from --model-id.
Local backends (flamingo/qwen) need the Ubuntu GPU box (8B, BF16); the dashscope
backend is a remote API call (no local weights, runs anywhere the dataset lives).

Predictions are cached, so re-running re-scores without regenerating; pass a
fresh --tag to force a new generation.
"""

import argparse
import base64
import io
import json
import os
import re
import tempfile
import time
from pathlib import Path

import torch
import torchaudio
import yaml
from dotenv import load_dotenv
from datasets import load_from_disk
from tqdm import tqdm
from transformers import (
    AutoModel, AutoProcessor, Qwen2AudioForConditionalGeneration,
)

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


def trim_to_words(text, target):
    """Keep whole sentences until cumulative words >= target, completing the
    sentence that crosses the line. Drops any trailing incomplete fragment, so
    output is always sentence-terminated and ~target words (slightly over).

    Used only for the prompt-less Qwen3-Omni captioner, whose long-form output
    can't be reined in by a prompt the way Flamingo/Qwen2-Audio were -> normalise
    to the MusicCaps reference length (~50 words) for a fair lexical comparison."""
    text = text.strip()
    try:
        from nltk import sent_tokenize
        sents = sent_tokenize(text)
    except Exception:
        sents = re.split(r"(?<=[.!?])\s+", text)
    # Complete sentences only (terminal punctuation) -> never cut mid-sentence.
    sents = [s.strip() for s in sents if s.strip() and s.strip()[-1] in ".!?"]
    if not sents:
        return text
    out, n = [], 0
    for s in sents:
        out.append(s)
        n += len(s.split())
        if n >= target:
            break
    return " ".join(out)


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


# --- Backends: each audio-LM family has its own load + generation API. ---
class FlamingoBackend:
    """nvidia/audio-flamingo-*: AutoModel, audio passed as a file path via the
    chat template (tokenize=True), input_features cast to model dtype."""

    def __init__(self, model_id):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()

    def caption(self, wav, prompt, gen_kwargs):
        model, processor = self.model, self.processor
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
                conversation, tokenize=True, add_generation_prompt=True,
                return_dict=True,
            ).to(model.device)
            if "input_features" in batch:
                batch["input_features"] = batch["input_features"].to(model.dtype)
            with torch.no_grad():
                generated = model.generate(**batch, **gen_kwargs)
        prompt_len = batch["input_ids"].shape[1]
        return processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class QwenBackend:
    """Qwen/Qwen2-Audio-*: Qwen2AudioForConditionalGeneration, audio passed as an
    in-memory 16 kHz array via processor(audios=...), ChatML two-step build."""

    def __init__(self, model_id):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto").eval()
        sr = self.processor.feature_extractor.sampling_rate
        assert sr == TARGET_SR, f"processor expects {sr} Hz, decode targets {TARGET_SR}"

    def caption(self, wav, prompt, gen_kwargs):
        model, processor = self.model, self.processor
        audio = wav.squeeze(0).to(torch.float32).numpy()
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": "clip"},  # marker; array below
                {"type": "text", "text": prompt},
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
            generated = model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs.input_ids.size(1)
        return processor.batch_decode(
            generated[:, prompt_len:], skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()


class DashScopeBackend:
    """Qwen3-Omni-* via the DashScope OpenAI-compatible API -- no local weights,
    audio sent base64-encoded over HTTP. The -Captioner variant is single-turn,
    audio-only and ignores text prompts, so `prompt` is unused here. Key from
    DASHSCOPE_API_KEY; base URL from DASHSCOPE_BASE_URL (default intl/Singapore).

    audio_payload selects the content shape: "input_audio" (OpenAI-canonical) or
    "audio_url" (DashScope data-URL form) -- switch via --audio-payload if one
    is rejected. Audio clips must be <=30 s (MusicCaps clips are 10 s)."""

    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(self, model_id, audio_payload="input_audio"):
        from openai import OpenAI
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise SystemExit("Set DASHSCOPE_API_KEY (DashScope / Model Studio key).")
        # `or` (not .get default): a blank DASHSCOPE_BASE_URL= in .env is an
        # empty string, which would otherwise override the intl default.
        base_url = os.environ.get("DASHSCOPE_BASE_URL") or self.DEFAULT_BASE_URL
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_id = model_id
        self.audio_payload = audio_payload
        self.usage = {"prompt": 0, "completion": 0, "calls": 0}  # billed tokens
        # Stub so the caller's `.model.dtype/.device` prints work for both local
        # and remote backends.
        self.model = type("RemoteModel", (), {"dtype": "api", "device": base_url})()

    def _audio_content(self, wav):
        # soundfile, not torchaudio.save: the torchcodec-backed save picks the
        # container from a file extension and can't write to an in-memory buffer.
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, wav.squeeze(0).numpy(), TARGET_SR, format="WAV", subtype="PCM_16")
        b64 = base64.b64encode(buf.getvalue()).decode()
        if self.audio_payload == "audio_url":
            return {"type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{b64}"}}
        # DashScope input_audio.data is a data URL, NOT raw base64 (raw base64 is
        # parsed as a URL and rejected). OpenAI/OpenRouter want raw base64 here --
        # diverge if this backend is ever pointed at those.
        return {"type": "input_audio",
                "input_audio": {"data": f"data:;base64,{b64}", "format": "wav"}}

    def caption(self, wav, prompt, gen_kwargs):
        messages = [{"role": "user", "content": [self._audio_content(wav)]}]
        last_err = None
        for attempt in range(3):  # ride out transient API/network errors
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=gen_kwargs.get("max_new_tokens", 128),
                    temperature=0,  # greedy, matches the pipeline's decoding
                )
                if resp.usage:
                    self.usage["prompt"] += resp.usage.prompt_tokens or 0
                    self.usage["completion"] += resp.usage.completion_tokens or 0
                    self.usage["calls"] += 1
                return resp.choices[0].message.content.strip()
            except Exception as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise last_err


BACKENDS = {"flamingo": FlamingoBackend, "qwen": QwenBackend,
            "dashscope": DashScopeBackend}


def resolve_backend(name, model_id):
    if name != "auto":
        return name
    mid = model_id.lower()
    if "omni" in mid or "captioner" in mid:  # DashScope slugs (qwen3-omni-*)
        return "dashscope"
    return "qwen" if "qwen" in mid else "flamingo"


def main():
    load_dotenv()  # pick up DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL from .env
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gpt2.yaml",
                        help="Only used for the split seed (must match evaluate.py).")
    parser.add_argument("--model-id", default="nvidia/audio-flamingo-next-captioner-hf")
    parser.add_argument("--backend", default="auto",
                        choices=["auto", "flamingo", "qwen", "dashscope"],
                        help="Generation API; auto-detected from --model-id.")
    parser.add_argument("--audio-payload", default="input_audio",
                        choices=["input_audio", "audio_url"],
                        help="dashscope backend only: audio content shape. Switch "
                             "to audio_url if input_audio is rejected.")
    parser.add_argument("--name", default="audio_flamingo",
                        help="Short name for output dir/filenames.")
    parser.add_argument("--tag", default=None,
                        help="Output tag (defaults to --name). Predictions cache key.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="1.0 = none, matching evaluate.py stage-1 greedy. "
                             "Raise only if outputs degenerate/loop.")
    parser.add_argument("--trim-words", type=int, default=None,
                        help="Sentence-aware trim of each hypothesis to ~N words "
                             "(end-of-sentence boundary). Applied at scoring time "
                             "(raw output stays cached). For the prompt-less "
                             "captioner; ~50 matches MusicCaps reference length.")
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
        # The -Captioner model takes audio only; record that honestly instead of
        # the unused MusicCaps prompt.
        if "captioner" in args.model_id.lower():
            prompt = "(none -- captioner takes audio only)"
        # Reproduce the EXACT test split evaluate.py uses.
        ds = load_from_disk(str(Path(cfg["data_dir"])))
        test_data = ds.train_test_split(test_size=0.1, seed=seed)["test"]
        if args.limit:
            test_data = test_data.select(range(min(args.limit, len(test_data))))
        print(f"Test split: {len(test_data)} rows (seed {seed}).")

        backend_name = resolve_backend(args.backend, args.model_id)
        print(f"Loading {args.model_id} (backend: {backend_name}) ...")
        if backend_name == "dashscope":
            backend = DashScopeBackend(args.model_id, audio_payload=args.audio_payload)
        else:
            backend = BACKENDS[backend_name](args.model_id)
        print("Loaded. dtype:", backend.model.dtype, "device:", backend.model.device)
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
                hyp = backend.caption(wav, prompt, gen_kwargs)
                references.append(meta[i]["caption"])
                hypotheses.append(hyp)
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Sample {i} ({meta[i].get('ytid')}) failed: {e}")

        print(f"Generated {len(hypotheses)}/{len(test_data)} captions ({failed} failed)")
        if getattr(backend, "usage", None):
            u = backend.usage
            print(f"API usage: {u['calls']} calls, {u['prompt']} prompt + "
                  f"{u['completion']} completion = {u['prompt'] + u['completion']} tokens")
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

    # Trim at scoring time so the cached predictions stay raw -- re-running with a
    # different --trim-words re-scores from cache without re-calling the API.
    scored_hyps = hypotheses
    if args.trim_words:
        scored_hyps = [trim_to_words(h, args.trim_words) for h in hypotheses]
        print(f"Trimmed hypotheses to ~{args.trim_words} words (sentence-aware).")
    scored_predictions = [
        {"reference": r, "hypothesis": h} for r, h in zip(references, scored_hyps)
    ]

    print("Computing metrics...")
    metrics = compute_metrics(references, scored_hyps)
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
        "trim_words": args.trim_words,
        "metrics": metrics,
        "predictions": scored_predictions,
    }
    out_path = base_dir / f"{args.name}_metrics_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
