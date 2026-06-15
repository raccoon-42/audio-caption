"""Run the LLM-as-judge panel over the same pairwise study the humans rated.

Each judge rates every pair exactly like a human rater: two questions (Q1 more
accurate, Q2 less wrong), each answered A / B / tie / cant_tell. Output drops
into the same json-dir shape compute_kappa.py reads, so a judge is just another
"rater" for the agreement analysis.

Two grounding conditions per audio-capable judge:
  audio : the judge hears the 10 s clip (site/audio/clip_*.mp3)
  text  : the judge sees only the human MusicCaps reference caption, no audio
The rubric/prompt is identical across judges and both conditions; only the
grounding payload (audio bytes vs reference text) swaps. Text-only judges
(Claude, Llama 4) run the text condition only.

Methodology controls (locked protocol):
  - temperature 0 where the API allows
  - both A/B presentation orders, averaged; order disagreement is reported
  - N repeats per order for intra-judge self-consistency
  - cant_tell abstention allowed, same as humans
  - exact model snapshot id recorded per run (closed models drift)

Providers are OpenAI-compatible: OpenRouter for Gemini/gpt-audio/Claude/Llama,
DashScope for Qwen. Keys from OPENROUTER_API_KEY / DASHSCOPE_API_KEY (.env).
"""
import argparse
import base64
import json
import re
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

CATS = ["A", "B", "tie", "cant_tell"]
DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Frontier-class general reasoners with the required modality (pre-registered
# inclusion criterion). Audio panel = 3 families; text panel = those 3 + Claude
# + Llama 4 (clean nesting). model ids are provider slugs -- verify/snapshot-pin
# before a real run; closed models drift.
JUDGES = {
    "gemini_2.5": {
        "provider": "openrouter", "model": "google/gemini-2.5-pro",
        "modalities": {"audio", "text"},
    },
    "gpt_audio": {
        "provider": "openrouter", "model": "openai/gpt-audio",
        # audio-only: the model 400s on a text-only request ("requires that
        # either input content or output modality contain audio") -> no text seat.
        "modalities": {"audio"},
    },
    # OpenAI's TEXT-panel seat: gpt-audio cannot run text-only, so a separate
    # text model fills the OpenAI lineage in the text condition. Different model
    # than gpt_audio -> NOT a valid per-judge audio-vs-text self-contrast for
    # OpenAI (that contrast holds only for gemini/qwen, same model both ways).
    "gpt_text": {
        "provider": "openrouter", "model": "openai/gpt-5.5",
        "modalities": {"text"},
    },
    "qwen3_omni": {
        # Panel's "Qwen3.5-Omni" judge. No qwen3-omni *instruct* slug exists on
        # Model Studio; the frontier non-realtime omni is qwen3.5-omni-plus.
        # Alias (not pinned); the served snapshot is still logged per result.
        "provider": "dashscope", "model": "qwen3.5-omni-plus",
        "modalities": {"audio", "text"},
    },
    "claude": {
        "provider": "openrouter", "model": "anthropic/claude-opus-4.8",
        "modalities": {"text"},
    },
    "llama4": {
        "provider": "openrouter", "model": "meta-llama/llama-4-maverick",
        "modalities": {"text"},
    },
}

# Identical rubric for every judge and both conditions. Only GROUNDING (what the
# ground truth is) swaps; the two questions + answer format never change.
RUBRIC = """\
You are comparing two candidate captions written for one short music clip.

{grounding}

The two captions are delimited below. Each is multi-sentence; treat everything
between a caption's tags as that single caption.

<caption_1>
{cap1}
</caption_1>

<caption_2>
{cap2}
</caption_2>

Answer two questions by judging each caption against the music:
  Q1 (accuracy): which caption describes the music MORE ACCURATELY?
  Q2 (errors):   which caption is LESS WRONG (fewer mistakes or hallucinations)?

For each question choose exactly one of:
  "1"          -> Caption 1 is better
  "2"          -> Caption 2 is better
  "tie"        -> the two are equally good
  "cant_tell"  -> you genuinely cannot judge from what you were given

Reply with ONLY a JSON object, no other text:
{{"q1": "1|2|tie|cant_tell", "q2": "1|2|tie|cant_tell"}}"""

GROUNDING_AUDIO = (
    "You are given the music clip itself (listen to the attached audio) and the "
    "two captions below."
)
GROUNDING_TEXT = (
    "You cannot hear the clip. Instead you are given a trusted human reference "
    "description of it; treat that reference as ground truth.\n\n"
    "<reference>\n{reference}\n</reference>"
)


def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def load_study(site_dir, key_path, pred_files):
    """Return {pid: pair} and {pid: key-entry} plus a reference caption per pid.

    The rater-facing pairs.json is blinded (no reference, no system labels); the
    reference needed for the text condition is recovered without the dataset:
      - non-sanity pairs always show one model hypothesis -> reverse-map that
        hypothesis back to its stored reference via the prediction files
      - sanity pairs show two references; the correct one is the ground truth
    """
    pairs = {p["id"]: p for p in
             json.loads((Path(site_dir) / "data" / "pairs.json").read_text())}
    key = {k["id"]: k for k in json.loads(Path(key_path).read_text())}

    hyp2ref = {}
    for pf in pred_files:
        for p in json.loads(Path(pf).read_text())["predictions"]:
            hyp2ref[norm(p["hypothesis"])] = p["reference"]

    references = {}
    for pid, k in key.items():
        pair = pairs[pid]
        if k["pair_type"] == "sanity":
            side = k.get("correct_side")
            references[pid] = pair["captionA"] if side == "A" else pair["captionB"]
            continue
        for cap in (pair["captionA"], pair["captionB"]):
            if norm(cap) in hyp2ref:
                references[pid] = hyp2ref[norm(cap)]
                break
        else:
            references[pid] = None  # text condition not scorable for this pair
    return pairs, key, references


def build_messages(pair, condition, reference, order, audio_dir, audio_format="mp3"):
    """One chat message list. order=0 shows captionA as Caption 1; order=1 swaps.

    Returns (messages, flip) where flip is True when the presentation order is
    swapped, so the judge's "1"/"2" can be mapped back to canonical A/B.
    """
    flip = order == 1
    cap1 = pair["captionB"] if flip else pair["captionA"]
    cap2 = pair["captionA"] if flip else pair["captionB"]

    if condition == "audio":
        grounding = GROUNDING_AUDIO
    else:
        grounding = GROUNDING_TEXT.format(reference=reference)
    text = RUBRIC.format(grounding=grounding, cap1=cap1, cap2=cap2)

    content = [{"type": "text", "text": text}]
    if condition == "audio":
        content.append(audio_content(
            audio_dir / Path(pair["audio"]).name, audio_format))
    messages = [{"role": "user", "content": content}]
    return messages, flip


def audio_content(mp3_path, audio_format="mp3", dashscope=False):
    """OpenAI-style input_audio block. DashScope wants a data: URL in `data`;
    OpenAI/OpenRouter want raw base64 (the captioner backend hit the same split).

    audio_format="wav" transcodes the clip to 16 kHz mono WAV in memory before
    sending: some valid MP3s are silently dropped by OpenAI's gpt-audio decoder
    (it replies "please provide the audio clip"), and WAV is decoded reliably."""
    if audio_format == "wav":
        import io
        import librosa
        import soundfile as sf
        y, _ = librosa.load(str(mp3_path), sr=16000, mono=True)
        buf = io.BytesIO()
        sf.write(buf, y, 16000, format="WAV", subtype="PCM_16")
        raw, fmt = buf.getvalue(), "wav"
    else:
        raw, fmt = Path(mp3_path).read_bytes(), "mp3"
    b64 = base64.b64encode(raw).decode()
    data = f"data:;base64,{b64}" if dashscope else b64
    return {"type": "input_audio", "input_audio": {"data": data, "format": fmt}}


def remap(answer, flip):
    """Map a judge "1"/"2"/tie/cant_tell to canonical A/B for the pair."""
    if answer in ("tie", "cant_tell"):
        return answer
    if answer == "1":
        return "B" if flip else "A"
    if answer == "2":
        return "A" if flip else "B"
    return None  # unparseable


def parse_response(raw):
    """Extract {q1, q2} from a model reply; tolerate prose around the JSON."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for q in ("q1", "q2"):
        v = str(obj.get(q, "")).strip().lower()
        out[q] = {"1": "1", "2": "2", "tie": "tie", "cant_tell": "cant_tell"}.get(v)
    return out


def plurality(votes):
    """Most common category; ties (no unique mode) resolve to 'tie'. Loses no
    info -- the full vote list is recorded alongside the collapsed label."""
    votes = [v for v in votes if v in CATS]
    if not votes:
        return None
    counts = Counter(votes)
    top = max(counts.values())
    winners = [c for c in CATS if counts.get(c, 0) == top]
    return winners[0] if len(winners) == 1 else "tie"


class Client:
    """Thin OpenAI-compatible wrapper; one per provider."""

    def __init__(self, provider):
        import os
        from openai import OpenAI
        self.provider = provider
        if provider == "openrouter":
            key = os.environ.get("OPENROUTER_API_KEY")
            if not key:
                raise SystemExit("Set OPENROUTER_API_KEY (.env).")
            self.client = OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
        elif provider == "dashscope":
            key = os.environ.get("DASHSCOPE_API_KEY")
            if not key:
                raise SystemExit("Set DASHSCOPE_API_KEY (.env).")
            base = os.environ.get("DASHSCOPE_BASE_URL") or DASHSCOPE_DEFAULT_BASE_URL
            self.client = OpenAI(api_key=key, base_url=base)
        else:
            raise ValueError(provider)
        self.usage = {"prompt": 0, "completion": 0, "calls": 0}
        self.snapshots = set()  # served model ids (closed models drift)

    def ask(self, model, messages, max_tokens):
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=0,
                )
                if resp.usage:
                    self.usage["prompt"] += resp.usage.prompt_tokens or 0
                    self.usage["completion"] += resp.usage.completion_tokens or 0
                    self.usage["calls"] += 1
                if resp.model:
                    self.snapshots.add(resp.model)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:  # ride out transient API/network errors
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise last_err


def fix_dashscope_audio(messages):
    """DashScope needs the data: URL form; rewrite audio blocks in place."""
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "input_audio":
                d = c["input_audio"]["data"]
                if not d.startswith("data:"):
                    c["input_audio"]["data"] = f"data:;base64,{d}"
    return messages


def run_judge(name, spec, condition, pairs, key, references, audio_dir,
              orders, repeats, limit, max_tokens, dry_run, only_pairs=None,
              audio_format="mp3"):
    client = None if dry_run else Client(spec["provider"])
    pids = list(pairs)
    if only_pairs:
        pids = [p for p in pids if p in only_pairs]
    elif limit:
        pids = pids[:limit]

    # Pre-filter so the progress bar total is the count actually scored.
    # Text condition = LLM-as-reference-metric: the reference is the gold
    # yardstick, so it can only score pairs where neither caption IS the
    # reference. That is exactly the gpt2_t5 pairs; ref_best/sanity put the
    # yardstick on one side (tautological) -> audio condition only.
    if condition == "text":
        pids = [p for p in pids
                if key[p]["pair_type"] == "gpt2_t5" and references.get(p) is not None]

    n_calls = n_unparsed = 0  # unparsed = reply with neither q1 nor q2 usable
    unparsed_samples = []     # (pid, order, raw[:300]) for diagnosis
    responses = {}
    bar = tqdm(pids, desc=f"{name}/{condition}", unit="pair", disable=dry_run)
    for pid in bar:
        pair = pairs[pid]
        canon = {"q1": [], "q2": []}      # canonical votes across orders x repeats
        per_order = {"q1": {}, "q2": {}}  # order -> plurality, for order-effect
        for order in orders:
            messages, flip = build_messages(
                pair, condition, references.get(pid), order, audio_dir,
                audio_format)
            if spec["provider"] == "dashscope":
                messages = fix_dashscope_audio(messages)
            order_votes = {"q1": [], "q2": []}
            for _ in range(repeats):
                if dry_run:
                    print(f"--- {name} [{condition}] {pid} order={order} ---")
                    print(json.dumps(messages, indent=2)[:1200])
                    parsed = {"q1": "1", "q2": "tie"}
                else:
                    raw = client.ask(spec["model"], messages, max_tokens)
                    parsed = parse_response(raw) or {"q1": None, "q2": None}
                    n_calls += 1
                    if parsed["q1"] is None and parsed["q2"] is None:
                        n_unparsed += 1
                        unparsed_samples.append((pid, order, raw[:300]))
                for q in ("q1", "q2"):
                    v = remap(parsed[q], flip) if parsed[q] else None
                    if v:
                        canon[q].append(v)
                        order_votes[q].append(v)
            for q in ("q1", "q2"):
                per_order[q][order] = plurality(order_votes[q])
        if not dry_run:
            bar.set_postfix(calls=n_calls, unparsed=n_unparsed)
        responses[pid] = {
            "q1": plurality(canon["q1"]), "q2": plurality(canon["q2"]),
            "q1_votes": canon["q1"], "q2_votes": canon["q2"],
            "order_effect_q1": len(set(per_order["q1"].values())) > 1,
            "order_effect_q2": len(set(per_order["q2"].values())) > 1,
        }
        if dry_run:
            break

    result = {
        "rater": f"{name}__{condition}",
        "judge": name, "condition": condition,
        "model": spec["model"], "provider": spec["provider"],
        "orders": orders, "repeats": repeats, "max_tokens": max_tokens,
        "responses": responses,
    }
    if client:
        result["usage"] = client.usage
        result["snapshots"] = sorted(client.snapshots)  # served model ids
        result["unparsed_calls"] = n_unparsed
        result["total_calls"] = n_calls
        result["unparsed_samples"] = unparsed_samples
    return result


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-dir", default="pairwise_eval/site")
    ap.add_argument("--key", default="pairwise_eval/key.json")
    ap.add_argument("--out-dir", default="results/llm_judge")
    ap.add_argument("--pred-files", nargs="+", default=[
        "results/gpt2/ablations/decoding/gpt2_ablation_trim_ngram2.json",
        "results/t5/ablations/decoding/t5_ablation_trim_rep1.2.json",
    ], help="Prediction JSONs used to recover the reference per pair (text cond).")
    ap.add_argument("--judges", nargs="+", default=list(JUDGES),
                    choices=list(JUDGES))
    ap.add_argument("--conditions", nargs="+", default=["audio", "text"],
                    choices=["audio", "text"])
    ap.add_argument("--orders", nargs="+", type=int, default=[0, 1],
                    choices=[0, 1], help="A/B presentation orders to average.")
    ap.add_argument("--repeats", type=int, default=3,
                    help="Calls per order (intra-judge self-consistency).")
    ap.add_argument("--max-tokens", type=int, default=2048,
                    help="Completion cap. Well above the tiny JSON because thinking "
                         "models (Gemini 2.5, Claude, gpt-5.5) spend tokens reasoning "
                         "first and truncate before the JSON at low caps (512 gave "
                         "~25%% unparsed on Gemini); non-thinking models stop early.")
    ap.add_argument("--limit", type=int, default=None, help="Cap pairs (debug).")
    ap.add_argument("--pairs", nargs="+", default=None,
                    help="Score only these pair ids (e.g. p04); for targeted "
                         "re-runs/diagnosis. Overrides --limit.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run judge/conditions whose output already exists.")
    ap.add_argument("--merge", action="store_true",
                    help="Merge this run's pairs into an existing output file "
                         "(update only the scored pids) instead of overwriting. "
                         "For targeted --pairs patching.")
    ap.add_argument("--audio-format", default="mp3", choices=["mp3", "wav"],
                    help="Audio container sent to judges. wav transcodes in memory; "
                         "use it when a judge silently drops an mp3 (gpt-audio).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print one built message per judge/condition, no API.")
    args = ap.parse_args()

    audio_dir = Path(args.site_dir) / "audio"
    pairs, key, references = load_study(args.site_dir, args.key, args.pred_files)
    print(f"Loaded {len(pairs)} pairs; "
          f"{sum(v is not None for v in references.values())} have a reference.")

    out_root = Path(args.out_dir)
    for name in args.judges:
        spec = JUDGES[name]
        for condition in args.conditions:
            if condition not in spec["modalities"]:
                continue
            out_path = out_root / condition / f"{name}.json"
            if (out_path.exists() and not args.overwrite and not args.merge
                    and not args.dry_run):
                print(f"\n=== judge {name} | {condition}: cached, skip "
                      f"(--overwrite to redo) ===")
                continue
            print(f"\n=== judge {name} | {condition} "
                  f"({spec['provider']}:{spec['model']}) ===")
            try:
                result = run_judge(
                    name, spec, condition, pairs, key, references, audio_dir,
                    args.orders, args.repeats, args.limit, args.max_tokens,
                    args.dry_run, only_pairs=args.pairs,
                    audio_format=args.audio_format)
            except Exception as e:  # one judge/condition failing must not abort the rest
                print(f"  FAILED: {type(e).__name__}: {e}")
                continue
            if args.dry_run:
                continue
            if args.merge and out_path.exists():
                prev = json.loads(out_path.read_text())
                prev.setdefault("responses", {}).update(result["responses"])
                result["responses"] = prev["responses"]  # keep untouched pids
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2))
            u = result.get("usage", {})
            scored = len(result["responses"])
            print(f"  {scored} pairs scored; {u.get('calls', 0)} API calls; "
                  f"{result.get('unparsed_calls', 0)} unparsed; "
                  f"snapshots={result.get('snapshots')} -> {out_path}")


if __name__ == "__main__":
    main()
