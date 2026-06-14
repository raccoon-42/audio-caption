"""Build the pairwise A-vs-B human-evaluation study from saved predictions.

Emits a self-contained static site under --out:
  data/pairs.json  : rater-facing pairs (neutral caption A / caption B)
  audio/clip_*.mp3 : transcoded 10s clips (small, web-friendly)
and a private --key file mapping each pair id to its true systems, used later
by compute_kappa.py for win-rates. The key is never shipped to raters.

Pairs:
  ref_best : GPT-2 best-decoding caption vs the human MusicCaps reference
  gpt2_t5  : GPT-2 best-decoding vs T5 best-decoding
  sanity   : human reference vs a reference written for a different clip

Predictions are joined to audio by matching the stored "reference" caption to
the test-split caption; position-based joins are unsafe because some runs drop
failed samples.
"""
import argparse
import json
import random
import re
import subprocess
from pathlib import Path

from datasets import load_from_disk


def norm(text):
    """Collapse whitespace so caption strings join across files."""
    return re.sub(r"\s+", " ", text).strip()


def load_predictions(path):
    data = json.loads(Path(path).read_text())
    return {norm(p["reference"]): p["hypothesis"] for p in data["predictions"]}


def build_clip_index(data_dir, seed):
    """Map normalised test-split caption -> clip identity.

    Column access avoids decoding the audio feature (some clips hold empty
    bytes, which the decoder rejects).
    """
    ds = load_from_disk(str(data_dir))
    test = ds.train_test_split(test_size=0.1, seed=seed)["test"]
    ytids, starts = test["ytid"], test["start_s"]
    ends, caps = test["end_s"], test["caption"]
    index = {}
    for ytid, start, end, cap in zip(ytids, starts, ends, caps):
        index[norm(cap)] = {
            "ytid": ytid, "start_s": start, "end_s": end, "reference": cap,
        }
    return index


def segment_path(segments_dir, clip):
    name = f"{clip['ytid']}_{clip['start_s']:.2f}_{clip['end_s']:.2f}.wav"
    return Path(segments_dir) / name


def transcode(src, dst):
    if dst.exists():
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-b:a", "128k", str(dst)],
        check=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/dataset")
    ap.add_argument("--segments", default="data/segments")
    ap.add_argument("--out", default="human_eval/site")
    ap.add_argument("--key", default="human_eval/key.json")
    ap.add_argument("--data-seed", type=int, default=42,
                    help="must match the seed used to generate predictions")
    ap.add_argument("--select-seed", type=int, default=123,
                    help="controls clip selection and A/B shuffling")
    ap.add_argument("--pred-best",
                    default="results/gpt2/ablations/decoding/gpt2_ablation_trim_ngram2.json",
                    help="GPT-2 best-decoding; paired vs reference and vs T5")
    ap.add_argument("--pred-t5",
                    default="results/t5/ablations/decoding/t5_ablation_trim_rep1.2.json")
    ap.add_argument("--n-ref-best", type=int, default=6)
    ap.add_argument("--n-gpt2-t5", type=int, default=6)
    ap.add_argument("--n-sanity", type=int, default=4)
    args = ap.parse_args()

    rng = random.Random(args.select_seed)

    clip_index = build_clip_index(args.data_dir, args.data_seed)
    best = load_predictions(args.pred_best)
    t5 = load_predictions(args.pred_t5)

    out = Path(args.out)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)

    pairs, key = [], []
    audio_cache = {}      # clip name -> mp3 filename
    used_clips = set()    # keep clips distinct across the whole study

    def clip_audio(clip):
        name = f"{clip['ytid']}_{clip['start_s']:.2f}_{clip['end_s']:.2f}"
        if name not in audio_cache:
            mp3 = f"clip_{len(audio_cache):04d}.mp3"
            transcode(segment_path(args.segments, clip), audio_dir / mp3)
            audio_cache[name] = mp3
        return audio_cache[name], name

    def add_pair(pair_type, cap, text_left, sys_left, text_right, sys_right,
                 correct=None):
        clip = clip_index[cap]
        mp3, clip_name = clip_audio(clip)
        used_clips.add(cap)
        # Blind the side assignment: randomise which system is shown as A.
        items = [(text_left, sys_left, "left"), (text_right, sys_right, "right")]
        rng.shuffle(items)
        # Opaque id so the pair type is not visible client-side.
        pid = f"p{len(pairs):02d}"
        pairs.append({
            "id": pid,
            "audio": f"audio/{mp3}",
            "captionA": items[0][0],
            "captionB": items[1][0],
        })
        entry = {"id": pid, "pair_type": pair_type, "clip": clip_name,
                 "systemA": items[0][1], "systemB": items[1][1]}
        if correct is not None:
            entry["correct_side"] = "A" if items[0][2] == correct else "B"
        key.append(entry)

    def take(candidates, n):
        """Distinct, not-yet-used clips, deterministically shuffled."""
        pool = [c for c in candidates if c not in used_clips]
        rng.shuffle(pool)
        if len(pool) < n:
            raise SystemExit(f"only {len(pool)} clips available, need {n}")
        return pool[:n]

    ref_best_caps = [c for c in clip_index if c in best]
    gpt2_t5_caps = [c for c in clip_index if c in best and c in t5]

    for cap in take(ref_best_caps, args.n_ref_best):
        add_pair("ref_best", cap,
                 clip_index[cap]["reference"], "reference",
                 best[cap], "gpt2_best")

    for cap in take(gpt2_t5_caps, args.n_gpt2_t5):
        add_pair("gpt2_t5", cap,
                 best[cap], "gpt2_best",
                 t5[cap], "t5_best")

    for cap in take(list(clip_index.keys()), args.n_sanity):
        wrong = rng.choice([c for c in clip_index if c not in used_clips])
        used_clips.add(wrong)
        add_pair("sanity", cap,
                 clip_index[cap]["reference"], "ref_correct",
                 clip_index[wrong]["reference"], "ref_mismatch",
                 correct="left")

    rng.shuffle(pairs)  # the site also reshuffles per rater
    (out / "data" / "pairs.json").write_text(json.dumps(pairs, indent=2))
    Path(args.key).write_text(json.dumps(key, indent=2))

    print(f"pairs: {len(pairs)} "
          f"(ref_best={args.n_ref_best}, gpt2_t5={args.n_gpt2_t5}, sanity={args.n_sanity})")
    print(f"distinct clips transcoded: {len(audio_cache)}")
    print(f"site -> {out}")
    print(f"key  -> {args.key} (keep private; not for raters)")


if __name__ == "__main__":
    main()
