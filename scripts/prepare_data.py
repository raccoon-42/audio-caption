"""Download MusicCaps audio and prepare the dataset for training."""

import os
import re
import subprocess
import argparse
import time
import threading
from pathlib import Path

import librosa
import torch
from datasets import load_dataset, load_from_disk, Audio
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor


stats = {"ok": 0, "fail": 0, "skip": 0, "bytes": 0}
stats_lock = threading.Lock()


def fmt_size(b):
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


def find_existing(full_audio_dir, ytid):
    import glob
    matches = glob.glob(f"{full_audio_dir}/{ytid}.*")
    return matches[0] if matches else None


def download_audio(ytid, full_audio_dir, retries=3, sleep_between=5):
    existing = find_existing(full_audio_dir, ytid)
    if existing:
        return existing, None

    url = f"https://www.youtube.com/watch?v={ytid}"
    out_template = f"{full_audio_dir}/{ytid}.%(ext)s"
    last_err = ""
    for attempt in range(1, retries + 1):
        res = subprocess.run(
            [
                "yt-dlp",
                "--cookies-from-browser", "firefox",
                "--js-runtimes", "node",
                "--remote-components", "ejs:github",
                "--sleep-interval", "3",
                "--max-sleep-interval", "6",
                "--throttled-rate", "100K",
                "--concurrent-fragments", "1",
                "-f", "bestaudio[ext=m4a]/bestaudio",
                url,
                "-o", out_template,
            ],
            capture_output=True,
            text=True,
        )
        downloaded = find_existing(full_audio_dir, ytid)
        if res.returncode == 0 and downloaded:
            return downloaded, None
        raw = res.stderr.strip().split("\n")[-1] if res.stderr.strip() else "unknown"
        last_err = re.sub(r"^ERROR:\s*\[youtube\]\s*\S+:\s*", "", raw) if "ERROR:" in raw else raw
        if "429" in res.stderr:
            time.sleep(sleep_between * attempt)
        else:
            break
    return None, last_err


def cut_segment(src, start, end, out):
    if not os.path.exists(src):
        return False
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-to", str(end),
            "-i", src,
            "-c:a", "pcm_s16le",
            "-ar", "44100",
            "-f", "wav",
            out,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0 and os.path.exists(out)


def process_sample(sample, segments_dir, full_audio_dir, failed_log):
    ytid = sample["ytid"]
    start = sample["start_s"]
    end = sample["end_s"]
    out = f"{segments_dir}/{ytid}_{start:.2f}_{end:.2f}.wav"

    if os.path.exists(out):
        with stats_lock:
            stats["skip"] += 1
            stats["bytes"] += os.path.getsize(out)
        return "skip", ytid, None

    full, err = download_audio(ytid, str(full_audio_dir))
    if full is None:
        with open(failed_log, "a") as f:
            f.write(f"DOWNLOAD_FAIL {ytid} | {err}\n")
        with stats_lock:
            stats["fail"] += 1
        return "fail", ytid, err

    if not cut_segment(full, start, end, out):
        with open(failed_log, "a") as f:
            f.write(f"CUT_FAIL {ytid} {start} {end}\n")
        with stats_lock:
            stats["fail"] += 1
        return "cut_fail", ytid, "ffmpeg segment failed"

    with stats_lock:
        stats["ok"] += 1
        stats["bytes"] += os.path.getsize(out)
    return "ok", ytid, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Root directory for all data artifacts")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precompute-clap", action="store_true",
                        help="Only precompute CLAP embeddings (skip download)")
    args = parser.parse_args()

    if args.precompute_clap:
        precompute_clap(args.data_dir)
        return

    data_dir = Path(args.data_dir).resolve()
    full_audio_dir = data_dir / "full_audio"
    segments_dir = data_dir / "segments"
    dataset_dir = data_dir / "dataset"
    failed_log = data_dir / "failed.txt"

    if failed_log.exists():
        failed_log.unlink()

    full_audio_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("google/musiccaps", split="train")
    total = len(ds)
    print(f"Total samples: {total}")

    # count already downloaded
    existing = sum(
        1 for s in ds
        if os.path.exists(
            f"{segments_dir}/{s['ytid']}_{s['start_s']:.2f}_{s['end_s']:.2f}.wav"
        )
    )
    print(f"Already downloaded: {existing}")
    print(f"Remaining: {total - existing}")
    print(f"Workers: {args.workers}\n")

    pbar = tqdm(total=total, desc="Processing")

    def worker(sample):
        result, ytid, err = process_sample(
            sample, str(segments_dir), str(full_audio_dir), str(failed_log)
        )
        with stats_lock:
            s = stats.copy()
        pbar.set_postfix(ok=s["ok"], fail=s["fail"], skip=s["skip"], size=fmt_size(s["bytes"]))
        pbar.update(1)
        if result == "ok":
            tqdm.write(f"  OK  {ytid}")
        elif result in ("fail", "cut_fail"):
            tqdm.write(f" FAIL {ytid} -- {err}")
        return result

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(worker, ds))

    pbar.close()

    print(f"\nDone. ok={stats['ok']}  fail={stats['fail']}  skip={stats['skip']}  downloaded={fmt_size(stats['bytes'])}")

    def has_segment(sample):
        ytid = sample["ytid"]
        start = sample["start_s"]
        end = sample["end_s"]
        path = f"{segments_dir}/{ytid}_{start:.2f}_{end:.2f}.wav"
        return os.path.exists(path)

    ds_filtered = ds.filter(has_segment, desc="Filtering available segments")
    print(f"Found {len(ds_filtered)} samples with audio")

    def add_audio_path(sample):
        ytid = sample["ytid"]
        start = sample["start_s"]
        end = sample["end_s"]
        sample["audio"] = f"{segments_dir}/{ytid}_{start:.2f}_{end:.2f}.wav"
        return sample

    ds_with_paths = ds_filtered.map(add_audio_path, desc="Adding audio paths")
    ds_with_audio = ds_with_paths.cast_column("audio", Audio(sampling_rate=44100))
    ds_with_audio.save_to_disk(str(dataset_dir))
    print(f"Dataset saved to {dataset_dir} ({len(ds_with_audio)} samples)")


def precompute_clap(data_dir):
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "dataset"
    out_path = data_dir / "clap_embeddings.pt"

    ds = load_from_disk(str(dataset_dir))
    print(f"Precomputing CLAP embeddings for {len(ds)} samples...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").to(device).eval()

    embeddings = []
    failed = 0
    for i in tqdm(range(len(ds)), desc="CLAP embeddings"):
        try:
            sample = ds[i]
            audio_array = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            if audio_array.ndim == 2:
                audio_array = audio_array.mean(axis=1)
            if sr != 48000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=48000)

            inputs = processor(audio=audio_array, sampling_rate=48000, return_tensors="pt")
            with torch.no_grad():
                out = model.get_audio_features(
                    **{k: v.to(device) for k, v in inputs.items()}
                )
                emb = out if isinstance(out, torch.Tensor) else out.pooler_output
            embeddings.append(emb.squeeze(0).cpu())
        except Exception as e:
            failed += 1
            embeddings.append(torch.zeros(model.config.projection_dim))
            if failed <= 5:
                print(f"  Sample {i} failed: {e}")

    embeddings = torch.stack(embeddings)
    torch.save(embeddings, out_path)
    print(f"Saved {embeddings.shape} to {out_path} ({failed} failed)")


if __name__ == "__main__":
    main()
