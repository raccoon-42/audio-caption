# Pairwise evaluation (human raters + LLM judges)

Pairwise A-vs-B evaluation for the music-captioning thesis. Each rater hears a
10s clip and picks the better of two blinded captions on two axes (more accurate
/ less wrong). The **same study** is run by two independent rater pools: human
raters (this directory's static site) and an LLM-judge panel (`llm_judge.py`).
`compute_kappa.py` ingests either pool and reports inter-rater Fleiss' kappa,
win-rates, and llm-vs-human consensus.

## Design

- **16 pairs**, every rater rates the same block (~5 min):
  - 6 x GPT-2 best-decoding (`trim_ngram2`, FENSE 0.580) vs human MusicCaps reference
  - 6 x GPT-2 best vs T5 best-decoding (`trim_rep1.2`, FENSE 0.576)
  - 4 x sanity (reference vs a reference written for a different clip)
- **2 questions/pair**: Q1 more accurate, Q2 less wrong. Answers A / B / Tie /
  Can't tell. "Can't tell" (no musical knowledge to judge) is an abstention,
  excluded from kappa and reported as an abstention rate.
- **10 raters** → Fleiss' kappa is computed on the shared block; win-rates on
  the two comparison types; sanity pairs filter inattentive raters.
- Expectation: GPT-2-vs-T5 is a statistical tie on FENSE, so it will likely be
  ~50/50 with humans too — itself a reportable corroboration.

## 1. Build the study

```
uv run python pairwise_eval/prepare_study.py
```

Writes `site/data/pairs.json` (rater-facing, blinded), `site/audio/clip_*.mp3`
(transcoded), and `pairwise_eval/key.json` (private system mapping; **never
published**). Pair ids are opaque (`p00`..`p15`) so the page does not reveal
which pairs are sanity checks. Defaults pick the top-FENSE decoding config per
model; override with `--pred-best` / `--pred-t5` / `--n-*` if needed.

## 2. Collect responses (Apps Script + Google Sheet)

1. Create a Google Sheet. **Extensions > Apps Script**, paste
   `pairwise_eval/apps_script.gs`, save.
2. **Deploy > New deployment > Web app**. Execute as: *Me*. Who has access:
   *Anyone*. Copy the Web app URL.
3. Paste that URL into `site/config.js` (`window.EVAL_ENDPOINT`).

Each submission appends one row per pair to the Sheet. The site also downloads
a `results_<rater>.json` as a fallback in case a POST fails.

## 3. Publish the site (GitHub Pages)

The site is fully static. Easiest path that keeps `key.json` private: publish a
**separate repo** containing only the contents of `pairwise_eval/site/`.

```
# from a fresh clone of a new public repo
cp -r /path/to/audio-caption/pairwise_eval/site/* .
git add . && git commit -m "music caption eval" && git push
```

Then enable **Settings > Pages > Deploy from branch > main / root**. Share the
Pages URL with the 10 raters. `key.json` stays in the thesis repo only.

(MusicCaps clips are public YouTube-sourced segments; publishing 16 short clips
for a research study is standard, but keep the site unlisted/shared by link.)

## 4. Analyse

Export the Sheet to CSV (File > Download > CSV) and/or collect the fallback
JSON files into a folder, then:

```
uv run python pairwise_eval/compute_kappa.py --csv responses.csv
# or
uv run python pairwise_eval/compute_kappa.py --json-dir pairwise_eval/responses
```

Reports: per-rater sanity accuracy (raters below 0.75 are dropped), Fleiss'
kappa and mean pairwise Cohen's kappa for Q1 and Q2, and win-rates per
comparison.

The same script ingests the LLM-judge panels (same `{rater, responses}` schema)
and reports llm-vs-human agreement. One condition per `--llm-dir`:

```
# llm-llm only (no human data needed)
uv run python pairwise_eval/compute_kappa.py --llm-dir results/llm_judge/audio
uv run python pairwise_eval/compute_kappa.py --llm-dir results/llm_judge/text

# all three: human-human + llm-llm + llm-vs-human consensus
uv run python pairwise_eval/compute_kappa.py --csv responses.csv \
    --llm-dir results/llm_judge/audio
```

Humans heard the clip, so the **audio** panel is the natural comparator; the
text panel only covers the `gpt2_t5` caption-vs-caption pairs (no audio, so no
sanity pairs — those judges are kept unfiltered). Cross-pool agreement reduces
each pool to a per-pair plurality label; within-pool ties are "undecided" and
excluded from the Cohen's kappa, reported separately.

## Files

| File | Purpose |
|------|---------|
| `prepare_study.py` | build pairs + transcode clips + write key |
| `site/` | static study (publish this) |
| `apps_script.gs` | Google Sheet collector |
| `compute_kappa.py` | Fleiss/Cohen kappa + win-rates + sanity filter + llm-vs-human consensus |
| `key.json` | private id→system map (do not publish) |
