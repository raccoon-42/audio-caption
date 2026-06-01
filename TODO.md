# TODO

> **Reset note (2026-06-01):** three fixes landed that invalidate all prior LR
> searches, baselines, and evals — they must be rerun from scratch:
> - Projection LayerNorm is now actually applied in training (was silently off,
>   so every earlier checkpoint was trained without it).
> - LR search validates on the val split, not the held-out test rows (leak fixed).
> - Metrics moved to the aac-metrics suite (CIDEr-D, SPICE, SPIDEr) + FENSE.
>
> So "OPT done" etc. no longer holds — everything resets. Delete old Optuna
> `*_lr_search.db` and `*_s1_best_proj.pt` before rerunning (they auto-resume).

## Phase 0: Eval environment (Ubuntu, one-time)

- [ ] `uv add aac-metrics`; install a JRE (`sudo apt install default-jre`)
- [ ] First eval downloads SPICE/CIDEr Java models (~1GB, needs network once)
- [ ] Smoke-test the API/keys before a full run:
      `uv run python -c "from aac_metrics import evaluate; c,_=evaluate(['a dog barks'],[['a dog is barking']]); print(sorted(c))"`

## Phase 1: LR Search (rerun all 4 — LayerNorm on, val-split fixed)

- [ ] Delete stale `results/*/*_lr_search.db` and `*_s1_best_proj.pt`
- [ ] GPT-2 S1+S2
- [ ] T5 S1+S2
- [ ] OPT S1+S2
- [ ] LLaMA S1+S2
- [ ] Check all LR results for ceiling/flooring before proceeding

## Phase 2: Baselines (retrain all 4 with LayerNorm) + eval

- [ ] GPT-2 baseline (S1+S2) + eval
- [ ] T5 baseline (S1+S2) + eval
- [ ] OPT baseline (S1+S2) + eval
- [ ] LLaMA baseline (S1+S2) + eval   ← LLaMA reference data point

## Phase 3: Architectural Ablations (6 configs/model)

- [ ] GPT-2 6 arch ablations + eval
- [ ] T5 6 arch ablations + eval
- [ ] OPT 6 arch ablations + eval
- Note: the depth-3 variant needs its own LR search first
  (`lr_search.py --proj-depth 3`) — adviser flagged depth↔LR coupling.

## Phase 4: Decoding Ablations (eval-only, on baseline checkpoint)

- [ ] GPT-2 18 decoding configs
- [ ] T5 18 decoding configs
- [ ] OPT 18 decoding configs

## Phase 5: LLaMA ablations — ONLY IF TIME REMAINS

- [ ] LLaMA 6 arch ablations + eval
- [ ] LLaMA 18 decoding configs
- Decision rule: time one LLaMA baseline train on the 5090; promote to full only
  if ~6 more fit before the deadline. Otherwise stop and report LLaMA at
  "baseline / limited scope" (adviser-sanctioned fallback).

## Phase 6 (optional): Joint Optuna Search

- [ ] Per-model joint search over (prefix_len, dropout, depth) + LR
- [ ] Per-model joint search over decoding params

## Phase 7: Final Comparison

- [ ] Update comparison notebook for new metric keys (CIDEr-D, SPICE, SPIDEr)
- [ ] Single-variable ablation tables
- [ ] Joint-optimized best configs (if phase 6 done)
- [ ] Thesis writeup
