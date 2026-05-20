# TODO

## Phase 1: LR Search

- [ ] LLaMA S1 LR search (running)
- [ ] LLaMA S2 LR search
- [ ] GPT-2 S2 LR re-search (hit ceiling at 9.95e-5, range widened)
- [ ] T5 S1+S2 LR re-search (both hit ceiling, range widened)
- [ ] Check all LR results for ceiling/flooring before proceeding

## Phase 2: Baselines

- [ ] LLaMA baseline (S1+S2) + eval
- [ ] GPT-2 baseline retrain if S2 LR changed
- [ ] T5 baseline retrain if S1/S2 LR changed
- OPT baseline: done, no LR issues

## Phase 3: Architectural Ablations

- [ ] LLaMA 6 arch ablations + eval
- [ ] GPT-2 retrain ablations if S2 LR changed
- [ ] T5 retrain ablations if S1/S2 LR changed
- OPT ablations: done

## Phase 4: Decoding Ablations (eval-only, on baseline checkpoint)

- [ ] GPT-2 18 decoding configs
- [ ] T5 18 decoding configs
- [ ] OPT 18 decoding configs
- [ ] LLaMA 18 decoding configs

## Phase 5 (optional): Joint Optuna Search

- [ ] Per-model joint search over (prefix_len, dropout, depth) + LR
- [ ] Per-model joint search over decoding params

## Phase 6: Final Comparison

- [ ] Update comparison notebook with all results
- [ ] Single-variable ablation tables
- [ ] Joint-optimized best configs (if phase 5 done)
- [ ] Thesis writeup
