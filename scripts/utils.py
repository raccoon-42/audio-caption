import json
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def override_lrs_for_depth(cfg, model_key, proj_depth, results_dir="results"):
    if proj_depth == 2:
        return
    lr_json = Path(results_dir) / model_key / f"{model_key}_depth{proj_depth}_lr_search.json"
    if not lr_json.exists():
        raise FileNotFoundError(
            f"No LR search results for depth={proj_depth}: {lr_json}\n"
            f"Run: uv run python scripts/lr_search.py --model {model_key} --proj-depth {proj_depth}"
        )
    with open(lr_json) as f:
        lr_data = json.load(f)
    cfg["stage1"]["lr"] = lr_data["stage1"]["best_lr"]
    cfg["stage2"]["projection_lr"] = lr_data["stage2"]["best_params"]["stage2_proj_lr"]
    cfg["stage2"]["lm_lr"] = lr_data["stage2"]["best_params"]["stage2_lm_lr"]
    print(f"Loaded depth-{proj_depth} LRs from {lr_json}")
