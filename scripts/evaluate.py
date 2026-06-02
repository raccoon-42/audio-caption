"""Shared evaluation script for all audio captioning models."""

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from datasets import load_from_disk
from aac_metrics import evaluate as aac_evaluate
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OPTForCausalLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from projection import Projection
from utils import set_seed


DEFAULT_MODEL_NAMES = {
    "gpt2": "gpt2",
    "t5": "t5-base",
    "opt": "facebook/opt-350m",

    "llama": "meta-llama/Llama-3.2-1B",
}


def load_model(model_key, model_name, device):
    if model_key == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        lm_dim = model.config.n_embd
        model_type = "decoder"
    elif model_key == "t5":
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        lm_dim = model.config.d_model
        model_type = "encoder_decoder"
    elif model_key == "opt":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
        lm_dim = model.config.word_embed_proj_dim
        model_type = "decoder"
    elif model_key == "llama":
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(device)
        lm_dim = model.config.hidden_size
        model_type = "decoder"
    else:
        raise ValueError(f"Unknown model: {model_key}")

    model.eval()
    return model, tokenizer, lm_dim, model_type


def load_checkpoints(model_key, model, projection, stage, ckpt_dir):
    if stage == 1:
        projection.load_state_dict(
            torch.load(ckpt_dir / "stage1_best.pt", weights_only=True, map_location="cpu")
        )
    else:
        projection.load_state_dict(
            torch.load(ckpt_dir / "stage2_proj_best.pt", weights_only=True, map_location="cpu")
        )
        lm_ckpt = ckpt_dir / f"stage2_{model_key}_best.pt"
        if lm_ckpt.exists():
            model.load_state_dict(
                torch.load(lm_ckpt, weights_only=True, map_location="cpu")
            )
        else:
            print(f"Warning: {lm_ckpt} not found, using pretrained weights")


@torch.no_grad()
def generate_caption(model_key, model, projection, tokenizer, audio_emb,
                     prefix_len, device, gen_kwargs):
    prefix = projection(audio_emb.unsqueeze(0))

    if model_key == "t5":
        encoder_attn = torch.ones(1, prefix_len, device=device)
        output_ids = model.generate(
            inputs_embeds=prefix,
            attention_mask=encoder_attn,
            **gen_kwargs,
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    elif model_key == "opt":
        outputs = model(inputs_embeds=prefix, use_cache=True)
        past = outputs.past_key_values

        start_id = model.config.bos_token_id or 2
        start_token = torch.tensor([[start_id]], device=device)
        attention_mask = torch.ones(1, prefix_len + 1, device=device)

        output_ids = model.generate(
            input_ids=start_token,
            attention_mask=attention_mask,
            past_key_values=past,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **gen_kwargs,
        )
        caption = tokenizer.decode(output_ids[0, 1:], skip_special_tokens=True).strip()
    else:
        attention_mask = torch.ones(1, prefix_len, device=device)
        output_ids = model.generate(
            inputs_embeds=prefix,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **gen_kwargs,
        )
        caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return caption


def trim_incomplete_sentence(text):
    match = re.search(r'(.*[.!?])', text)
    if match:
        return match.group(1).strip()
    return text.strip()


def compute_metrics(references, hypotheses):
    references = [r.replace("\n", " ").strip() for r in references]
    hypotheses = [h.replace("\n", " ").strip() for h in hypotheses]
    mult_references = [[ref] for ref in references]

    print("  Computing aac-metrics...")
    corpus_scores, _ = aac_evaluate(
        hypotheses, mult_references,
        metrics=[
            "bleu_1", "bleu_4", "meteor", "rouge_l",
            "spider", "fense",
        ],
    )

    def score(key):
        return float(corpus_scores[key].item())

    return {
        "BLEU-1": score("bleu_1"),
        "BLEU-4": score("bleu_4"),
        "METEOR": score("meteor"),
        "ROUGE-L": score("rouge_l"),
        "CIDEr-D": score("cider_d"),
        "SPICE": score("spice"),
        "SPIDEr": score("spider"),
        "SBERT-sim": score("sbert_sim"),
        "FER": score("fer"),
        "FENSE": score("fense"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["gpt2", "t5", "opt", "llama"])
    parser.add_argument("--stage", type=int, default=2, choices=[1, 2])
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--prefix-len", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--proj-depth", type=int, default=2)
    parser.add_argument("--use-layernorm", action="store_true")
    parser.add_argument("--trim-incomplete", action="store_true")
    parser.add_argument("--ckpt-tag", type=str, default=None,
                        help="Checkpoint dir tag, e.g. 'gpt2_prefix4'")
    parser.add_argument("--ablation-tag", type=str, default=None,
                        help="Tag for output filename, e.g. 'rep_1.2'")
    args = parser.parse_args()

    config_path = args.config or f"configs/{args.model}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg["seed"]
    set_seed(seed)

    results_dir = Path(cfg["results_dir"]) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    tag = args.ablation_tag or args.ckpt_tag
    pred_path = results_dir / f"{args.model}_predictions_{tag}.json"

    if pred_path.exists():
        print(f"Found cached predictions: {pred_path}")
        with open(pred_path) as f:
            pred_data = json.load(f)
        references = [p["reference"] for p in pred_data["predictions"]]
        hypotheses = [p["hypothesis"] for p in pred_data["predictions"]]
        model_name = pred_data["model"]
        failed = pred_data["num_failed"]
        gen_kwargs = pred_data["gen_kwargs"]
        predictions = pred_data["predictions"]
    else:
        prefix_len = args.prefix_len or cfg["prefix_len"]
        dropout = args.dropout if args.dropout is not None else cfg["dropout"]
        use_layernorm = args.use_layernorm or cfg.get("use_layernorm", False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = cfg.get("model_name", DEFAULT_MODEL_NAMES.get(args.model, args.model))

        print("Loading precomputed CLAP embeddings...")
        data_dir = Path(cfg["data_dir"])
        all_clap = torch.load(data_dir.parent / "clap_embeddings.pt", weights_only=True)
        audio_dim = all_clap.shape[1]

        print(f"Loading {model_name}...")
        model, tokenizer, lm_dim, model_type = load_model(args.model, model_name, device)

        projection = Projection(audio_dim, lm_dim, prefix_len, dropout=dropout, depth=args.proj_depth, use_layernorm=use_layernorm).to(device)

        ckpt_dir = Path(cfg["checkpoint_dir"]) / (args.ckpt_tag or args.model)
        load_checkpoints(args.model, model, projection, args.stage, ckpt_dir)
        projection.eval()

        ds = load_from_disk(str(data_dir))
        split1 = ds.train_test_split(test_size=0.1, seed=seed)
        test_data = split1["test"]
        test_indices = test_data._indices.column("indices").to_pylist()
        test_clap = all_clap[test_indices]

        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            num_beams=args.num_beams,
        )
        if args.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = args.repetition_penalty
        if args.no_repeat_ngram_size > 0:
            gen_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
        if args.num_beams > 1:
            gen_kwargs["length_penalty"] = args.length_penalty
        if args.do_sample:
            gen_kwargs["temperature"] = args.temperature
            gen_kwargs["top_p"] = args.top_p

        print(f"Evaluating {args.model} stage {args.stage} on {len(test_data)} test samples")
        print(f"Generation config: {gen_kwargs}")

        references = []
        hypotheses = []
        failed = 0

        for i in tqdm(range(len(test_data)), desc="Generating"):
            try:
                audio_emb = test_clap[i].to(device)

                caption = generate_caption(
                    args.model, model, projection, tokenizer, audio_emb,
                    prefix_len, device, gen_kwargs,
                )

                if args.trim_incomplete:
                    caption = trim_incomplete_sentence(caption)

                references.append(test_data[i]["caption"])
                hypotheses.append(caption)
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Sample {i} failed: {e}")

        print(f"Generated {len(hypotheses)}/{len(test_data)} captions ({failed} failed)")

        predictions = [
            {"reference": ref, "hypothesis": hyp}
            for ref, hyp in zip(references, hypotheses)
        ]
        with open(pred_path, "w") as f:
            json.dump({
                "model": model_name,
                "stage": args.stage,
                "seed": seed,
                "num_samples": len(hypotheses),
                "num_failed": failed,
                "trim_incomplete": args.trim_incomplete,
                "gen_kwargs": gen_kwargs,
                "predictions": predictions,
            }, f, indent=2)
        print(f"Predictions cached to {pred_path}")

    print("Computing metrics...")
    metrics = compute_metrics(references, hypotheses)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    results = {
        "model": model_name,
        "stage": args.stage,
        "seed": seed,
        "num_samples": len(hypotheses),
        "num_failed": failed,
        "trim_incomplete": args.trim_incomplete,
        "gen_kwargs": gen_kwargs,
        "metrics": metrics,
        "predictions": predictions,
    }

    out_path = results_dir / f"{args.model}_ablation_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
