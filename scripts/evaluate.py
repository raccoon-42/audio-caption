"""Shared evaluation script for all audio captioning models."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from fense.evaluator import Evaluator
from rouge_score import rouge_scorer
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

import nltk
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("omw-1.4", quiet=True)

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


def compute_cider(references, hypotheses, n=4):
    """CIDEr-D metric."""
    doc_freq = Counter()
    for ref in references:
        tokens = ref.lower().split()
        ngram_set = set()
        for k in range(1, n + 1):
            for i in range(len(tokens) - k + 1):
                ngram_set.add(tuple(tokens[i:i + k]))
        doc_freq.update(ngram_set)
    num_docs = len(references)

    def get_ngrams(text):
        tokens = text.lower().split()
        ngrams = {}
        for k in range(1, n + 1):
            counts = Counter()
            for i in range(len(tokens) - k + 1):
                counts[tuple(tokens[i:i + k])] += 1
            ngrams[k] = counts
        return ngrams, len(tokens)

    def tfidf_vec(ngrams, length, k):
        vec = {}
        denom = max(length - k + 1, 1)
        for ng, count in ngrams[k].items():
            tf = count / denom
            idf = math.log(max(num_docs, 1.0) / max(doc_freq.get(ng, 0), 1.0))
            vec[ng] = tf * idf
        return vec

    def cosine(v1, v2):
        common = set(v1) & set(v2)
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        n1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        n2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        return dot / (n1 * n2) if n1 > 0 and n2 > 0 else 0.0

    scores = []
    for ref, hyp in zip(references, hypotheses):
        ref_ng, ref_len = get_ngrams(ref)
        hyp_ng, hyp_len = get_ngrams(hyp)
        score = 0.0
        for k in range(1, n + 1):
            ref_vec = tfidf_vec(ref_ng, ref_len, k)
            hyp_vec = tfidf_vec(hyp_ng, hyp_len, k)
            score += cosine(hyp_vec, ref_vec)
        scores.append(score / n)

    return float(np.mean(scores) * 10)


def compute_metrics(references, hypotheses):
    refs_tokenized = [[ref.lower().split()] for ref in references]
    hyps_tokenized = [hyp.lower().split() for hyp in hypotheses]

    smooth = SmoothingFunction().method1
    bleu1 = corpus_bleu(refs_tokenized, hyps_tokenized,
                        weights=(1.0, 0, 0, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(refs_tokenized, hyps_tokenized,
                        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        score = nltk_meteor([ref.lower().split()], hyp.lower().split())
        meteor_scores.append(score)
    meteor = float(np.mean(meteor_scores))

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = []
    for ref, hyp in zip(references, hypotheses):
        score = scorer.score(ref, hyp)
        rouge_scores.append(score["rougeL"].fmeasure)
    rouge_l = float(np.mean(rouge_scores))

    cider = compute_cider(references, hypotheses)

    fense_eval = Evaluator(device="cuda" if torch.cuda.is_available() else "cpu")
    list_refs = [[ref] for ref in references]
    fense = float(fense_eval.corpus_score(hypotheses, list_refs, agg_score="mean"))

    return {
        "BLEU-1": float(bleu1),
        "BLEU-4": float(bleu4),
        "METEOR": meteor,
        "ROUGE-L": rouge_l,
        "CIDEr": cider,
        "FENSE": fense,
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
    parser.add_argument("--ckpt-tag", type=str, default=None,
                        help="Checkpoint dir tag, e.g. 'gpt2_prefix4'")
    parser.add_argument("--ablation-tag", type=str, default=None,
                        help="Tag for output filename, e.g. 'rep_1.2'")
    args = parser.parse_args()

    config_path = args.config or f"configs/{args.model}.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    prefix_len = args.prefix_len or cfg["prefix_len"]
    dropout = args.dropout if args.dropout is not None else cfg["dropout"]

    seed = cfg["seed"]
    set_seed(seed)
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
    # 1. Split off test set (10%) - this matches dataset.py
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

            references.append(test_data[i]["caption"])
            hypotheses.append(caption)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  Sample {i} failed: {e}")

    print(f"Generated {len(hypotheses)}/{len(test_data)} captions ({failed} failed)")

    print("Computing metrics...")
    metrics = compute_metrics(references, hypotheses)

    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    results_dir = Path(cfg["results_dir"]) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "model": model_name,
        "stage": args.stage,
        "seed": seed,
        "num_samples": len(hypotheses),
        "num_failed": failed,
        "gen_kwargs": gen_kwargs,
        "metrics": metrics,
        "predictions": [
            {"reference": ref, "hypothesis": hyp}
            for ref, hyp in zip(references, hypotheses)
        ],
    }

    if args.ablation_tag:
        out_path = results_dir / f"{args.model}_ablation_{args.ablation_tag}.json"
    else:
        out_path = results_dir / f"{args.model}_eval_stage{args.stage}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
__name__ == "__main__":
    main()
   main()
