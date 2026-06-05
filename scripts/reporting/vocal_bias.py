#!/usr/bin/env python3
"""Quantify vocal-presence and vocalist-gender behaviour of the captioners.

Three read-only analyses:
  1. Caption stats: per-model vocal-mention rates on instrumental vs vocal
     references, and a gender confusion matrix (reference vs hypothesis),
     from the baseline greedy prediction JSONs.
  2. Corpus prior: vocal-mention and gender-term frequencies over all
     MusicCaps captions in data/dataset.
  3. CLAP zero-shot probe: classify every precomputed audio embedding as
     vocal/instrumental and male/female via text-prompt cosine similarity,
     scored against caption-derived labels. Measures how much of each
     attribute the frozen audio embedding carries.

Prints all statistics and writes two LaTeX table fragments to
reports/tables/: vocal_bias.tex (per-model prediction stats) and
vocal_bias_context.tex (corpus prior + CLAP probe).
"""
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F

VOCAL = re.compile(r"\b(vocal|vocalist|vocals|singer|singing|sings|sung|voice|choir|chant)", re.I)
NOVOC = re.compile(r"\b(instrumental|no singer|no vocals?|no voice)", re.I)
FEM = re.compile(r"\b(female|woman|girl)\b", re.I)
MAL = re.compile(r"\b(male|man|boy)\b", re.I)


def prediction_stats(model):
    preds = json.load(open(f"results/{model}/predictions/{model}_predictions_{model}.json"))["predictions"]
    refs = [x["reference"] for x in preds]
    hyps = [x["hypothesis"] for x in preds]
    n = len(preds)

    ref_instr = [bool(NOVOC.search(r)) for r in refs]
    ref_vocal = [bool(VOCAL.search(r)) and not NOVOC.search(r) for r in refs]
    hyp_vocal = [bool(VOCAL.search(h)) for h in hyps]

    n_instr, n_voc = sum(ref_instr), sum(ref_vocal)
    fp = sum(1 for i in range(n) if ref_instr[i] and hyp_vocal[i])
    tp = sum(1 for i in range(n) if ref_vocal[i] and hyp_vocal[i])

    print(f"\n=== {model} (baseline greedy, n={n}) ===")
    print(f"instrumental refs: {n_instr}; hypothesis mentions vocals: {fp} ({fp / n_instr:.0%})")
    print(f"vocal refs: {n_voc}; hypothesis mentions vocals: {tp} ({tp / n_voc:.0%})")
    print(f"overall vocal-mention rate: {sum(hyp_vocal) / n:.0%}; "
          f"'male vocalist' phrase rate: {sum('male vocalist' in h for h in hyps) / n:.0%}")

    cm = {"ff": 0, "fm": 0, "mm": 0, "mf": 0}
    for r, h in zip(refs, hyps):
        rf, rm = bool(FEM.search(r)), bool(MAL.search(r))
        hf, hm = bool(FEM.search(h)), bool(MAL.search(h))
        if rf and not rm:
            if hf and not hm:
                cm["ff"] += 1
            elif hm and not hf:
                cm["fm"] += 1
        elif rm and not rf:
            if hm and not hf:
                cm["mm"] += 1
            elif hf and not hm:
                cm["mf"] += 1
    nf, nm = cm["ff"] + cm["fm"], cm["mm"] + cm["mf"]
    print(f"gender ref->hyp: female->female {cm['ff']}/{nf} ({cm['ff'] / nf:.0%}), "
          f"female->male {cm['fm']} | male->male {cm['mm']}/{nm} ({cm['mm'] / nm:.0%}), "
          f"male->female {cm['mf']}")
    return {
        "n": n,
        "instr_n": n_instr, "instr_fp": fp,
        "vocal_n": n_voc, "vocal_tp": tp,
        "vocal_rate": sum(hyp_vocal) / n,
        "male_vocalist_rate": sum("male vocalist" in h for h in hyps) / n,
        "cm": cm, "nf": nf, "nm": nm,
    }


def corpus_prior():
    from datasets import load_from_disk
    caps = load_from_disk("data/dataset")["caption"]
    n = len(caps)
    v = sum(bool(VOCAL.search(c)) for c in caps)
    f = sum(bool(FEM.search(c)) and not MAL.search(c) for c in caps)
    m = sum(bool(MAL.search(c)) and not FEM.search(c) for c in caps)
    mv = sum("male vocalist" in c for c in caps)
    print(f"\n=== corpus prior (n={n}) ===")
    print(f"vocal-mention {v / n:.0%}; male-only {m / n:.0%} vs female-only {f / n:.0%} "
          f"(ratio {m / f:.1f}:1); 'male vocalist' phrase {mv / n:.0%}")
    return {"n": n, "vocal": v / n, "male": m / n, "female": f / n,
            "ratio": m / f, "male_vocalist": mv / n}


def clap_probe():
    from datasets import load_from_disk
    from transformers import ClapModel, ClapProcessor

    emb = torch.load("data/clap_embeddings.pt", map_location="cpu", weights_only=False)
    caps = load_from_disk("data/dataset")["caption"]
    model = ClapModel.from_pretrained("laion/clap-htsat-unfused").eval()
    proc = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
    audio = F.normalize(emb.float(), dim=-1)

    def text_emb(prompts):
        tok = proc(text=prompts, return_tensors="pt", padding=True)
        with torch.no_grad():
            out = model.get_text_features(**tok)
        return F.normalize(out.pooler_output if hasattr(out, "pooler_output") else out, dim=-1)

    print("\n=== CLAP zero-shot probe (all clips) ===")

    t = text_emb(["music with singing and vocals", "instrumental music without any vocals"])
    pred = (audio @ t.T).argmax(1) == 0
    lab = [1 if (VOCAL.search(c) and not NOVOC.search(c)) else 0 if NOVOC.search(c) else -1 for c in caps]
    idx = [i for i, l in enumerate(lab) if l >= 0]
    acc = sum(int(pred[i]) == lab[i] for i in idx) / len(idx)
    rv = sum(int(pred[i]) == 1 and lab[i] == 1 for i in idx) / sum(lab[i] == 1 for i in idx)
    ri = sum(int(pred[i]) == 0 and lab[i] == 0 for i in idx) / sum(lab[i] == 0 for i in idx)
    print(f"vocal/instrumental: n={len(idx)} acc={acc:.0%} recall(vocal)={rv:.0%} recall(instr)={ri:.0%}")
    vi = {"n": len(idx), "acc": acc, "recall_vocal": rv, "recall_instr": ri}

    t = text_emb(["a song with a male singer", "a song with a female singer"])
    pred = (audio @ t.T).argmax(1) == 0
    lab = [1 if (MAL.search(c) and not FEM.search(c)) else 0 if (FEM.search(c) and not MAL.search(c)) else -1
           for c in caps]
    idx = [i for i, l in enumerate(lab) if l >= 0]
    acc = sum(int(pred[i]) == lab[i] for i in idx) / len(idx)
    rm = sum(int(pred[i]) == 1 and lab[i] == 1 for i in idx) / sum(lab[i] == 1 for i in idx)
    rf = sum(int(pred[i]) == 0 and lab[i] == 0 for i in idx) / sum(lab[i] == 0 for i in idx)
    print(f"male/female: n={len(idx)} acc={acc:.0%} recall(male)={rm:.0%} recall(female)={rf:.0%}")
    return {"vi": vi, "gender": {"n": len(idx), "acc": acc, "recall_male": rm, "recall_female": rf}}


def write_latex(stats, prior, probe, out_dir="reports/tables"):
    """Emit booktabs fragments: per-model bias table + context (prior/probe) table."""
    def pct(x):
        return f"{x:.0%}".replace("%", r"\%")

    models = list(stats)
    names = {"gpt2": "GPT-2", "t5": "T5"}
    header = " & ".join(names.get(m, m) for m in models)
    rows = [
        ("Vocal mention on vocal reference",
         lambda s: f"{s['vocal_tp']}/{s['vocal_n']} ({pct(s['vocal_tp'] / s['vocal_n'])})"),
        ("Vocal mention on instrumental reference",
         lambda s: f"{s['instr_fp']}/{s['instr_n']} ({pct(s['instr_fp'] / s['instr_n'])})"),
        ("Overall vocal-mention rate",
         lambda s: pct(s["vocal_rate"])),
        ("``male vocalist'' phrase rate",
         lambda s: pct(s["male_vocalist_rate"])),
        ("Female ref.\\ $\\rightarrow$ female caption",
         lambda s: f"{s['cm']['ff']}/{s['nf']} ({pct(s['cm']['ff'] / s['nf'])})"),
        ("Female ref.\\ $\\rightarrow$ male caption",
         lambda s: f"{s['cm']['fm']}/{s['nf']} ({pct(s['cm']['fm'] / s['nf'])})"),
        ("Male ref.\\ $\\rightarrow$ male caption",
         lambda s: f"{s['cm']['mm']}/{s['nm']} ({pct(s['cm']['mm'] / s['nm'])})"),
        ("Male ref.\\ $\\rightarrow$ female caption",
         lambda s: f"{s['cm']['mf']}/{s['nm']} ({pct(s['cm']['mf'] / s['nm'])})"),
    ]
    n = stats[models[0]]["n"]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Vocal-presence and vocalist-gender statistics per model "
        rf"(baseline greedy predictions, $n={n}$ test clips). Gender rows count only "
        r"captions that mention exactly one gender.}",
        r"\label{tab:vocal_bias}",
        r"\footnotesize",
        r"\begin{tabular}{l" + "r" * len(models) + "}",
        r"\toprule",
        rf" & {header} \\",
        r"\midrule",
    ]
    for name, fmt in rows:
        cells = " & ".join(fmt(stats[m]) for m in models)
        lines.append(rf"{name} & {cells} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(out_dir, "vocal_bias.tex").write_text("\n".join(lines) + "\n")

    vi, ge = probe["vi"], probe["gender"]
    ctx = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Model-independent context for the vocalist bias: training-caption "
        r"prior over all MusicCaps captions, and zero-shot probes of the frozen "
        r"\textsc{Clap} audio embeddings against caption-derived labels.}",
        r"\label{tab:vocal_bias_context}",
        r"\footnotesize",
        r"\begin{tabular}{llr}",
        r"\toprule",
        rf"Corpus prior ($n={prior['n']}$ captions) & Vocal mention & {pct(prior['vocal'])} \\",
        rf" & Male-only gender mention & {pct(prior['male'])} \\",
        rf" & Female-only gender mention & {pct(prior['female'])} \\",
        rf" & Male:female ratio & {prior['ratio']:.1f}:1 \\",
        rf" & ``male vocalist'' phrase & {pct(prior['male_vocalist'])} \\",
        r"\midrule",
        rf"\textsc{{Clap}} probe: vocal/instr.\ ($n={vi['n']}$) & Accuracy & {pct(vi['acc'])} \\",
        rf" & Recall (vocal) & {pct(vi['recall_vocal'])} \\",
        rf" & Recall (instrumental) & {pct(vi['recall_instr'])} \\",
        r"\midrule",
        rf"\textsc{{Clap}} probe: gender ($n={ge['n']}$) & Accuracy & {pct(ge['acc'])} \\",
        rf" & Recall (male) & {pct(ge['recall_male'])} \\",
        rf" & Recall (female) & {pct(ge['recall_female'])} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    Path(out_dir, "vocal_bias_context.tex").write_text("\n".join(ctx) + "\n")
    print(f"\nwrote {out_dir}/vocal_bias.tex and {out_dir}/vocal_bias_context.tex")


if __name__ == "__main__":
    stats = {m: prediction_stats(m) for m in ["gpt2", "t5"]}
    prior = corpus_prior()
    probe = clap_probe()
    write_latex(stats, prior, probe)
