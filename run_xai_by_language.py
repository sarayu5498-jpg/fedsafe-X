import os
import sys
import json
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(os.cpu_count() or 4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
sys.path.insert(0, BACKEND_DIR)

from run_revised_experiments import load_fedsafe_cm
from run_remaining_tables_cpu import train_federated_cpu
from lime_explainer import LocalXAIExplainer

OUT_XAI = os.path.join(RESULTS_DIR, "xai_by_language_tables_12_13.json")


def random_word_eraser(explainer, text: str, top_k: int = 5, seed: int = 0):
    """Random-word baseline for Table 12 comparison against LIME."""
    rng = np.random.default_rng(seed)
    words = text.split()
    base_prob = float(explainer.predict_probs([text])[0, 1])
    if len(words) <= 1:
        return 0.0, 0.0, set(words)
    k = min(top_k, len(words))
    chosen_idx = set(rng.choice(len(words), size=k, replace=False).tolist())
    chosen_words = {words[i] for i in chosen_idx}

    removed_text = " ".join(w for i, w in enumerate(words) if i not in chosen_idx) or ""
    only_text = " ".join(words[i] for i in sorted(chosen_idx))
    p_removed = float(explainer.predict_probs([removed_text])[0, 1])
    p_only = float(explainer.predict_probs([only_text])[0, 1])
    return base_prob - p_removed, base_prob - p_only, chosen_words


if __name__ == "__main__":
    fedsafe_nodes = load_fedsafe_cm()
    pooled_cm = pd.concat(fedsafe_nodes.values(), ignore_index=True)

    # Train FedSafe-X (K=3) once (~2 mins on CPU)
    test_df, model, tokenizer = train_federated_cpu(fedsafe_nodes, rounds=6, seed=0, use_masking=True)
    explainer = LocalXAIExplainer(model, tokenizer, max_len=64)

    table12 = {}
    table13 = []

    # 1. Random words baseline across flagged/toxic messages
    rand_comps, rand_suffs, rand_jaccs = [], [], []
    sample_pool = pooled_cm[pooled_cm["label"] == 1].head(12)
    for _, row in sample_pool.iterrows():
        c0, s0, w0 = random_word_eraser(explainer, row["text"], top_k=5, seed=0)
        _, _, w1 = random_word_eraser(explainer, row["text"], top_k=5, seed=1)
        rand_comps.append(c0)
        rand_suffs.append(s0)
        rand_jaccs.append(len(w0 & w1) / max(1, len(w0 | w1)))

    table12["Random words"] = {
        "comprehensiveness": f"{np.mean(rand_comps):.4f} ± {np.std(rand_comps):.4f}",
        "sufficiency": f"{np.mean(rand_suffs):.4f} ± {np.std(rand_suffs):.4f}",
        "jaccard": f"{np.mean(rand_jaccs):.4f} ± {np.std(rand_jaccs):.4f}",
    }

    # 2. Per-language LIME ERASER & Jaccard across English, Hinglish, and Tanglish
    for lang in ["English", "Hinglish", "Tanglish"]:
        lang_toxic = pooled_cm[(pooled_cm["language"] == lang) & (pooled_cm["label"] == 1)].copy()
        probs = explainer.predict_probs(lang_toxic["text"].tolist())[:, 1]
        lang_toxic["pred_prob"] = probs
        lang_toxic = lang_toxic.sort_values("pred_prob", ascending=False).reset_index(drop=True)

        comps, suffs, jaccs = [], [], []
        eval_subset = lang_toxic.head(6)
        for idx, row in eval_subset.iterrows():
            r0 = explainer.explain_with_faithfulness(row["text"], top_k=5, num_samples=200, seed=0)
            r1 = explainer.explain_with_faithfulness(row["text"], top_k=5, num_samples=200, seed=1)
            comps.append(r0["comprehensiveness"])
            suffs.append(r0["sufficiency"])
            w0, w1 = set(r0["top_words"]), set(r1["top_words"])
            jaccs.append(len(w0 & w1) / max(1, len(w0 | w1)))

            if idx == 0:
                table13.append({
                    "language": lang,
                    "text": row["text"],
                    "P_toxic": round(r0["prob_toxic"], 4),
                    "top_words": r0["attributions"],
                })

        table12[f"LIME, {lang}"] = {
            "comprehensiveness": f"{np.mean(comps):.4f} ± {np.std(comps):.4f}",
            "sufficiency": f"{np.mean(suffs):.4f} ± {np.std(suffs):.4f}",
            "jaccard": f"{np.mean(jaccs):.4f} ± {np.std(jaccs):.4f}",
        }
        print(f"Finished {lang}: {table12[f'LIME, {lang}']}", flush=True)

    with open(OUT_XAI, "w", encoding="utf-8") as f:
        json.dump({"table_12": table12, "table_13": table13}, f, indent=2)
    print(f"\nSaved language-stratified Tables 12 & 13 to {OUT_XAI}", flush=True)