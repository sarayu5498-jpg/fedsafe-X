import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Maximize CPU thread utilization
torch.set_num_threads(os.cpu_count() or 4)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "backend"))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, BACKEND_DIR)

from run_revised_experiments import (
    load_fedsafe_cm, stratified_70_10_20,
    evaluate_predictions, paired_bootstrap_ci
)
from qkd_engine import BB84KeyAgreement
from federated_node import FederatedClientNode, TextDataset
from server_aggregator import ServerAggregator
from lime_explainer import LocalXAIExplainer
from forensic_generator import TamperEvidentEvidenceManager

OUT_FILE = os.path.join(RESULTS_DIR, "ml_xai_latency_results.json")


def save_checkpoint(data: dict):
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [CHECKPOINT SAVED] -> {OUT_FILE}", flush=True)


def freeze_lower_layers(model):
    """Freezes lower transformer layers so CPU fine-tuning updates only the top layer + classifier (~6x faster)."""
    for name, param in model.named_parameters():
        if "classifier" in name or "pre_classifier" in name or "layer.5" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def run_federated_k3_cpu(nodes_dict, model_name="distilbert-base-multilingual-cased", rounds=3, seed=0, use_masking=True, dp_sigma=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    global_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    freeze_lower_layers(global_model)

    aggregator = ServerAggregator(global_model, frac_bits=16)
    qkd = BB84KeyAgreement(num_qubits=4096)

    node_names = ["dating_app", "social_network", "gaming_lobby"]
    clients, test_splits = [], []
    for idx, name in enumerate(node_names):
        tr, _, te = stratified_70_10_20(nodes_dict[name], seed=seed)
        client = FederatedClientNode(idx, name, tr, model_name, tokenizer, max_len=64, batch_size=32, lr=5e-5)
        freeze_lower_layers(client.model)
        clients.append(client)
        test_splits.append(te)

    pairwise_keys = {i: {} for i in range(len(clients))}
    for i in range(len(clients)):
        for j in range(i + 1, len(clients)):
            k_ij, _ = qkd.negotiate_pairwise_key(node_names[i], node_names[j])
            pairwise_keys[i][j] = k_ij
            pairwise_keys[j][i] = k_ij

    total_N = sum(c.n_samples for c in clients)
    regime_label = f"Masking={use_masking}, DP={dp_sigma}"

    for r in range(1, rounds + 1):
        t_round = time.perf_counter()
        global_sd = global_model.state_dict()
        updates, shapes = [], None
        for c in clients:
            c.set_weights(global_sd)
            delta, _ = c.local_train(epochs=1)
            keys_arg = pairwise_keys[c.node_idx] if use_masking else {}
            vec, shapes = c.encode_and_mask_update(
                delta, total_N, keys_arg, round_t=r,
                clip_norm=1.0, frac_bits=16, dp_sigma=dp_sigma, num_clients_K=len(clients)
            )
            updates.append(vec)
        aggregator.aggregate_masked_updates(updates, shapes)
        print(f"  [{regime_label}] Round {r}/{rounds} finished in {time.perf_counter() - t_round:.1f}s", flush=True)

    global_model.eval()
    pooled_test = pd.concat(test_splits, ignore_index=True)
    test_ds = TextDataset(pooled_test["text"].tolist(), pooled_test["label"].tolist(), tokenizer, max_len=64)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

    probs = []
    with torch.no_grad():
        for batch in test_loader:
            logits = global_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            probs.extend(torch.softmax(logits, dim=-1)[:, 1].numpy().tolist())

    pooled_test["prob"] = probs
    return pooled_test, global_model, tokenizer


def benchmark_latency_and_lime_cpu(global_model, tokenizer, test_df):
    global_model.eval()

    n_lat = 100
    print(f"  Running CPU inference latency benchmark ({n_lat} messages)...", flush=True)
    sample_texts = (test_df["text"].tolist() * 2)[:n_lat]
    inf_times = []
    for txt in sample_texts:
        t0 = time.perf_counter()
        enc = tokenizer(txt, truncation=True, padding=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            _ = torch.softmax(global_model(**enc).logits, dim=-1)
        inf_times.append((time.perf_counter() - t0) * 1000.0)

    explainer = LocalXAIExplainer(global_model, tokenizer, max_len=64)
    evidence_mgr = TamperEvidentEvidenceManager(os.path.join(RESULTS_DIR, "test_evidence"), b"secret_platform_key")
    lime_times, hash_times, pdf_times, comps, suffs, examples = [], [], [], [], [], []

    n_lime = 5
    flagged_df = test_df[test_df["label"] == 1].head(n_lime).reset_index(drop=True)
    for idx, row in flagged_df.iterrows():
        t0 = time.perf_counter()
        res = explainer.explain_with_faithfulness(row["text"], top_k=5, num_samples=150, seed=0)
        dt_lime = (time.perf_counter() - t0) * 1000.0
        lime_times.append(dt_lime)
        comps.append(res["comprehensiveness"])
        suffs.append(res["sufficiency"])
        print(f"  LIME sample {idx + 1}/{n_lime} done ({dt_lime:.1f} ms)", flush=True)

        t1 = time.perf_counter()
        evidence_mgr.append_strike(f"user_{idx}", row.get("node", "social_network"), res["prob_toxic"], res["top_words"])
        hash_times.append((time.perf_counter() - t1) * 1000.0)

        if len(examples) < 3:
            examples.append({"language": row.get("language", "Mixed"), "text": row["text"][:80], "prob": res["prob_toxic"], "top": res["attributions"]})

    t2 = time.perf_counter()
    evidence_mgr.generate_signed_evidence_report("user_0")
    pdf_times.append((time.perf_counter() - t2) * 1000.0)

    return {
        "device_used": f"CPU ({os.cpu_count()} threads)",
        "table_12_lime": {
            "comprehensiveness_mean": float(np.mean(comps)),
            "sufficiency_mean": float(np.mean(suffs)),
        },
        "table_13_examples": examples,
        "table_14_latency_ms": {
            "inference_median": float(np.median(inf_times)),
            "inference_p95": float(np.percentile(inf_times, 95)),
            "lime_median": float(np.median(lime_times)),
            "lime_p95": float(np.percentile(lime_times, 95)),
            "hash_chain_median": float(np.median(hash_times)),
            "hash_chain_p95": float(np.percentile(hash_times, 95)),
            "pdf_sign_median": float(np.median(pdf_times)),
            "pdf_sign_p95": float(np.percentile(pdf_times, 95)),
        }
    }


if __name__ == "__main__":
    chosen_model = "distilbert-base-multilingual-cased"
    num_rounds = 3
    print(f"=== Fast CPU Execution ({os.cpu_count()} threads) | Model: {chosen_model} | Rounds: {num_rounds} ===", flush=True)

    fedsafe_nodes = load_fedsafe_cm()
    output = {"model_used": chosen_model, "rounds": num_rounds, "table_6_seed0": {}}

    print("\n--- [1/4] Training Unmasked FedAvg ---", flush=True)
    unmasked_df, _, _ = run_federated_k3_cpu(fedsafe_nodes, model_name=chosen_model, rounds=num_rounds, seed=0, use_masking=False)
    output["table_6_seed0"]["unmasked_fedavg"] = evaluate_predictions(unmasked_df["label"], unmasked_df["prob"])
    save_checkpoint(output)

    print("\n--- [2/4] Training FedSafe-X (BB84 + ChaCha20 Masking) ---", flush=True)
    masked_df, best_model, tok = run_federated_k3_cpu(fedsafe_nodes, model_name=chosen_model, rounds=num_rounds, seed=0, use_masking=True, dp_sigma=0.0)
    diff, ci, p_val = paired_bootstrap_ci(masked_df["label"], masked_df["prob"], unmasked_df["prob"])
    output["table_6_seed0"]["fedsafe_x"] = evaluate_predictions(masked_df["label"], masked_df["prob"])
    output["table_6_seed0"]["diff_vs_fedavg"] = diff
    output["table_6_seed0"]["ci_95"] = ci
    output["table_6_seed0"]["p_value"] = p_val
    save_checkpoint(output)

    print("\n--- [3/4] Training FedSafe-X + Distributed DP (sigma=0.5) ---", flush=True)
    dp_df, _, _ = run_federated_k3_cpu(fedsafe_nodes, model_name=chosen_model, rounds=num_rounds, seed=0, use_masking=True, dp_sigma=0.5)
    output["table_6_seed0"]["fedsafe_x_dp"] = evaluate_predictions(dp_df["label"], dp_df["prob"])
    save_checkpoint(output)

    print("\n--- [4/4] Running LIME ERASER & Latency Benchmarks ---", flush=True)
    xai_and_lat = benchmark_latency_and_lime_cpu(best_model, tok, masked_df)
    output.update(xai_and_lat)
    save_checkpoint(output)

    print("\n=== ALL STEPS COMPLETE! ===", flush=True)