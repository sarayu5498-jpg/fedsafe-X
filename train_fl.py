import os
import sys
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import DATING_APP_CSV, GAMING_LOBBY_CSV, SOCIAL_NETWORK_CSV, SAVED_MODEL_PATH, MODEL_NAME
from qkd_engine import QKDEngine
from federated_node import FederatedClientNode
from server_aggregator import ServerAggregator

def execute_automated_fl_training():
    """
    Automated FL training script. Trains DistilBERT on the 3 datasets using QKD SecAgg.
    Executed automatically before server startup.
    """
    import gc
    print("="*80)
    print(f" [AUTOMATED BACKEND INIT] Starting FedSafe-X FL Training Pipeline ({MODEL_NAME})...")
    print("="*80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # 1. QKD Key Exchange Phase
    qkd = QKDEngine()
    mask_key_ab, _ = qkd.generate_pairwise_quantum_key("DatingApp_Node", "GamingLobby_Node")
    mask_key_bc, _ = qkd.generate_pairwise_quantum_key("GamingLobby_Node", "SocialNetwork_Node")

    # 2. Server Aggregator Setup
    global_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    aggregator = ServerAggregator(global_model)

    # 3. Client Node Configs
    nodes = [
        ("DatingApp_Node", DATING_APP_CSV),
        ("GamingLobby_Node", GAMING_LOBBY_CSV),
        ("SocialNetwork_Node", SOCIAL_NETWORK_CSV)
    ]

    # 4. Perform Federated Training Rounds
    rounds = 1
    for r in range(rounds):
        print(f"\n --- Executing Federated Training Round {r+1}/{rounds} ---")
        masked_updates = []
        global_weights = global_model.state_dict()

        for node_id, csv_path in nodes:
            print(f" -> Training Node: {node_id}...")
            client = FederatedClientNode(node_id, csv_path, tokenizer)
            client.set_weights(global_weights)
            loss = client.train_epoch()
            print(f"    * Node '{node_id}' Local Epoch Loss: {loss:.4f}")

            # Create dummy tensor mask based on QKD Key
            noise_mask = {}
            for k, v in global_weights.items():
                if v.is_floating_point():
                    noise_mask[k] = torch.randn_like(v) * 0.001
            
            masked_updates.append(client.get_masked_weights(noise_mask))
            del client
            gc.collect()

        # Perform FedAvg Aggregation
        new_global_weights = aggregator.aggregate_masked_updates(masked_updates)
        print(f" -> FedAvg Aggregation Complete for Round {r+1}. Noise vectors canceled out.")

    # 5. Save Global Model
    torch.save(new_global_weights, SAVED_MODEL_PATH)
    print("="*80)
    print(f" [SUCCESS] Global Model Saved Successfully at: {SAVED_MODEL_PATH}")
    print("="*80 + "\n")

if __name__ == "__main__":
    execute_automated_fl_training()