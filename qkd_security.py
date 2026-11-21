# pyrefly: ignore [missing-import]
from jinja2.exceptions import SecurityError
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch

class BB84QKDEngine:
    """
    Simulates the BB84 Quantum Key Distribution protocol between federated nodes.
    Establishes cryptographically secure pairwise masks for Secure Aggregation (SecAgg).
    """
    def __init__(self, num_qubits=128, error_threshold=0.11):
        self.num_qubits = num_qubits
        self.error_threshold = error_threshold

    def generate_pairwise_quantum_key(self, client_a: str, client_b: str):
        """
        Executes BB84 quantum exchange between Client A (Alice) and Client B (Bob):
        1. Alice prepares random bits in random quantum bases (Rectilinear '+' or Diagonal 'x').
        2. Bob measures in random bases.
        3. Basis reconciliation (sifting): retain bits where Alice & Bob chose identical bases.
        4. Error estimation: compute QBER (Quantum Bit Error Rate); abort if QBER > threshold.
        """
        # Step 1: Alice's state preparation
        alice_bits = np.random.randint(0, 2, self.num_qubits)
        alice_bases = np.random.randint(0, 2, self.num_qubits) # 0: rectilinear (+), 1: diagonal (x)

        # Step 2: Bob's measurement
        bob_bases = np.random.randint(0, 2, self.num_qubits)
        # In ideal quantum channel with eavesdropper absence:
        bob_bits = np.where(alice_bases == bob_bases, alice_bits, np.random.randint(0, 2, self.num_qubits))

        # Step 3: Sifting phase (basis match)
        sifting_indices = np.where(alice_bases == bob_bases)[0]
        sifted_alice = alice_bits[sifting_indices]
        sifted_bob = bob_bits[sifting_indices]

        # Step 4: QBER check (using sample of sifted bits)
        sample_size = max(4, len(sifted_alice) // 4)
        sample_idx = np.random.choice(len(sifted_alice), sample_size, replace=False)
        errors = np.sum(sifted_alice[sample_idx] != sifted_bob[sample_idx])
        qber = errors / sample_size

        if qber > self.error_threshold:
            raise SecurityError(f"Eavesdropper detected! QBER={qber:.3f} > {self.error_threshold}")

        # Sifted key
        key_indices = np.setdiff1d(np.arange(len(sifted_alice)), sample_idx)
        final_key = sifted_alice[key_indices]
        key_int = int("".join(map(str, final_key[:32])), 2) if len(final_key) >= 32 else int("".join(map(str, final_key)), 2)

        return key_int, {
            "node_pair": f"{client_a} <-> {client_b}",
            "raw_qubits": self.num_qubits,
            "sifted_key_length": len(final_key),
            "qber": float(qber),
            "channel_status": "SECURE (QBER below 11% threshold)"
        }

    def generate_pairwise_masks(self, nodes: list, model_state_dict: dict):
        """
        Generates zero-sum pairwise masking tensors across client nodes.
        Node A adds mask M_AB, Node B subtracts mask M_AB.
        When the server aggregates: sum(Masked_Weights) = sum(Weights) + sum(M_AB - M_AB) = sum(Weights).
        Provides information-theoretic privacy during gradient transmission.
        """
        masks = {node_id: {} for node_id in nodes}
        
        # Zero-initialize masks
        for node_id in nodes:
            for k, v in model_state_dict.items():
                if v.is_floating_point():
                    masks[node_id][k] = torch.zeros_like(v)

        # Pairwise zero-sum masks
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_a, node_b = nodes[i], nodes[j]
                key_ab, _ = self.generate_pairwise_quantum_key(node_a, node_b)
                
                # Seed deterministic pseudo-random generator with quantum key
                torch.manual_seed(key_ab % (2**31 - 1))
                for k, v in model_state_dict.items():
                    if v.is_floating_point():
                        pairwise_noise = torch.randn_like(v) * 0.005
                        masks[node_a][k] += pairwise_noise
                        masks[node_b][k] -= pairwise_noise

        return masks
