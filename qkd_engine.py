import os
import hmac
import hashlib
import numpy as np
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


class KeyAgreementError(Exception):
    """Raised when QBER meets or exceeds the Shor-Preskill 11% eavesdropping threshold."""
    pass


class BB84KeyAgreement:
    """
    Pluggable Key-Agreement Interface instantiated with a software BB84 QKD simulation.
    Note: Provides protocol-logic validation and a drop-in interface for hardware QKD
    or post-quantum KEM (e.g., ML-KEM). Software simulation offers no physical security.
    """
    def __init__(self, num_qubits: int = 4096, sample_fraction: float = 0.25, qber_threshold: float = 0.11):
        self.num_qubits = num_qubits
        self.sample_fraction = sample_fraction
        self.qber_threshold = qber_threshold

    def negotiate_pairwise_key(
        self,
        client_i: str,
        client_j: str,
        channel_error_rate: float = 0.0,
        eavesdropper: bool = False,
        rng: np.random.Generator = None
    ) -> tuple[bytes, dict]:
        if rng is None:
            rng = np.random.default_rng()

        # 1. Alice prepares random bits in random bases (0: Rectilinear +, 1: Diagonal x)
        alice_bits = rng.integers(0, 2, size=self.num_qubits, dtype=np.uint8)
        alice_bases = rng.integers(0, 2, size=self.num_qubits, dtype=np.uint8)

        # Optional intercept-resend adversary (Eve)
        transmitted_bits = alice_bits.copy()
        if eavesdropper:
            eve_bases = rng.integers(0, 2, size=self.num_qubits, dtype=np.uint8)
            eve_mismatch = (eve_bases != alice_bases)
            transmitted_bits[eve_mismatch] = rng.integers(0, 2, size=int(np.sum(eve_mismatch)), dtype=np.uint8)

        # Channel bit-flip noise
        if channel_error_rate > 0.0:
            flips = rng.random(size=self.num_qubits) < channel_error_rate
            transmitted_bits ^= flips.astype(np.uint8)

        # 2. Bob measures in random bases
        bob_bases = rng.integers(0, 2, size=self.num_qubits, dtype=np.uint8)
        bob_bits = transmitted_bits.copy()
        bob_mismatch = (bob_bases != alice_bases)
        bob_bits[bob_mismatch] = rng.integers(0, 2, size=int(np.sum(bob_mismatch)), dtype=np.uint8)

        # 3. Basis sifting
        sifted_mask = (alice_bases == bob_bases)
        alice_sifted = alice_bits[sifted_mask]
        bob_sifted = bob_bits[sifted_mask]
        sifted_len = len(alice_sifted)

        # 4. QBER estimation on 25% random sample
        n_sampled = max(1, int(sifted_len * self.sample_fraction))
        sample_idx = rng.choice(sifted_len, size=n_sampled, replace=False)
        n_errors = int(np.sum(alice_sifted[sample_idx] != bob_sifted[sample_idx]))
        qber = n_errors / n_sampled

        aborted = bool(qber >= self.qber_threshold)
        telemetry = {
            "pair": f"{client_i}<->{client_j}",
            "transmitted_qubits": self.num_qubits,
            "sifted_bits": sifted_len,
            "sampled_bits": n_sampled,
            "qber": qber,
            "aborted": aborted
        }

        if aborted:
            raise KeyAgreementError(f"BB84 exchange aborted between {client_i} and {client_j}: QBER={qber:.4f} >= {self.qber_threshold}")

        # 5. Compress unsampled sifted bits via SHA-256 HKDF into 256-bit key k_{ij}
        unsampled_mask = np.ones(sifted_len, dtype=bool)
        unsampled_mask[sample_idx] = False
        raw_secret = np.packbits(alice_sifted[unsampled_mask]).tobytes()

        k_ij = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"FedSafeX-BB84-Seed",
            info=f"{min(client_i, client_j)}:{max(client_i, client_j)}".encode("utf-8"),
        ).derive(raw_secret)

        return k_ij, telemetry


def expand_chacha20_mask(k_ij: bytes, round_t: int, num_params: int) -> np.ndarray:
    """
    Equation (3): m_{ij}^{(t)} = ChaCha20(HKDF(k_{ij}, salt=t), 0^{4d}) in (Z_{2^32})^d
    Expands a 256-bit pairwise secret into a fresh uint32 mask vector of length num_params.
    """
    round_key = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=str(round_t).encode("utf-8"),
        info=b"FedSafeX-Round-Mask",
    ).derive(k_ij)

    nonce = b"\x00" * 16  # 16-byte nonce for cryptography.hazmat ChaCha20
    cipher = Cipher(algorithms.ChaCha20(round_key, nonce), mode=None)
    encryptor = cipher.encryptor()
    zero_stream = b"\x00" * (num_params * 4)
    keystream = encryptor.update(zero_stream) + encryptor.finalize()
    return np.frombuffer(keystream, dtype=np.uint32).copy()