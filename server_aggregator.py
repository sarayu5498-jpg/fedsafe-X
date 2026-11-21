import numpy as np
import torch


class ServerAggregator:
    """
    Aggregates pairwise-masked uint32 vectors mod 2^32.
    All pairwise masks cancel identically; the server recovers only the weighted sum.
    """
    def __init__(self, global_model, frac_bits: int = 16):
        self.global_model = global_model
        self.frac_bits = frac_bits

    def aggregate_masked_updates(self, masked_vectors: list[np.ndarray], shapes: list) -> dict:
        # Sum masked vectors modulo 2^32
        total_mod = np.zeros_like(masked_vectors[0], dtype=np.uint32)
        for y_i in masked_vectors:
            total_mod += y_i

        # Convert unsigned uint32 back to signed two's-complement int32
        signed_sum = total_mod.astype(np.uint32).view(np.int32).astype(np.float64)
        decoded_delta = signed_sum * (2.0 ** (-self.frac_bits))

        # Apply global update: theta_{t+1} = theta_t + sum(w_i * delta_theta_i)
        current_state = self.global_model.state_dict()
        offset = 0
        for k, shape, numel in shapes:
            chunk = decoded_delta[offset:offset + numel].reshape(shape)
            current_state[k] = current_state[k].cpu() + torch.from_numpy(chunk).to(current_state[k].dtype)
            offset += numel

        self.global_model.load_state_dict(current_state)
        return self.global_model.state_dict()