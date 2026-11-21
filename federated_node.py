import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from qkd_engine import expand_chacha20_mask


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(self.labels[idx]), dtype=torch.long)
        }


class FederatedClientNode:
    def __init__(self, node_idx: int, node_id: str, train_df: pd.DataFrame, model_name: str, tokenizer, max_len: int = 128, batch_size: int = 16, lr: float = 2e-5):
        self.node_idx = node_idx
        self.node_id = node_id
        self.n_samples = len(train_df)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = TextDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_len=max_len)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(self.device)
        self.lr = lr

    def set_weights(self, global_state_dict: dict):
        self.model.load_state_dict({k: v.to(self.device) for k, v in global_state_dict.items()})

    def local_train(self, epochs: int = 2, weight_decay: float = 0.01) -> tuple[dict, float]:
        """Trains locally for E epochs and returns parameter update delta_theta_i = theta_i - theta_t."""
        initial_weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items() if v.is_floating_point()}
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        total_steps = len(self.dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == "cuda"))

        self.model.train()
        total_loss = 0.0
        for _ in range(epochs):
            for batch in self.dataloader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                total_loss += loss.item()

        delta_theta = {}
        for k, v in self.model.state_dict().items():
            if v.is_floating_point():
                delta_theta[k] = v.detach().cpu() - initial_weights[k]

        avg_loss = total_loss / max(1, total_steps)
        return delta_theta, avg_loss

    def encode_and_mask_update(
        self,
        delta_theta: dict,
        total_samples_N: int,
        pairwise_keys: dict,
        round_t: int,
        clip_norm: float = 1.0,
        frac_bits: int = 16,
        dp_sigma: float = 0.0,
        num_clients_K: int = 3
    ) -> tuple[np.ndarray, list]:
        """
        Implements Equations (4) and (5):
        1. Clip L2 norm <= C
        2. Weight by w_i = n_i / N
        3. Optional Distributed DP Gaussian noise N(0, sigma^2 C^2 / K)
        4. Fixed-point encode mod 2^32
        5. Add pairwise ChaCha20 masks for j > i and subtract for j < i (mod 2^32)
        """
        shapes = [(k, delta_theta[k].shape, delta_theta[k].numel()) for k in sorted(delta_theta.keys())]
        flat_update = np.concatenate([delta_theta[k].numpy().ravel().astype(np.float64) for k, _, _ in shapes])

        # L2 norm clipping
        l2_norm = np.linalg.norm(flat_update)
        if clip_norm > 0 and l2_norm > clip_norm:
            flat_update = flat_update * (clip_norm / l2_norm)

        # Optional Distributed DP noise
        if dp_sigma > 0.0:
            noise_std = (dp_sigma * clip_norm) / np.sqrt(num_clients_K)
            flat_update = flat_update + np.random.normal(0.0, noise_std, size=flat_update.shape)
        
        # Sample weighting w_i = n_i / N and fixed-point encoding (mod 2^32)
        w_i = self.n_samples / total_samples_N
        flat_update *= (2 ** frac_bits) * w_i
        np.rint(flat_update, out=flat_update)
        y_i = flat_update.astype(np.int32).view(np.uint32)
        del flat_update

        # Pairwise additive masking mod 2^32 (in-place uint32 wraps mod 2^32 natively)
        num_params = len(y_i)
        for j_idx, k_ij in pairwise_keys.items():
            if j_idx == self.node_idx:
                continue
            mask = expand_chacha20_mask(k_ij, round_t, num_params)
            if j_idx > self.node_idx:
                y_i += mask
            else:
                y_i -= mask
            del mask

        return y_i, shapes