import re
import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer


class LocalXAIExplainer:
    """
    Computes LIME token attributions (M=500) and evaluates ERASER faithfulness
    (Comprehensiveness and Sufficiency) + multi-seed Jaccard stability.
    """
    def __init__(self, model, tokenizer, max_len: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.device = next(model.parameters()).device
        self.model.eval()
        self.explainer = LimeTextExplainer(class_names=["Benign", "Harassing"])

    def predict_probs(self, texts: list[str]) -> np.ndarray:
        all_probs = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i:i + batch_size])
            inputs = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    def explain_with_faithfulness(self, text: str, top_k: int = 5, num_samples: int = 500, seed: int = 0) -> dict:
        self.explainer.random_state = np.random.RandomState(seed)
        orig_prob = float(self.predict_probs([text])[0, 1])

        exp = self.explainer.explain_instance(
            text, self.predict_probs, num_features=top_k, num_samples=num_samples
        )
        word_weights = exp.as_list()[:top_k]
        top_words = [w for w, _ in word_weights]

        # Equation (8): ERASER Comprehensiveness & Sufficiency
        tokens = re.findall(r"\w+|\S", text)
        top_set = set(w.lower() for w in top_words)

        removed_text = " ".join([t for t in tokens if t.lower() not in top_set]) or "[EMPTY]"
        only_top_text = " ".join([t for t in tokens if t.lower() in top_set]) or "[EMPTY]"

        prob_without_rk = float(self.predict_probs([removed_text])[0, 1])
        prob_only_rk = float(self.predict_probs([only_top_text])[0, 1])

        comprehensiveness = orig_prob - prob_without_rk
        sufficiency = orig_prob - prob_only_rk

        return {
            "prob_toxic": orig_prob,
            "attributions": [{"word": w, "weight": float(s)} for w, s in word_weights],
            "top_words": top_words,
            "comprehensiveness": comprehensiveness,
            "sufficiency": sufficiency
        }