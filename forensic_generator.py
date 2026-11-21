import os
import hmac
import json
import hashlib
from datetime import datetime, timezone
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from fpdf import FPDF


class TamperEvidentEvidenceManager:
    """
    Maintains pseudonymised per-user strike logs using HMAC-SHA256,
    chains entries via SHA-256 (Eq. 9), and signs reports with Ed25519 (Eq. 10).
    """
    def __init__(self, reports_dir: str, platform_hmac_key: bytes, signing_key: Ed25519PrivateKey = None):
        self.reports_dir = reports_dir
        os.makedirs(self.reports_dir, exist_ok=True)
        self.platform_hmac_key = platform_hmac_key
        self.signing_key = signing_key or Ed25519PrivateKey.generate()
        self.user_chains: dict[str, list[dict]] = {}

    def pseudonymize_user(self, raw_user_id: str) -> str:
        return hmac.new(self.platform_hmac_key, raw_user_id.encode("utf-8"), hashlib.sha256).hexdigest()

    def append_strike(self, raw_user_id: str, platform_node: str, toxicity_score: float, top_k_words: list[str]) -> dict:
        pid = self.pseudonymize_user(raw_user_id)
        chain = self.user_chains.setdefault(pid, [])
        prev_hash = chain[-1]["hash"] if chain else ("0" * 64)
        timestamp_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        tokens_str = ",".join(top_k_words)

        # Equation (9): h_j = SHA-256(h_{j-1} || pid || timestamp_j || score_j || tokens_j)
        payload = f"{prev_hash}|{pid}|{timestamp_utc}|{toxicity_score:.6f}|{tokens_str}".encode("utf-8")
        curr_hash = hashlib.sha256(payload).hexdigest()

        entry = {
            "strike_number": len(chain) + 1,
            "pid": pid,
            "platform_node": platform_node,
            "timestamp_utc": timestamp_utc,
            "toxicity_score": round(toxicity_score, 4),
            "top_lime_tokens": top_k_words,
            "prev_hash": prev_hash,
            "hash": curr_hash
        }
        chain.append(entry)
        return entry

    def generate_signed_evidence_report(self, raw_user_id: str) -> dict:
        pid = self.pseudonymize_user(raw_user_id)
        chain = self.user_chains.get(pid, [])
        if not chain:
            raise ValueError("No strikes recorded for user.")

        final_hash_hex = chain[-1]["hash"]
        # Equation (10): sigma = Sign_{sk_platform}(h_n)
        signature_bytes = self.signing_key.sign(bytes.fromhex(final_hash_hex))
        signature_hex = signature_bytes.hex()
        pub_key_hex = self.signing_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ).hex()

        report_id = f"EVID-{pid[:12].upper()}"
        report_payload = {
            "report_id": report_id,
            "pseudonymous_user_id": pid,
            "total_strikes": len(chain),
            "final_chain_hash_sha256": final_hash_hex,
            "ed25519_signature": signature_hex,
            "platform_public_key_ed25519": pub_key_hex,
            "strike_log": chain,
            "privacy_notice": "Raw message text and direct personal identifiers are omitted in compliance with GDPR, CCPA, and India DPDP Act 2023."
        }

        json_path = os.path.join(self.reports_dir, f"{report_id}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2)

        pdf_path = os.path.join(self.reports_dir, f"{report_id}.pdf")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "FEDSAFE-X TAMPER-EVIDENT MODERATION EVIDENCE RECORD", ln=True, align="C")
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 7, f"Report ID: {report_id}", ln=True)
        pdf.cell(0, 7, f"Pseudonymous ID (HMAC-SHA256): {pid}", ln=True)
        pdf.cell(0, 7, f"Total Strikes Logged: {len(chain)} (Queued for Human Moderator Review)", ln=True)
        pdf.cell(0, 7, f"Final Chain Hash (h_n): {final_hash_hex}", ln=True)
        pdf.multi_cell(0, 6, f"Ed25519 Signature: {signature_hex}")
        pdf.ln(4)
        for e in chain:
            pdf.cell(0, 6, f"Strike #{e['strike_number']} | {e['timestamp_utc']} | Node: {e['platform_node']} | P={e['toxicity_score']:.3f} | Tokens: {e['top_lime_tokens']}", ln=True)
        pdf.output(pdf_path)

        return report_payload