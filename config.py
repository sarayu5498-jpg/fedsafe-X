import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset File Paths
DATING_APP_CSV = os.path.join(BASE_DIR, "dating_app.csv")
GAMING_LOBBY_CSV = os.path.join(BASE_DIR, "gaming_lobby.csv")
SOCIAL_NETWORK_CSV = os.path.join(BASE_DIR, "social_network.csv")

# Model, Evidence & Ledger File Paths
SAVED_MODEL_PATH = os.path.join(BASE_DIR, "fedsafe_x_global_model.pt")
REPORTS_DIR = os.path.join(BASE_DIR, "forensic_reports")
JSON_LEDGER_PATH = os.path.join(BASE_DIR, "blocked_ledger.json")
EXCEL_LEDGER_PATH = os.path.join(BASE_DIR, "user_warning_ledger.xlsx")
WORD_TOXICITY_EXCEL_PATH = os.path.join(BASE_DIR, "word_toxicity_ledger.xlsx")

os.makedirs(REPORTS_DIR, exist_ok=True)

# Upgraded Multi-Lingual Transformer Architecture
MODEL_NAME = "distilbert-base-multilingual-cased"
HARASSMENT_THRESHOLD = 0.65       # Standard toxicity threshold (65%)
MUTE_TOXICITY_THRESHOLD = 0.85    # High-severity threshold for 1-minute mute (85%)
MAX_WARNINGS_BEFORE_REPORT = 10   # Block threshold

# Load .env file if present
ENV_PATH = os.path.join(BASE_DIR, ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

# Admin Configuration
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "manokothandapani2004@gmail.com")

# SMTP Email Configuration (Requires 16-character Google App Password from https://myaccount.google.com/apppasswords)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SENDER_EMAIL = os.getenv("SENDER_EMAIL", "sarayuthellakula@gmail.com")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD", "cnju jkyt ewjp cefp")
