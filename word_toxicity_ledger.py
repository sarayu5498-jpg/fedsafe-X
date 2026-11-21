import os
import sys
import re
import threading
import openpyxl
from datetime import datetime, timezone, timedelta
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from config import WORD_TOXICITY_EXCEL_PATH, HARASSMENT_THRESHOLD, MUTE_TOXICITY_THRESHOLD

class WordToxicityLedger:
    """
    Manages persistent word-level and text toxicity records in a dedicated Excel file (word_toxicity_ledger.xlsx).
    Provides two sheets:
      1. 'All_Words_Log': Chronological log of every word processed from every chat message.
      2. 'Unique_Words_Summary': Aggregated summary of unique words, their toxicity score, risk level, and frequency.
    """
    def __init__(self, excel_path: str = WORD_TOXICITY_EXCEL_PATH):
        self.excel_path = excel_path
        self.lock = threading.Lock()
        self._initialize_excel()

    @staticmethod
    def get_full_ist_timestamp() -> str:
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M:%S IST")

    @staticmethod
    def get_risk_label(score: float) -> str:
        if score >= MUTE_TOXICITY_THRESHOLD:
            return "HIGH RISK (MUTE >=85%)"
        elif score >= HARASSMENT_THRESHOLD:
            return "WARNING (>=65%)"
        return "SAFE (<65%)"

    def _initialize_excel(self):
        with self.lock:
            if not os.path.exists(self.excel_path):
                try:
                    wb = openpyxl.Workbook()
                    
                    # Sheet 1: Detailed chronological log
                    ws_log = wb.active
                    ws_log.title = "All_Words_Log"
                    ws_log.append([
                        "Timestamp IST",
                        "Sender Username",
                        "Sender Email",
                        "Sender Mobile",
                        "Original Message",
                        "Word",
                        "Toxicity Score",
                        "Toxicity (%)",
                        "Word Risk Level",
                        "Message Overall Toxicity (%)"
                    ])

                    # Sheet 2: Unique word summary
                    ws_summary = wb.create_sheet(title="Unique_Words_Summary")
                    ws_summary.append([
                        "Word",
                        "Toxicity Score",
                        "Toxicity (%)",
                        "Word Risk Level",
                        "Total Frequency",
                        "Last Seen IST"
                    ])

                    wb.save(self.excel_path)
                    print(f" [WORD TOXICITY LEDGER INITIALIZED]: Created {self.excel_path}")
                except Exception as e:
                    print(f" [WORD TOXICITY LEDGER INIT ERROR]: {str(e)}")

    def log_words_toxicity(
        self,
        sender_username: str,
        sender_email: str,
        sender_mobile: str,
        original_message: str,
        word_scores: list,  # List of tuples: [(word, score), ...]
        message_toxicity_prob: float
    ):
        """
        Logs each word and its calculated toxicity to the Excel ledger.
        """
        if not word_scores:
            return

        timestamp = self.get_full_ist_timestamp()
        msg_pct_str = f"{message_toxicity_prob * 100:.2f}%"

        with self.lock:
            try:
                if not os.path.exists(self.excel_path):
                    wb = openpyxl.Workbook()
                    ws_log = wb.active
                    ws_log.title = "All_Words_Log"
                    ws_log.append([
                        "Timestamp IST", "Sender Username", "Sender Email", "Sender Mobile",
                        "Original Message", "Word", "Toxicity Score", "Toxicity (%)",
                        "Word Risk Level", "Message Overall Toxicity (%)"
                    ])
                    ws_summary = wb.create_sheet(title="Unique_Words_Summary")
                    ws_summary.append([
                        "Word", "Toxicity Score", "Toxicity (%)", "Word Risk Level",
                        "Total Frequency", "Last Seen IST"
                    ])
                else:
                    wb = openpyxl.load_workbook(self.excel_path)

                ws_log = wb["All_Words_Log"] if "All_Words_Log" in wb.sheetnames else wb.create_sheet("All_Words_Log")
                ws_summary = wb["Unique_Words_Summary"] if "Unique_Words_Summary" in wb.sheetnames else wb.create_sheet("Unique_Words_Summary")

                # Read existing summary rows into memory for quick update
                summary_data = {}
                for row_idx in range(2, ws_summary.max_row + 1):
                    word_val = ws_summary.cell(row=row_idx, column=1).value
                    if word_val:
                        w_key = str(word_val).strip().lower()
                        freq_val = ws_summary.cell(row=row_idx, column=5).value
                        try:
                            freq = int(freq_val) if freq_val is not None else 1
                        except (ValueError, TypeError):
                            freq = 1
                        summary_data[w_key] = row_idx

                for word, score in word_scores:
                    clean_word = str(word).strip()
                    if not clean_word:
                        continue

                    w_score = float(score)
                    w_pct = f"{w_score * 100:.2f}%"
                    risk_label = self.get_risk_label(w_score)

                    # Append to chronological log
                    ws_log.append([
                        timestamp,
                        sender_username,
                        sender_email,
                        sender_mobile,
                        original_message,
                        clean_word,
                        round(w_score, 4),
                        w_pct,
                        risk_label,
                        msg_pct_str
                    ])

                    # Update or append to Unique_Words_Summary
                    w_key = clean_word.lower()
                    if w_key in summary_data:
                        row_idx = summary_data[w_key]
                        current_freq_cell = ws_summary.cell(row=row_idx, column=5)
                        try:
                            current_freq = int(current_freq_cell.value) if current_freq_cell.value is not None else 1
                        except (ValueError, TypeError):
                            current_freq = 1
                        current_freq_cell.value = current_freq + 1
                        # Update latest score and timestamp
                        ws_summary.cell(row=row_idx, column=2, value=round(w_score, 4))
                        ws_summary.cell(row=row_idx, column=3, value=w_pct)
                        ws_summary.cell(row=row_idx, column=4, value=risk_label)
                        ws_summary.cell(row=row_idx, column=6, value=timestamp)
                    else:
                        new_row = [
                            clean_word,
                            round(w_score, 4),
                            w_pct,
                            risk_label,
                            1,
                            timestamp
                        ]
                        ws_summary.append(new_row)
                        summary_data[w_key] = ws_summary.max_row

                wb.save(self.excel_path)
                print(f" [WORD TOXICITY LOGGED]: Saved {len(word_scores)} words to {self.excel_path}")

            except PermissionError:
                print(f"\n [EXCEL ACCESS WARNING] Could not write to '{self.excel_path}' because the file is open in Microsoft Excel or another application. Please close the file to allow subsequent updates.\n")
            except Exception as e:
                print(f"\n [WORD TOXICITY LEDGER ERROR] Failed writing to Excel: {str(e)}\n")

    def load_log_df(self) -> pd.DataFrame:
        """Returns the All_Words_Log sheet as a Pandas DataFrame."""
        if not os.path.exists(self.excel_path):
            self._initialize_excel()
        with self.lock:
            try:
                return pd.read_excel(self.excel_path, sheet_name="All_Words_Log", engine="openpyxl")
            except Exception:
                return pd.DataFrame()

    def load_summary_df(self) -> pd.DataFrame:
        """Returns the Unique_Words_Summary sheet as a Pandas DataFrame."""
        if not os.path.exists(self.excel_path):
            self._initialize_excel()
        with self.lock:
            try:
                return pd.read_excel(self.excel_path, sheet_name="Unique_Words_Summary", engine="openpyxl")
            except Exception:
                return pd.DataFrame()
