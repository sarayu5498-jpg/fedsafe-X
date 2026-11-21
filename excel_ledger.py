import os
import threading
import pandas as pd
from datetime import datetime, timezone, timedelta
from config import EXCEL_LEDGER_PATH, MAX_WARNINGS_BEFORE_REPORT

class ExcelWarningLedger:
    """
    Manages user warning records and account block statuses in a persistent Excel file (user_warning_ledger.xlsx).
    """
    def __init__(self, excel_path: str = EXCEL_LEDGER_PATH):
        self.excel_path = excel_path
        self.lock = threading.Lock()
        self._initialize_excel()

    @staticmethod
    def get_full_ist_timestamp() -> str:
        ist_tz = timezone(timedelta(hours=5, minutes=30))
        return datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M:%S IST")

    def _initialize_excel(self):
        with self.lock:
            if not os.path.exists(self.excel_path):
                df = pd.DataFrame(columns=[
                    "Mobile Number", 
                    "Email ID", 
                    "Username", 
                    "Warning Count", 
                    "Account Status", 
                    "Last Updated IST"
                ])
                df.to_excel(self.excel_path, index=False, engine='openpyxl')
                print(f" 📊 [EXCEL LEDGER INITIALIZED]: Created {self.excel_path}")

    def load_df(self) -> pd.DataFrame:
        if not os.path.exists(self.excel_path):
            self._initialize_excel()
        with self.lock:
            try:
                df = pd.read_excel(self.excel_path, engine='openpyxl', dtype=str)
                df = df.dropna(how='all')
                return df.fillna("")
            except Exception:
                return pd.DataFrame()

    def get_user_record(self, email_id: str, mobile_number: str) -> dict:
        with self.lock:
            try:
                df = pd.read_excel(self.excel_path, engine='openpyxl', dtype=str) if os.path.exists(self.excel_path) else pd.DataFrame()
            except PermissionError:
                raise RuntimeError(f"Permission denied to read '{self.excel_path}'. Please close the file if it is open in another program (like Excel).")
        
        if df.empty:
            return {"warning_count": 0, "status": "ACTIVE"}

        clean_email = str(email_id).strip().lower()
        clean_mobile = str(mobile_number).strip()

        df = df.dropna(how='all')
        if 'Email ID' not in df.columns or 'Mobile Number' not in df.columns:
            return {"warning_count": 0, "status": "ACTIVE"}

        email_series = df['Email ID'].fillna("").astype(str).str.lower().str.strip()
        mobile_series = df['Mobile Number'].fillna("").astype(str).str.strip()

        match = df[(email_series == clean_email) | (mobile_series == clean_mobile)]
        if not match.empty:
            row = match.iloc[0]
            try:
                warning_count = int(float(row.get('Warning Count', 0)))
            except (ValueError, TypeError):
                warning_count = 0
            status = str(row.get('Account Status', 'ACTIVE')).upper()
            return {
                "warning_count": warning_count,
                "status": status
            }
        return {"warning_count": 0, "status": "ACTIVE"}

    def is_user_blocked(self, email_id: str, mobile_number: str) -> bool:
        record = self.get_user_record(email_id, mobile_number)
        return record["status"] == "BLOCKED" or record["warning_count"] >= MAX_WARNINGS_BEFORE_REPORT

    def increment_warning(self, username: str, mobile_number: str, email_id: str) -> int:
        with self.lock:
            try:
                df = pd.read_excel(self.excel_path, engine='openpyxl', dtype=str) if os.path.exists(self.excel_path) else pd.DataFrame()
            except PermissionError:
                raise RuntimeError(f"Permission denied to read '{self.excel_path}'. Please close the file if it is open in another program (like Excel).")
                
            clean_email = str(email_id).strip().lower()
            clean_mobile = str(mobile_number).strip()

            if not df.empty and 'Email ID' in df.columns and 'Mobile Number' in df.columns:
                email_series = df['Email ID'].fillna("").astype(str).str.lower().str.strip()
                mobile_series = df['Mobile Number'].fillna("").astype(str).str.strip()
                match_idx = df[(email_series == clean_email) | (mobile_series == clean_mobile)].index
            else:
                match_idx = pd.Index([])

            if not match_idx.empty:
                idx = match_idx[0]
                try:
                    current_count = int(float(df.at[idx, 'Warning Count'])) + 1
                except (ValueError, TypeError):
                    current_count = 1
                df.at[idx, 'Warning Count'] = str(current_count)
                df.at[idx, 'Username'] = username
                df.at[idx, 'Last Updated IST'] = self.get_full_ist_timestamp()
                
                if current_count >= MAX_WARNINGS_BEFORE_REPORT:
                    df.at[idx, 'Account Status'] = "BLOCKED"
            else:
                current_count = 1
                new_row = {
                    "Mobile Number": clean_mobile,
                    "Email ID": clean_email,
                    "Username": username,
                    "Warning Count": str(current_count),
                    "Account Status": "BLOCKED" if current_count >= MAX_WARNINGS_BEFORE_REPORT else "ACTIVE",
                    "Last Updated IST": self.get_full_ist_timestamp()
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            try:
                df.to_excel(self.excel_path, index=False, engine='openpyxl')
            except PermissionError:
                raise RuntimeError(f"Permission denied to write to '{self.excel_path}'. Please close the file if it is open in another program (like Excel).")
                
            print(f" 📊 [EXCEL LEDGER UPDATED]: User ({clean_email}) Warning Count -> {current_count}")
            return current_count

    def mark_blocked(self, username: str, mobile_number: str, email_id: str):
        with self.lock:
            try:
                df = pd.read_excel(self.excel_path, engine='openpyxl', dtype=str) if os.path.exists(self.excel_path) else pd.DataFrame()
            except PermissionError:
                raise RuntimeError(f"Permission denied to read '{self.excel_path}'. Please close the file if it is open in another program (like Excel).")
                
            clean_email = str(email_id).strip().lower()
            clean_mobile = str(mobile_number).strip()

            if not df.empty and 'Email ID' in df.columns and 'Mobile Number' in df.columns:
                email_series = df['Email ID'].fillna("").astype(str).str.lower().str.strip()
                mobile_series = df['Mobile Number'].fillna("").astype(str).str.strip()
                match_idx = df[(email_series == clean_email) | (mobile_series == clean_mobile)].index
            else:
                match_idx = pd.Index([])

            if not match_idx.empty:
                idx = match_idx[0]
                df.at[idx, 'Account Status'] = "BLOCKED"
                df.at[idx, 'Warning Count'] = str(MAX_WARNINGS_BEFORE_REPORT)
                df.at[idx, 'Last Updated IST'] = self.get_full_ist_timestamp()
            else:
                new_row = {
                    "Mobile Number": clean_mobile,
                    "Email ID": clean_email,
                    "Username": username,
                    "Warning Count": str(MAX_WARNINGS_BEFORE_REPORT),
                    "Account Status": "BLOCKED",
                    "Last Updated IST": self.get_full_ist_timestamp()
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            try:
                df.to_excel(self.excel_path, index=False, engine='openpyxl')
            except PermissionError:
                raise RuntimeError(f"Permission denied to write to '{self.excel_path}'. Please close the file if it is open in another program (like Excel).")
                
            print(f" 🔒 [EXCEL LEDGER PERMANENT BLOCK]: User ({clean_email}) marked as BLOCKED")