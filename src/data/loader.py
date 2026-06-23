from __future__ import annotations

from pathlib import Path

import pandas as pd


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the configured dataset is missing."""


class WorkbookLoader:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load_all_sheets(self) -> dict[str, pd.DataFrame]:
        if not self.path.exists():
            raise DatasetNotFoundError(f"Dataset not found at {self.path}")
        return pd.read_excel(self.path, sheet_name=None, engine="openpyxl")

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        if not self.path.exists():
            raise DatasetNotFoundError(f"Dataset not found at {self.path}")
        return pd.read_excel(self.path, sheet_name=sheet_name, engine="openpyxl")
