"""The FY27 recompete cliff: federal contract follow-on outcomes and incumbent-loss risk."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DUCKDB_PATH = PROCESSED_DIR / "recompete.duckdb"
SQL_DIR = ROOT / "sql"
MODEL_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "tableau" / "exports"
APP_DATA_DIR = ROOT / "app" / "data"
FIGURES_DIR = ROOT / "reports" / "figures"
