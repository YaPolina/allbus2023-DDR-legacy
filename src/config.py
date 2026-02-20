from pathlib import Path

# Project root = folder that contains pyproject.toml
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DERIVED_DIR = DATA_DIR / "derived"

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# You can change this depending on what you downloaded from GESIS
RAW_DATA_FILENAME = "ALLBUS_2023.dta"  # or "ALLBUS_2023.sav"
RAW_DATA_PATH = RAW_DIR / RAW_DATA_FILENAME

DERIVED_SAMPLE_PATH = DERIVED_DIR / "analysis_sample.parquet"
