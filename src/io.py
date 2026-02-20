from __future__ import annotations

from pathlib import Path
import pandas as pd
import pyreadstat


def read_allbus(path: Path) -> pd.DataFrame:
    """
    Read ALLBUS dataset from .dta (Stata) or .sav (SPSS).
    Returns a pandas DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".dta":
        df, meta = pyreadstat.read_dta(str(path), apply_value_formats=False)
    elif suffix == ".sav":
        df, meta = pyreadstat.read_sav(str(path), apply_value_formats=False)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .dta or .sav")

    return df
