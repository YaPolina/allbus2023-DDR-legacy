from __future__ import annotations
import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Minimal cleaning:
    - standardize column names (lowercase)
    - convert obvious missing codes if present later (we'll refine once we see codebook)
    """
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out
