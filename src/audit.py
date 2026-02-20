from __future__ import annotations
import pandas as pd


def make_variable_audit(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Create a compact overview: dtype, missing share, number of unique values, and top values.
    """
    rows = []
    n = len(df)

    for col in df.columns:
        s = df[col]
        miss = s.isna().mean()
        nunique = s.nunique(dropna=True)

        top_vals = (
            s.value_counts(dropna=True)
            .head(top_n)
            .to_dict()
        )

        rows.append(
            {
                "variable": col,
                "dtype": str(s.dtype),
                "missing_share": float(miss),
                "nunique": int(nunique),
                "top_values": str(top_vals),
            }
        )

    audit = pd.DataFrame(rows).sort_values(["missing_share", "nunique"], ascending=[True, True])
    return audit
