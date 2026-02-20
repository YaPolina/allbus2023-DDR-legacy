from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def find_repo_root(start: Path | None = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").exists():
            return p
    return Path(__file__).resolve().parents[1]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def wavg(x: np.ndarray, w: np.ndarray) -> float:
    m = (~np.isnan(x)) & (~np.isnan(w)) & (w > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


def percentile_ci(samples: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(np.nanquantile(samples, alpha / 2))
    hi = float(np.nanquantile(samples, 1 - alpha / 2))
    return lo, hi


def make_age1990_bins(age_1990: np.ndarray) -> tuple[np.ndarray, list[str]]:
    bins = [-100, 0, 5, 10, 15, 18, 100]
    labels = ["<=0", "1-5", "6-10", "11-15", "16-18", "19+"]
    cat = pd.cut(age_1990, bins=bins, labels=labels, include_lowest=True)
    lab_arr = pd.Series(cat).astype("string").fillna("nan").to_numpy()
    return lab_arr, labels


def make_yborn_bins(yborn: np.ndarray) -> tuple[np.ndarray, list[str]]:
    bins = [1900, 1945, 1955, 1965, 1975, 1985, 1990, 2005]
    labels = ["<=1945", "1946-55", "1956-65", "1966-75", "1976-85", "1986-90", "1991+"]
    cat = pd.cut(yborn, bins=bins, labels=labels, include_lowest=True)
    lab_arr = pd.Series(cat).astype("string").fillna("nan").to_numpy()
    return lab_arr, labels


def main() -> None:
    root = find_repo_root()

    in_path = root / "data" / "derived" / "step5_cate_individual_dose10.csv"
    out_tables = root / "outputs" / "tables"
    safe_mkdir(out_tables)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)

    required = ["psi_aipw", "cate_hat", "w", "yborn", "age_1990"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column: {c}")

    psi = pd.to_numeric(df["psi_aipw"], errors="coerce").to_numpy(dtype=float)
    cate = pd.to_numeric(df["cate_hat"], errors="coerce").to_numpy(dtype=float)
    w = pd.to_numeric(df["w"], errors="coerce").to_numpy(dtype=float)
    yborn = pd.to_numeric(df["yborn"], errors="coerce").to_numpy(dtype=float)
    age_1990 = pd.to_numeric(df["age_1990"], errors="coerce").to_numpy(dtype=float)

    # keep complete cases
    m = (
        (~np.isnan(psi)) & (~np.isnan(cate)) & (~np.isnan(w)) & (w > 0)
        & (~np.isnan(yborn)) & (~np.isnan(age_1990))
    )
    psi, cate, w, yborn, age_1990 = psi[m], cate[m], w[m], yborn[m], age_1990[m]

    n = len(psi)
    if n == 0:
        raise ValueError("No valid rows after filtering (all missing?).")

    # fixed bin labels once
    age_lab_arr, age_labels = make_age1990_bins(age_1990)
    yb_lab_arr, yb_labels = make_yborn_bins(yborn)

    # masks (fixed once)
    age_masks = {lab: (age_lab_arr == lab) for lab in age_labels}
    yb_masks = {lab: (yb_lab_arr == lab) for lab in yb_labels}

    # point estimates
    ate_point = wavg(psi, w)

    # bootstrap
    B = 500  # if slow: set to 200; if you want smoother CI: 1000
    rng = np.random.default_rng(42)

    ate_bs = np.full(B, np.nan, dtype=float)
    cate_age_bs = {lab: np.full(B, np.nan, dtype=float) for lab in age_labels}
    cate_yb_bs = {lab: np.full(B, np.nan, dtype=float) for lab in yb_labels}

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        counts = np.bincount(idx, minlength=n).astype(float)

        w_b = w * counts

        ate_bs[b] = wavg(psi, w_b)

        for lab, mask in age_masks.items():
            if mask.sum() == 0:
                continue
            cate_age_bs[lab][b] = wavg(cate[mask], w_b[mask])

        for lab, mask in yb_masks.items():
            if mask.sum() == 0:
                continue
            cate_yb_bs[lab][b] = wavg(cate[mask], w_b[mask])

    # overall CI
    ate_lo, ate_hi = percentile_ci(ate_bs, alpha=0.05)
    overall = pd.DataFrame([{
        "bootstrap_B": B,
        "n": int(n),
        "ate_point": float(ate_point),
        "ate_ci_low": ate_lo,
        "ate_ci_high": ate_hi,
    }])
    overall.to_csv(out_tables / "step6_dose10_ci_overall.csv", index=False)

    # Age-bin CI
    rows = []
    for lab in age_labels:
        samples = cate_age_bs[lab]
        if np.all(np.isnan(samples)):
            continue
        lo, hi = percentile_ci(samples, alpha=0.05)
        mask = age_masks[lab]
        rows.append({
            "age_1990_bin": lab,
            "n": int(mask.sum()),
            "cate_point_w": wavg(cate[mask], w[mask]),
            "cate_ci_low": lo,
            "cate_ci_high": hi,
            "bootstrap_B": B,
        })
    pd.DataFrame(rows).to_csv(out_tables / "step6_dose10_ci_by_age1990.csv", index=False)

    # Yborn-bin CI
    rows = []
    for lab in yb_labels:
        samples = cate_yb_bs[lab]
        if np.all(np.isnan(samples)):
            continue
        lo, hi = percentile_ci(samples, alpha=0.05)
        mask = yb_masks[lab]
        rows.append({
            "yborn_bin": lab,
            "n": int(mask.sum()),
            "cate_point_w": wavg(cate[mask], w[mask]),
            "cate_ci_low": lo,
            "cate_ci_high": hi,
            "bootstrap_B": B,
        })
    pd.DataFrame(rows).to_csv(out_tables / "step6_dose10_ci_by_yborn.csv", index=False)

    print(f"[ok] wrote: {out_tables / 'step6_dose10_ci_overall.csv'}")
    print(f"[ok] wrote: {out_tables / 'step6_dose10_ci_by_age1990.csv'}")
    print(f"[ok] wrote: {out_tables / 'step6_dose10_ci_by_yborn.csv'}")


if __name__ == "__main__":
    main()
