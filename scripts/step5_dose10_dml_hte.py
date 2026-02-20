from __future__ import annotations

from pathlib import Path
import sys

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


def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def wavg(x: np.ndarray, w: np.ndarray) -> float:
    m = (~np.isnan(x)) & (~np.isnan(w)) & (w > 0)
    if m.sum() == 0:
        return float("nan")
    return float(np.average(x[m], weights=w[m]))


def dml_aipw_ate(
    df: pd.DataFrame,
    y_col: str,
    t_col: str,
    x_cols_num: list[str],
    x_cols_cat: list[str],
    w_col: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """
    Cross-fitted AIPW for binary treatment & binary outcome.
    Returns ATE + robust SE + overlap diagnostics.
    """
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError("scikit-learn is required. Install with: poetry add scikit-learn") from e

    data = df.copy()

    keep = [y_col, t_col, w_col] + x_cols_num + x_cols_cat
    data = data[keep].copy()

    data = data.dropna(subset=[y_col, t_col, w_col]).copy()
    data[y_col] = to_num(data[y_col])
    data[t_col] = to_num(data[t_col])
    data[w_col] = to_num(data[w_col])
    data = data.dropna(subset=[y_col, t_col, w_col]).copy()
    data = data[data[w_col] > 0].copy()

    Y = data[y_col].astype(int).to_numpy()
    T = data[t_col].astype(int).to_numpy()
    W = data[w_col].astype(float).to_numpy()

    X = data[x_cols_num + x_cols_cat].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), x_cols_num),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), x_cols_cat),
        ],
        remainder="drop",
    )

    e_model = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    y_model = Pipeline(steps=[
        ("pre", pre),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=25,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    e_hat = np.full_like(Y, np.nan, dtype=float)
    m1_hat = np.full_like(Y, np.nan, dtype=float)
    m0_hat = np.full_like(Y, np.nan, dtype=float)

    for tr, te in skf.split(X, T):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        Y_tr, T_tr, W_tr = Y[tr], T[tr], W[tr]

        e_model.fit(X_tr, T_tr, clf__sample_weight=W_tr)
        e_hat[te] = e_model.predict_proba(X_te)[:, 1]

        mask1 = (T_tr == 1)
        mask0 = (T_tr == 0)

        if mask1.sum() >= 50:
            y_model.fit(X_tr[mask1], Y_tr[mask1], clf__sample_weight=W_tr[mask1])
            m1_hat[te] = y_model.predict_proba(X_te)[:, 1]
        else:
            m1_hat[te] = np.average(Y_tr, weights=W_tr)

        if mask0.sum() >= 50:
            y_model.fit(X_tr[mask0], Y_tr[mask0], clf__sample_weight=W_tr[mask0])
            m0_hat[te] = y_model.predict_proba(X_te)[:, 1]
        else:
            m0_hat[te] = np.average(Y_tr, weights=W_tr)

    eps = 1e-3
    e_hat = np.clip(e_hat, eps, 1 - eps)

    psi = (m1_hat - m0_hat) + (T * (Y - m1_hat) / e_hat) - ((1 - T) * (Y - m0_hat) / (1 - e_hat))

    ate = float(np.sum(W * psi) / np.sum(W))
    var = float(np.sum((W ** 2) * ((psi - ate) ** 2)) / (np.sum(W) ** 2))
    se = float(np.sqrt(var))
    ci_low = ate - 1.96 * se
    ci_high = ate + 1.96 * se

    y_t = float(np.sum(W[T == 1] * Y[T == 1]) / np.sum(W[T == 1]))
    y_c = float(np.sum(W[T == 0] * Y[T == 0]) / np.sum(W[T == 0]))
    diff_w = y_t - y_c

    overlap = {
        "e_hat_q01_q05_q50_q95_q99": list(np.quantile(e_hat, [0.01, 0.05, 0.5, 0.95, 0.99])),
        "treated_share_unw": float(T.mean()),
        "treated_share_w": float(np.sum(W * T) / np.sum(W)),
    }

    return {
        "n": int(len(Y)),
        "ate": ate,
        "se": se,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "naive_diff_w": float(diff_w),
        "y_mean_w": float(np.sum(W * Y) / np.sum(W)),
        "y_treated_w": y_t,
        "y_control_w": y_c,
        "psi": psi,
        "e_hat": e_hat,
        "m1_hat": m1_hat,
        "m0_hat": m0_hat,
        **overlap,
    }


def make_bins_age1990(age_1990: np.ndarray) -> pd.Categorical:
    bins = [-100, 0, 5, 10, 15, 18, 100]
    labels = ["<=0", "1-5", "6-10", "11-15", "16-18", "19+"]
    return pd.cut(age_1990, bins=bins, labels=labels, include_lowest=True)


def make_bins_yborn(yborn: np.ndarray) -> pd.Categorical:
    bins = [1900, 1945, 1955, 1965, 1975, 1985, 1990, 2005]
    labels = ["<=1945", "1946-55", "1956-65", "1966-75", "1976-85", "1986-90", "1991+"]
    return pd.cut(yborn, bins=bins, labels=labels, include_lowest=True)


def main() -> None:
    try:
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.ensemble import RandomForestRegressor
    except Exception:
        print("[error] scikit-learn missing. Run: poetry add scikit-learn", file=sys.stderr)
        raise

    root = find_repo_root()

    in_path = root / "data" / "derived" / "step2_model_afd.csv"
    out_ind = root / "data" / "derived" / "step5_cate_individual_dose10.csv"
    out_tables = root / "outputs" / "tables"
    safe_mkdir(out_ind.parent)
    safe_mkdir(out_tables)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    df = pd.read_csv(in_path)

    need = ["respid", "y_afd_vote", "wghtpew", "yborn", "gdr_socialized_10plus"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column in {in_path.name}: {c}")

    df["yborn"] = to_num(df["yborn"])
    df = df[df["yborn"] <= 1990].copy()

    df["y_afd_vote"] = to_num(df["y_afd_vote"])
    df["gdr_socialized_10plus"] = to_num(df["gdr_socialized_10plus"])
    df["wghtpew"] = to_num(df["wghtpew"])

    df = df.dropna(subset=["y_afd_vote", "gdr_socialized_10plus", "wghtpew", "yborn"]).copy()
    df = df[df["wghtpew"] > 0].copy()

    df["age_1990"] = 1990 - df["yborn"]

    x_num = ["yborn"]
    x_cat: list[str] = []
    if "sex" in df.columns:
        df["sex"] = to_num(df["sex"])
        x_cat = ["sex"]

    dml_out = dml_aipw_ate(
        df=df,
        y_col="y_afd_vote",
        t_col="gdr_socialized_10plus",
        x_cols_num=x_num,
        x_cols_cat=x_cat,
        w_col="wghtpew",
        n_splits=5,
        random_state=42,
    )

    # DR-learner stage-2: regress psi on X_hte
    x_num_hte = ["yborn", "age_1990"]
    x_cat_hte = x_cat.copy()
    X = df[x_num_hte + x_cat_hte].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("imp", SimpleImputer(strategy="median"))]), x_num_hte),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), x_cat_hte),
        ],
        remainder="drop",
    )

    cate_model = Pipeline(steps=[
        ("pre", pre),
        ("reg", RandomForestRegressor(
            n_estimators=1400,
            min_samples_leaf=50,
            random_state=7,
            n_jobs=-1
        ))
    ])

    psi = dml_out["psi"]
    W = df["wghtpew"].to_numpy(dtype=float)

    cate_model.fit(X, psi, reg__sample_weight=W)
    cate_hat = cate_model.predict(X)

    out = pd.DataFrame({
        "respid": df["respid"].values,
        "y": df["y_afd_vote"].astype(int).values,
        "t": df["gdr_socialized_10plus"].astype(int).values,
        "w": W,
        "yborn": df["yborn"].values,
        "age_1990": df["age_1990"].values,
        "e_hat": dml_out["e_hat"],
        "m1_hat": dml_out["m1_hat"],
        "m0_hat": dml_out["m0_hat"],
        "psi_aipw": psi,
        "cate_hat": cate_hat,
    })
    out.to_csv(out_ind, index=False)

    overall = pd.DataFrame([{
        "sample": "yborn<=1990_party_choosers",
        "n": dml_out["n"],
        "treated_share_w": dml_out["treated_share_w"],
        "y_mean_w": dml_out["y_mean_w"],
        "naive_diff_w": dml_out["naive_diff_w"],
        "ate_aipw": dml_out["ate"],
        "se": dml_out["se"],
        "ci_low": dml_out["ci_low"],
        "ci_high": dml_out["ci_high"],
        "ate_from_mean_cate_hat": wavg(cate_hat.astype(float), W),
        "e_hat_q01_q05_q50_q95_q99": dml_out["e_hat_q01_q05_q50_q95_q99"],
    }])
    overall.to_csv(out_tables / "step5_dose10_overall.csv", index=False)

    # --- bin tables (point estimates only; CI in step6) ---

    age_bin = make_bins_age1990(out["age_1990"].to_numpy(dtype=float))
    yb_bin = make_bins_yborn(out["yborn"].to_numpy(dtype=float))

    def bin_mean(df_in: pd.DataFrame, bin_obj, bin_name: str) -> pd.DataFrame:
        tmp = df_in.copy()
        # Ensure Series aligned with tmp index
        tmp[bin_name] = pd.Series(bin_obj, index=tmp.index).astype(str)

        rows = []
        for k, g in tmp.groupby(bin_name, dropna=False):
            rows.append({
                bin_name: str(k),
                "n": int(len(g)),
                "cate_mean_w": wavg(g["cate_hat"].to_numpy(dtype=float), g["w"].to_numpy(dtype=float)),
            })
        return pd.DataFrame(rows)

    tab_age = bin_mean(out, age_bin, "age_1990_bin")
    tab_age.to_csv(out_tables / "step5_dose10_cate_by_age1990.csv", index=False)

    tab_yb = bin_mean(out, yb_bin, "yborn_bin")
    tab_yb.to_csv(out_tables / "step5_dose10_cate_by_yborn.csv", index=False)

    print(f"[ok] wrote: {out_ind}")
    print(f"[ok] wrote: {out_tables / 'step5_dose10_overall.csv'}")
    print(f"[ok] wrote: {out_tables / 'step5_dose10_cate_by_age1990.csv'}")
    print(f"[ok] wrote: {out_tables / 'step5_dose10_cate_by_yborn.csv'}")


if __name__ == "__main__":
    main()
