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


def main() -> None:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestRegressor

    root = find_repo_root()
    in_path = root / "data" / "derived" / "step2_model_afd.csv"
    out_ind = root / "data" / "derived" / "step4_cate_individual.csv"
    out_dir_tables = root / "outputs" / "tables"
    safe_mkdir(out_ind.parent)
    safe_mkdir(out_dir_tables)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    df = pd.read_csv(in_path)

    # Required columns
    required = ["respid", "y_afd_vote", "east_youth", "wghtpew", "yborn"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"Missing column in input: {c}")

    # Basic derived moderators (pre-treatment)
    df["yborn"] = pd.to_numeric(df["yborn"], errors="coerce")
    df["age_1990"] = 1990 - df["yborn"]

    # Define Y, T, W
    df["y_afd_vote"] = pd.to_numeric(df["y_afd_vote"], errors="coerce")
    df["east_youth"] = pd.to_numeric(df["east_youth"], errors="coerce")
    df["wghtpew"] = pd.to_numeric(df["wghtpew"], errors="coerce")

    df = df.dropna(subset=["y_afd_vote", "east_youth", "wghtpew", "yborn"]).copy()
    df = df[df["wghtpew"] > 0].copy()

    Y = df["y_afd_vote"].astype(int).to_numpy()
    T = df["east_youth"].astype(int).to_numpy()
    W = df["wghtpew"].astype(float).to_numpy()

    # Moderators X for heterogeneity (pre-treatment only)
    x_num = ["yborn", "age_1990"]
    x_cat = []
    if "sex" in df.columns:
        x_cat = ["sex"]

    X = df[x_num + x_cat].copy()

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]), x_num),
            ("cat", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), x_cat),
        ],
        remainder="drop"
    )

    # Nuisance models (cross-fitting)
    e_model = Pipeline(steps=[
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    # For binary outcome, regressor on {0,1} is fine
    y_model = Pipeline(steps=[
        ("pre", pre),
        ("reg", RandomForestRegressor(
            n_estimators=600,
            min_samples_leaf=30,
            random_state=42,
            n_jobs=-1
        ))
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n = len(df)
    e_hat = np.full(n, np.nan, dtype=float)
    m1_hat = np.full(n, np.nan, dtype=float)
    m0_hat = np.full(n, np.nan, dtype=float)

    for tr, te in skf.split(X, T):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        Y_tr, T_tr, W_tr = Y[tr], T[tr], W[tr]

        # propensity
        e_model.fit(X_tr, T_tr, clf__sample_weight=W_tr)
        e_hat[te] = e_model.predict_proba(X_te)[:, 1]

        # outcome models (separate by treatment)
        mask1 = (T_tr == 1)
        mask0 = (T_tr == 0)

        if mask1.sum() >= 50:
            y_model.fit(X_tr[mask1], Y_tr[mask1], reg__sample_weight=W_tr[mask1])
            m1_hat[te] = y_model.predict(X_te)
        else:
            m1_hat[te] = np.average(Y_tr, weights=W_tr)

        if mask0.sum() >= 50:
            y_model.fit(X_tr[mask0], Y_tr[mask0], reg__sample_weight=W_tr[mask0])
            m0_hat[te] = y_model.predict(X_te)
        else:
            m0_hat[te] = np.average(Y_tr, weights=W_tr)

    # stabilize e_hat
    eps = 1e-3
    e_hat = np.clip(e_hat, eps, 1 - eps)

    # AIPW pseudo-outcome (individual effect score)
    psi = (m1_hat - m0_hat) + (T * (Y - m1_hat) / e_hat) - ((1 - T) * (Y - m0_hat) / (1 - e_hat))

    # Stage-2: learn CATE(x) by regressing psi on X
    cate_model = Pipeline(steps=[
        ("pre", pre),
        ("reg", RandomForestRegressor(
            n_estimators=1200,
            min_samples_leaf=40,
            random_state=7,
            n_jobs=-1
        ))
    ])
    cate_model.fit(X, psi, reg__sample_weight=W)
    cate_hat = cate_model.predict(X)

    # Save individual-level outputs
    out = pd.DataFrame({
        "respid": df["respid"].values,
        "y": Y,
        "t": T,
        "w": W,
        "yborn": df["yborn"].values,
        "age_1990": df["age_1990"].values,
        "e_hat": e_hat,
        "m1_hat": m1_hat,
        "m0_hat": m0_hat,
        "psi_aipw": psi,
        "cate_hat": cate_hat,
    })
    out.to_csv(out_ind, index=False)

    # Overall summaries
    ate_aipw = wavg(psi, W)
    ate_cate = wavg(cate_hat, W)

    overall = pd.DataFrame([{
        "n": int(n),
        "treated_share_w": float(np.sum(W * T) / np.sum(W)),
        "y_mean_w": wavg(Y.astype(float), W),
        "ate_from_aipw_mean_psi": ate_aipw,
        "ate_from_mean_cate_hat": ate_cate,
        "e_hat_q01_q05_q50_q95_q99": list(np.quantile(e_hat, [0.01, 0.05, 0.5, 0.95, 0.99])),
    }])
    overall.to_csv(out_dir_tables / "step4_cate_overall.csv", index=False)

    # Binning helper
    def bin_table(values: np.ndarray, bins, labels, name: str) -> pd.DataFrame:
        b = pd.cut(values, bins=bins, labels=labels, include_lowest=True)
        dfb = pd.DataFrame({"bin": b, "cate_hat": cate_hat, "w": W})
        grp = dfb.groupby("bin", dropna=False)
        rows = []
        for k, g in grp:
            rows.append({
                name: str(k),
                "n": int(len(g)),
                "cate_mean_w": wavg(g["cate_hat"].to_numpy(dtype=float), g["w"].to_numpy(dtype=float)),
            })
        return pd.DataFrame(rows)

    # By age_1990 bins (interpretation: cohorts that were older/younger at reunification)
    age = out["age_1990"].to_numpy(dtype=float)
    bins_age = [-100, 0, 5, 10, 15, 18, 100]
    labels_age = ["<=0", "1-5", "6-10", "11-15", "16-18", "19+"]
    tab_age = bin_table(age, bins_age, labels_age, "age_1990_bin")
    tab_age.to_csv(out_dir_tables / "step4_cate_by_age1990.csv", index=False)

    # By birth year bins
    yb = out["yborn"].to_numpy(dtype=float)
    bins_yb = [1900, 1945, 1955, 1965, 1975, 1985, 1990, 2005]
    labels_yb = ["<=1945", "1946-55", "1956-65", "1966-75", "1976-85", "1986-90", "1991+"]
    tab_yb = bin_table(yb, bins_yb, labels_yb, "yborn_bin")
    tab_yb.to_csv(out_dir_tables / "step4_cate_by_yborn.csv", index=False)

    print(f"[ok] wrote: {out_ind}")
    print(f"[ok] wrote: {out_dir_tables / 'step4_cate_overall.csv'}")
    print(f"[ok] wrote: {out_dir_tables / 'step4_cate_by_age1990.csv'}")
    print(f"[ok] wrote: {out_dir_tables / 'step4_cate_by_yborn.csv'}")


if __name__ == "__main__":
    main()
