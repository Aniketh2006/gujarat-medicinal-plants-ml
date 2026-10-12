"""
Stage 2 (corrected): Gujarat medicinal plants
Part A: Where is the crop grown? (yield > 0)  -> logistic regression, leave-one-district-out AUC
Part B: Yield where grown (yield > 0)         -> Bayesian hierarchical model (PyMC)

Run:  python stage2_v2_corrected.py
"""
import sys
import numpy as np
import pandas as pd

# =====================================================================
# STEP 1. SETTINGS - EDIT THESE TO MATCH YOUR CSV COLUMN NAMES
# =====================================================================
CSV_FILE = "Stage1_District_Plant_Year_Panel.csv"

DISTRICT = "District"
CROP     = "Plant"
YIELD    = "Yield_kg_ha"
HEAT     = "Heat_Stress_C"
RAIN     = "Rainfall_Stress_mm"
PH       = "pH_Stress"

STRESS = {"Heat": HEAT, "Rainfall": RAIN, "pH": PH}

DRAWS, TUNE, CHAINS, SEED = 1000, 1000, 4, 42


# =====================================================================
# STEP 2. LOAD AND CHECK DATA
# =====================================================================
def load():
    df = pd.read_csv(CSV_FILE)
    print("Columns in your file:\n ", list(df.columns), "\n")
    needed = [DISTRICT, CROP, YIELD, HEAT, RAIN, PH]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: these names in STEP 1 are not in your CSV: {missing}\n"
                 "Edit the settings block at the top to match the column list above.")
    df = df.dropna(subset=needed).copy()
    df["grown"] = (df[YIELD] > 0).astype(int)
    print("Rows:", len(df))
    print("Share of rows with yield = 0 (not grown), by crop:")
    print((1 - df.groupby(CROP)["grown"].mean()).round(3).to_string(), "\n")
    return df


# =====================================================================
# STEP 3. PART A - PRESENCE (is the crop grown here?)
# =====================================================================
def part_a(df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    rows = []
    for crop, g in df.groupby(CROP):
        feats = [c for c in STRESS.values() if g[c].std() > 0]  # drop constant columns
        if g["grown"].nunique() < 2 or not feats:
            rows.append({"Crop": crop, "AUC": np.nan, "Note": "only one class or no varying features"})
            continue
        X, y, grp = g[feats].values, g["grown"].values, g[DISTRICT].values
        prob = np.zeros(len(g))
        for tr, te in LeaveOneGroupOut().split(X, y, grp):
            if len(set(y[tr])) < 2:
                prob[te] = y[tr].mean()
                continue
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            m.fit(X[tr], y[tr])
            prob[te] = m.predict_proba(X[te])[:, 1]
        rows.append({"Crop": crop, "AUC": round(roc_auc_score(y, prob), 3),
                     "Features": ", ".join(feats), "Note": "leave-one-district-out"})
    out = pd.DataFrame(rows)
    print("PART A - presence model\n", out.to_string(index=False), "\n")
    out.to_csv("stage2_v2_part_a_presence.csv", index=False)


# =====================================================================
# STEP 4. PART B - YIELD WHERE GROWN (Bayesian hierarchical)
# =====================================================================
def label(hdi_lo, hdi_hi, pd_dir, identified):
    if not identified:
        return "not identified"
    if hdi_lo > 0 or hdi_hi < 0:
        return "clear"
    if pd_dir >= 0.90:
        return "moderate"
    if pd_dir >= 0.80:
        return "weak"
    return "inconclusive"


def part_b(df):
    import pymc as pm
    import arviz as az

    d = df[df[YIELD] > 0].copy()          # ONLY districts where the crop is grown
    crops = sorted(d[CROP].unique())
    ci = d[CROP].map({c: i for i, c in enumerate(crops)}).values
    dists = sorted(d[DISTRICT].unique())
    di = d[DISTRICT].map({x: i for i, x in enumerate(dists)}).values

    # standardise yield and predictors WITHIN each crop
    y = np.zeros(len(d))
    X = np.zeros((len(d), 3))
    ident = np.ones((len(crops), 3), dtype=bool)
    for k, c in enumerate(crops):
        m = ci == k
        y[m] = (d.loc[m, YIELD] - d.loc[m, YIELD].mean()) / d.loc[m, YIELD].std()
        for j, col in enumerate(STRESS.values()):
            s = d.loc[m, col].std()
            if s > 0:
                X[m, j] = (d.loc[m, col] - d.loc[m, col].mean()) / s
            else:
                ident[k, j] = False       # constant in this crop -> cannot estimate

    print(f"PART B - {len(d)} grown rows, {len(crops)} crops, {len(dists)} districts")
    with pm.Model():
        mu_b = pm.Normal("mu_b", 0, 0.5, shape=3)
        sd_b = pm.HalfNormal("sd_b", 0.5)
        z = pm.Normal("z", 0, 1, shape=(len(crops), 3))
        b = pm.Deterministic("b", mu_b + sd_b * z)               # crop-specific slopes
        a = pm.Normal("a", 0, 0.5, shape=len(crops))
        sd_d = pm.HalfNormal("sd_d", 0.5)
        zd = pm.Normal("zd", 0, 1, shape=len(dists))
        mu = a[ci] + sd_d * zd[di] + (b[ci] * X).sum(axis=1)
        pm.Normal("obs", mu, pm.HalfNormal("sigma", 1), observed=y)
        idata = pm.sample(DRAWS, tune=TUNE, chains=CHAINS, cores=1,
                          target_accept=0.95, random_seed=SEED, progressbar=True)

    summ = az.summary(idata, var_names=["b"])
    div = int(idata.sample_stats["diverging"].sum())
    print("\nDivergences:", div, "| max r_hat:", float(summ["r_hat"].max()))

    post = idata.posterior["b"].stack(s=("chain", "draw")).values   # crop x pred x samples
    rows = []
    for k, c in enumerate(crops):
        for j, name in enumerate(STRESS):
            s = post[k, j]
            lo, hi = np.percentile(s, [3, 97])
            pdir = max((s > 0).mean(), (s < 0).mean())
            if not ident[k, j]:   # stress never varies for this crop: do not report a number
                rows.append({"Crop": c, "Effect": name, "Mean": np.nan, "HDI_3%": np.nan,
                             "HDI_97%": np.nan, "P_direction": np.nan,
                             "Evidence": "not identified"})
                continue
            rows.append({"Crop": c, "Effect": name, "Mean": round(s.mean(), 3),
                         "HDI_3%": round(lo, 3), "HDI_97%": round(hi, 3),
                         "P_direction": round(pdir, 3),
                         "Evidence": label(lo, hi, pdir, ident[k, j])})
    out = pd.DataFrame(rows)
    print("\nPART B - crop-specific effects on yield (standardised)\n", out.to_string(index=False))
    out.to_csv("stage2_v2_part_b_yield.csv", index=False)
    print("\nSaved: stage2_v2_part_a_presence.csv, stage2_v2_part_b_yield.csv")


if __name__ == "__main__":
    data = load()
    part_a(data)
    part_b(data)
