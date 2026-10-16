"""
STAGE 2 (v2, corrected) - Gujarat medicinal plants
Fixes vs v1:
 1. Two-part ("hurdle") analysis: (a) WHERE is each crop grown? (b) among districts where it IS grown, how do stress
    features relate to yield?  v1 mixed 'not grown' (yield=0) with real yields, which flipped the heat result.
 2. Plant-specific slopes with partial pooling (not one slope forced on all crops) + year effect.
 3. Evidence language based on the credible interval, not 'P>0.5 = confirmed'.
Input: Stage1_District_Plant_Year_Panel.csv   Output: Stage2_v2_part_a_presence.csv, Stage2_v2_part_b_yield.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pymc as pm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

df = pd.read_csv("Stage1_District_Plant_Year_Panel.csv")
STRESS = ["Heat_Stress_C", "Rainfall_Stress_mm", "pH_Stress"]
NAMES = ["heat", "rainfall", "pH"]

def tier(x):
    """x = posterior samples. Honest evidence label."""
    lo, hi = np.percentile(x, [3, 97]); p = max((x < 0).mean(), (x > 0).mean())
    if lo > 0 or hi < 0: lab = "clear" if p > 0.99 else "moderate"
    elif p >= 0.90: lab = "weak/suggestive (interval includes 0)"
    else: lab = "inconclusive"
    return lo, hi, p, lab

# ---------------- PART A: where is the crop grown? (leave-one-district-out AUC) ----------------
rows = []
feats = ["Season_Avg_Temp_C", "Season_Total_Rainfall_mm", "Season_Avg_Humidity_pct", "District_pH_Estimate"]
for plant, g in df.groupby("Plant"):
    g = g.reset_index(drop=True); y = (g.Yield_kg_ha > 0).astype(int)
    if min((y == 0).sum(), (y == 1).sum()) < 10:
        rows.append(dict(Plant=plant, N=len(g), Pct_grown=round(y.mean() * 100, 1), LODO_AUC="n/a (fewer than 10 rows in one class)")); continue
    p = np.zeros(len(g))
    for tr, te in LeaveOneGroupOut().split(g, y, g.District):
        if y.iloc[tr].nunique() < 2: p[te] = y.iloc[tr].mean(); continue
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(g.loc[tr, feats], y.iloc[tr])
        p[te] = m.predict_proba(g.loc[te, feats])[:, 1]
    rows.append(dict(Plant=plant, N=len(g), Pct_grown=round(y.mean() * 100, 1), LODO_AUC=round(roc_auc_score(y, p), 3)))
A = pd.DataFrame(rows); A.to_csv("Stage2_v2_part_a_presence.csv", index=False)
print("PART A - can climate/soil tell WHERE a crop is grown? (AUC 0.5 = no skill, 1.0 = perfect)\n", A.to_string(index=False))

# ---------------- PART B: yield among districts where the crop is grown ----------------
G = df[df.Yield_kg_ha > 0].reset_index(drop=True)
G["y"] = G.groupby("Plant").Yield_kg_ha.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
for c in STRESS:  # z-score within plant; if a plant has no variation in a stress (always 0) it stays 0 and is NOT identified
    G[c + "_z"] = G.groupby("Plant")[c].transform(lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else 0.0)
pi, pl = pd.factorize(G.Plant); di, dl = pd.factorize(G.District); yi, yl = pd.factorize(G.Agri_Year)
X = G[[c + "_z" for c in STRESS]].values
MIN_NONZERO = 10  # data-sufficiency rule applied to every crop and every stress
nz = {(p, k): int((G.loc[G.Plant == p, STRESS[k]] > 0).sum()) for p in pl for k in range(3)}
ident = {key: v >= MIN_NONZERO for key, v in nz.items()}

with pm.Model():
    mu_b = pm.Normal("mu_b", 0, 0.5, shape=3); tau_b = pm.HalfNormal("tau_b", 0.5, shape=3)
    b_raw = pm.Normal("b_raw", 0, 1, shape=(len(pl), 3)); b = pm.Deterministic("b", mu_b + b_raw * tau_b)
    ds = pm.HalfNormal("ds", 1); do = pm.Normal("do", 0, 1, shape=len(dl))
    yr = pm.Normal("yr", 0, 0.5, shape=len(yl))
    mu = do[di] * ds + yr[yi] + (b[pi] * X).sum(1)
    s = pm.HalfNormal("s", 1); nu = pm.Exponential("nu", 1 / 10) + 2
    pm.StudentT("obs", nu=nu, mu=mu, sigma=s, observed=G.y.values)
    tr = pm.sample(1000, tune=1500, chains=4, cores=1, target_accept=0.95, random_seed=7, progressbar=False)

out = []
def add(label, samp, n=None):
    lo, hi, p, lab = tier(samp); out.append(dict(Scope=label, N_nonzero_rows=n, Mean=round(samp.mean(), 3), CI94=f"[{lo:+.2f}, {hi:+.2f}]", P_direction=round(p, 2), Evidence=lab))
for k, n in enumerate(NAMES): add(f"POOLED {n}", tr.posterior["mu_b"].values[..., k].flatten())
for i, p in enumerate(pl):
    for k, n in enumerate(NAMES):
        if ident[(p, k)]: add(f"{p}: {n}", tr.posterior["b"].values[..., i, k].flatten(), nz[(p, k)])
        else: out.append(dict(Scope=f"{p}: {n}", N_nonzero_rows=nz[(p, k)], Mean=np.nan, CI94="-", P_direction=np.nan, Evidence=f"NOT IDENTIFIED (fewer than {MIN_NONZERO} rows with non-zero stress)"))
B = pd.DataFrame(out); B.to_csv("Stage2_v2_part_b_yield.csv", index=False)
print("\nPART B - yield (within-crop z-score) vs stress, grown districts only, n =", len(G)); print(B.to_string(index=False))
import arviz as az
sm = az.summary(tr, var_names=["mu_b","tau_b","ds","s","nu"]); print("\nmax r_hat:", float(sm["r_hat"].max()), "| min ESS (bulk):", int(sm["ess_bulk"].min()), "| divergences:", int(tr.sample_stats["diverging"].sum()))
