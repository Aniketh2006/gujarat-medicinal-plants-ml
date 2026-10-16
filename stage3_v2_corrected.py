"""
STAGE 3 (v2, corrected) - discover relationships
Fixes vs v1: no latitude/longitude, no plant-constant columns (they only encode crop identity), yield standardised within crop,
two separate questions (where is the crop grown? / yield where it is grown), and every score is compared with a baseline.
Input: Stage1_District_Plant_Year_Panel.csv
Output: Stage3_v2_summary.csv, Stage3_v2_presence_SHAP.png, Stage3_v2_presence_top_interactions.csv
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, xgboost as xgb, shap
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, roc_auc_score

df = pd.read_csv("Stage1_District_Plant_Year_Panel.csv")
CLIM = ["Season_Avg_Temp_C", "Season_Total_Rainfall_mm", "Season_Avg_Humidity_pct", "District_pH_Estimate"]
plants = pd.get_dummies(df.Plant, prefix="Plant").astype(int)
P = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)

def lodo(model_fn, X, y, groups):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = model_fn().fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict_proba(X.iloc[te])[:, 1] if hasattr(m, "predict_proba") else m.predict(X.iloc[te])
    return pred

summary = []
# ---- Q1: WHERE is each crop grown? (presence vs absence) ----
X = pd.concat([df[CLIM], plants], axis=1); y = (df.Yield_kg_ha > 0).astype(int)
p = lodo(lambda: xgb.XGBClassifier(**P), X, y, df.District)
base = np.zeros(len(df))                                   # baseline: knows only the crop (its share of grown districts)
for d in df.District.unique():
    te = (df.District == d).values; base[te] = df[te].Plant.map((y[~te].groupby(df.Plant[~te]).mean())).values
summary.append(dict(Question="Where is the crop grown? (AUC)", Model=round(roc_auc_score(y, p), 3), Baseline_crop_only=round(roc_auc_score(y, base), 3)))

# ---- Q2: yield where the crop IS grown (within-crop z-score) ----
g = df[df.Yield_kg_ha > 0].reset_index(drop=True)
g["y"] = g.groupby("Plant").Yield_kg_ha.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
Xg = pd.concat([g[CLIM + ["Heat_Stress_C", "Rainfall_Stress_mm", "pH_Stress"]], pd.get_dummies(g.Plant, prefix="Plant").astype(int)], axis=1)
pg = lodo(lambda: xgb.XGBRegressor(**P), Xg, g.y, g.District)
summary.append(dict(Question="Yield where grown (R2; 0 = no better than crop average)", Model=round(r2_score(g.y, pg), 3), Baseline_crop_only=0.0))
S = pd.DataFrame(summary); S.to_csv("Stage3_v2_summary.csv", index=False); print(S.to_string(index=False))

# ---- SHAP for the presence model (the question with real signal) ----
m = xgb.XGBClassifier(**P).fit(X, y)
ex = shap.TreeExplainer(m); sv = ex(X)
plt.figure(); shap.summary_plot(sv, X, show=False); plt.tight_layout(); plt.savefig("Stage3_v2_presence_SHAP.png", dpi=120, bbox_inches="tight"); plt.close()
iv = np.abs(ex.shap_interaction_values(X)).mean(0); np.fill_diagonal(iv, 0)
pairs = [dict(Feature_1=X.columns[i], Feature_2=X.columns[j], Mean_Abs_Interaction=round(iv[i, j], 4)) for i in range(len(X.columns)) for j in range(i + 1, len(X.columns))]
T = pd.DataFrame(pairs).sort_values("Mean_Abs_Interaction", ascending=False).head(10); T.to_csv("Stage3_v2_presence_top_interactions.csv", index=False)
print("\nTop interactions in the presence model:\n", T.to_string(index=False))
