"""
STAGE 2 — VALIDATE TRADITIONAL KNOWLEDGE
Gujarat Medicinal Plants ML Project

WHAT THIS SCRIPT DOES:
Tests whether deviating from each plant's documented "ideal" conditions
(FAO EcoCrop / DMAPR requirement ranges) actually predicts lower yield.
This is the Bayesian hierarchical model mam specified:
    yield ~ heat_stress + rainfall_stress + pH_stress + (1 | district)
Using PyMC, with weakly-informative priors, so you get a POSTERIOR
PROBABILITY (e.g. "89% probability that heat stress reduces yield") rather
than a plain p-value -- appropriate given we only have 426 rows.

INPUT: Stage1_District_Plant_Year_Panel.csv (from Person 1 / Stage 1)
OUTPUT: Stage2_Bayesian_Results.csv + a results summary printed to screen
        + a trace plot image (Stage2_trace_plot.png)
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib
matplotlib.use("Agg")  # safe for headless/no-display environments
import matplotlib.pyplot as plt

RANDOM_SEED = 42


def load_and_prepare(path="Stage1_District_Plant_Year_Panel.csv"):
    df = pd.read_csv(path)

    # Standardize YIELD within each plant too (same logic Stage 1 used for
    # the stress features) -- this is necessary because Isabgol yields
    # (~900 kg/ha) and Castor yields (~1800 kg/ha) are on completely
    # different scales; without this the pooled model would be dominated
    # by whichever plant has the biggest raw numbers.
    df['Yield_zscore'] = df.groupby('Plant')['Yield_kg_ha'].transform(
        lambda x: (x - x.mean()) / x.std(ddof=0)
    )

    # Drop any row with a missing stress z-score (a plant missing one
    # requirement field, e.g. Fenugreek's absolute ranges) rather than
    # silently filling it -- the model must only see real values.
    needed = ['Yield_zscore', 'Heat_Stress_C_zscore', 'Rainfall_Stress_mm_zscore', 'pH_Stress_zscore', 'District']
    before = len(df)
    df = df.dropna(subset=needed).reset_index(drop=True)
    print(f"Rows used: {len(df)} (dropped {before - len(df)} with missing stress/yield values)")

    district_idx, district_labels = pd.factorize(df['District'])
    df['district_idx'] = district_idx
    return df, district_labels


def build_and_sample_model(df, n_districts):
    coords = {"district": np.arange(n_districts)}
    with pm.Model(coords=coords) as model:
        # --- weakly-informative priors -----------------------------------
        # Fixed-effect slopes: centered at 0 (0 = "at the plant's documented
        # ideal range", since our stress features are gap-based and already
        # 0 there), wide enough (sd=1) to not force a result either way.
        beta_heat = pm.Normal("beta_heat", mu=0, sigma=1)
        beta_rain = pm.Normal("beta_rain", mu=0, sigma=1)
        beta_ph = pm.Normal("beta_ph", mu=0, sigma=1)

        # District random intercept (partial pooling): districts shrink
        # toward the overall mean unless the data strongly says otherwise.
        district_sigma = pm.HalfNormal("district_sigma", sigma=1)
        district_offset = pm.Normal("district_offset", mu=0, sigma=1, dims="district")
        district_effect = district_offset * district_sigma

        mu = (
            district_effect[df['district_idx'].values]
            + beta_heat * df['Heat_Stress_C_zscore'].values
            + beta_rain * df['Rainfall_Stress_mm_zscore'].values
            + beta_ph * df['pH_Stress_zscore'].values
        )

        sigma = pm.HalfNormal("sigma", sigma=1)
        pm.Normal("yield_obs", mu=mu, sigma=sigma, observed=df['Yield_zscore'].values)

        # cores=1, chains=4 run sequentially: slower, but far more reliable
        # across different machines (especially Windows, where PyMC's
        # multiprocessing sampler can be unstable) than letting PyMC guess
        # the CPU/core count automatically.
        trace = pm.sample(2000, tune=1500, target_accept=0.95, random_seed=RANDOM_SEED,
                           chains=4, cores=1, progressbar=True)
    return model, trace


def summarize(trace):
    summary = az.summary(trace, var_names=["beta_heat", "beta_rain", "beta_ph"])
    print("\n=== Posterior summary (mean effect of each stress on standardized yield) ===")
    print(summary)

    results = []
    for name, label in [("beta_heat", "Heat stress"), ("beta_rain", "Rainfall stress"), ("beta_ph", "pH stress")]:
        samples = trace.posterior[name].values.flatten()
        p_negative = (samples < 0).mean()  # probability this stress REDUCES yield
        results.append({
            "Stress_Factor": label,
            "Posterior_Mean_Effect": round(samples.mean(), 3),
            "Probability_Reduces_Yield_pct": round(p_negative * 100, 1),
            "Conclusion": (
                f"Traditional claim CONFIRMED with {p_negative*100:.1f}% posterior probability"
                if p_negative > 0.5 else
                f"Traditional claim NOT supported ({ (1-p_negative)*100:.1f}% probability effect is non-negative)"
            ),
        })
    results_df = pd.DataFrame(results)
    print("\n=== Validate Traditional Knowledge: final answer per stress factor ===")
    print(results_df.to_string(index=False))
    return results_df


def main():
    df, district_labels = load_and_prepare()
    print(f"\nDistricts: {len(district_labels)} | Plants: {df['Plant'].nunique()} | Rows: {len(df)}")

    model, trace = build_and_sample_model(df, n_districts=len(district_labels))

    results_df = summarize(trace)
    results_df.to_csv("Stage2_Bayesian_Results.csv", index=False)

    az.plot_trace(trace, var_names=["beta_heat", "beta_rain", "beta_ph"])
    plt.tight_layout()
    plt.savefig("Stage2_trace_plot.png", dpi=120)
    print("\nSaved: Stage2_Bayesian_Results.csv, Stage2_trace_plot.png")

    # quick convergence check -- r_hat should be close to 1.0 (< 1.01 is good)
    rhat = az.rhat(trace, var_names=["beta_heat", "beta_rain", "beta_ph"])
    print("\nConvergence check (r_hat, should all be ~1.00):")
    print(rhat)


if __name__ == "__main__":
    main()
