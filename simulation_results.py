from pathlib import Path
base_dir = Path(__file__).resolve().parent

from itertools import product
import joblib
import math
import numpy as np
import pandas as pd
import plotnine as pn
from pyssed import MAD, MADMod
from scipy.stats import t
from tqdm import tqdm
from typing import Tuple

from src.bandit import Reward, TSNormal, TSBernoulli
from src.model import FastOLSModel, OLSModel
from src.utils import last

generator = np.random.default_rng(seed=123)

# Define my own slightly customized theme
def theme_daniel():
    return (
        pn.theme_light()
        + pn.theme(
            legend_key=pn.element_blank(),
            panel_border=pn.element_blank(),
            strip_background=pn.element_rect(fill="white", color="white"),
            strip_text=pn.element_text(color="black")
        )
    )

# Reward function that scales the strength of the covariate signal and 
# ITE heterogeneity
def reward_covar_adj(arm: int, covariate_signal: float = 1.) -> Reward:
    ate = {0: 0.0, 1: 1.0}
    # Draw covariates
    X1 = np.random.randn()
    X2 = np.random.randn()
    X3 = np.random.randn()
    X_df = pd.DataFrame({"X_1": [X1], "X_2": [X2], "X_3": [X3]})
    # Base outcome model (covariate signal scaled)
    cov_effect = covariate_signal * (0.3 * X1 + X2 - 0.5 * X3)
    # Treatment effect
    tau_i = ate[arm]
    # Generate outcome
    mean = tau_i + cov_effect
    Y_i = generator.normal(mean, 1)
    return Reward(outcome=float(Y_i), covariates=X_df)


def reward_vanilla(arm: int) -> Reward:
    reward = reward_covar_adj(arm=arm)
    reward = Reward(outcome=reward.outcome)
    return reward

# Figure 1 --------------------------------------------------------------------

# Vanilla MAD algorithm for 10000 iterations
mad = MAD(
    bandit=TSNormal(k=2, control=0, reward=reward_vanilla),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(10e3)
)
mad.fit(verbose=True, early_stopping=False, mc_adjust=None)

# Covariate adjusted MAD algorithm for 10000 iterations
mad_covar_adj = MAD(
    bandit=TSNormal(k=2, control=0, reward=reward_covar_adj),
    alpha=0.05,
    delta=lambda x: 1./(x**0.24),
    t_star=int(10e3),
    model=OLSModel,
    pooled=False,
    n_warmup=50
)
mad_covar_adj.fit(
    verbose=True,
    early_stopping=False,
    mc_adjust=None
)

# Compare the ATE path width of the two methods
estimates = []
for which, mad_alg in enumerate([mad, mad_covar_adj]):
    ates = mad_alg._ate[1]
    radii = mad_alg._cs_radius[1]
    ubs = np.nan_to_num([x + y for (x, y) in zip(ates, radii)], nan=np.inf)
    lbs = np.nan_to_num([x - y for (x, y) in zip(ates, radii)], nan=-np.inf)
    estimates_df = pd.DataFrame({
        "ate": ates,
        "lb": lbs,
        "ub": ubs,
        "t": range(1, len(ates) + 1),
        "which": which
    })
    estimates.append(estimates_df)
estimates = (
    pd
    .concat(estimates)
    .assign(
        which=lambda df: (
            df["which"]
            .apply(lambda x: "MAD" if x == 0 else "MADCovar")
        )
    )
)

figure_1 = (
    pn.ggplot(
        data=estimates,
        mapping=pn.aes(
            x="t",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which",
            fill="which"
        )
    )
    + pn.geom_line()
    + pn.geom_hline(yintercept=1.0, linetype="dashed")
    + pn.geom_linerange(alpha=0.01)
    + pn.coord_cartesian(ylim=(-0, 2))
    + theme_daniel()
    + pn.labs(
        y="ATE",
        color="",
        fill=""
    )
)
figure_1.save(
    base_dir / "figures" / "figure1.png",
    width=6,
    height=3,
    dpi=300
)

# Figure 2 --------------------------------------------------------------------

def reward_fn(arm: int, covariate_signal: float = 1.0, noise_level: str = "low") -> Reward:
    """
    noise_level \in {"low", "medium", "high"}:
      - "low": 2 noise covariates (total d=5)
      - "medium": 12 noise covariates (total d=15)
      - "high": 27 noise covariates (total d=30)
    """
    ate = {0: 0.0, 1: 1.0}
    # Map noise_level to noise count
    noise_map = {"low": 2, "medium": 22, "high": 47}
    n_noise = noise_map[noise_level]
    # Draw signal covariates
    X_sig = generator.normal(size=3)
    # Draw noise covariates
    X_noise = generator.normal(size=n_noise)
    # Build DataFrame
    cols = [f"X_{i+1}" for i in range(3)] + [f"X_noise_{j+1}" for j in range(n_noise)]
    X_df = pd.DataFrame([np.concatenate([X_sig, X_noise])], columns=cols)
    # Signal effect only from first 3 covariates
    beta = np.array([2.3, 0.9, -1.7])
    cov_effect = covariate_signal * (beta @ X_sig)
    # Treatment effect (no heterogeneity here)
    tau_i = ate[arm]
    # Outcome
    mean = 0.5 + tau_i + cov_effect
    Y_i = generator.normal(mean, 1)
    return Reward(outcome=float(Y_i), covariates=X_df)

def reward_vanilla(arm: int, covariate_signal: float = 1.0, noise_level: str = "low") -> float:
    return Reward(
        outcome=reward_fn(
            arm=arm,
            covariate_signal=covariate_signal,
            noise_level=noise_level
        ).outcome
    )

def compare_methods(
    i,
    reward,
    reward_vanilla,
    t_star,
    covar_signal,
    noise_level,
    verbose=False
):
    # No covariate adjustment
    mad_no_adjust = MAD(
        bandit=TSNormal(
            k=2,
            control=0,
            reward=lambda x: reward_vanilla(x, covar_signal, noise_level)
        ),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.2),
        t_star=t_star
    )
    mad_no_adjust.fit(verbose=verbose, early_stopping=False, mc_adjust=None)
    # Covariate adjustment
    mad_adjust = MAD(
        bandit=TSNormal(
            k=2,
            control=0,
            reward=lambda x: reward(x, covar_signal, noise_level)
        ),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.2),
        t_star=t_star,
        model=FastOLSModel,
        pooled=True,
        n_warmup=50
    )
    mad_adjust.fit(verbose=verbose, early_stopping=False, mc_adjust=None)
    interval_width_covar = pd.concat([
        pd.DataFrame({
            "arm": [k]*len(mad_adjust._cs_radius[k]),
            "ate_estimate": mad_adjust._ate[k],
            "iter": range(len(mad_adjust._cs_radius[k])),
            "interval_width": [2*x for x in mad_adjust._cs_radius[k]],
            "idx": [i]*len(mad_adjust._cs_radius[k]),
            "method": ["MADCovar"]*len(mad_adjust._cs_radius[k]),
            "covar_strength": [covar_signal]*len(mad_adjust._cs_radius[k]),
            "noise_level": [noise_level]*len(mad_adjust._cs_radius[k])
        })
        for k in range(1, 2)
    ]).reset_index(drop=True)

    interval_width_nocovar = pd.concat([
        pd.DataFrame({
            "arm": [k]*len(mad_no_adjust._cs_radius[k]),
            "ate_estimate": mad_no_adjust._ate[k],
            "iter": range(len(mad_no_adjust._cs_radius[k])),
            "interval_width": [2*x for x in mad_no_adjust._cs_radius[k]],
            "idx": [i]*len(mad_no_adjust._cs_radius[k]),
            "method": ["MAD"]*len(mad_no_adjust._cs_radius[k]),
            "covar_strength": [covar_signal]*len(mad_no_adjust._cs_radius[k]),
            "noise_level": [noise_level]*len(mad_no_adjust._cs_radius[k])
        })
        for k in range(1, 2)
    ]).reset_index(drop=True)
    return pd.concat([interval_width_covar, interval_width_nocovar], ignore_index=True)
    
# Run interval width and MSE simulations
covar_strength = [0.1, 0.5, 1.0]
noise_levels = ["low", "medium", "high"]
strength_grid = list(product(covar_strength, noise_levels))
n_sim = 100

# Simulate MADCovar improvments across a grid of DGPs
sim_results_list = []
for tup in strength_grid:
    print(f"Covariate strength: {tup[0]}; Noise level: {tup[1]}")
    sim_results = [
        x for x in
        tqdm(
            joblib.Parallel(return_as="generator", n_jobs=-1)(
                joblib.delayed(compare_methods)(
                    i,
                    reward=reward_fn,
                    reward_vanilla=reward_vanilla,
                    t_star=int(1e4),
                    covar_signal=tup[0],
                    noise_level=tup[1]
                ) for i in range(n_sim)
            ),
            total=n_sim
        )
    ]
    sim_results_list.extend(sim_results)

# Aggregate results
sim_results_df = pd.concat(sim_results_list, ignore_index=True)
sim_results_df["interval_width"] = (
    sim_results_df["interval_width"].replace([np.inf, -np.inf], np.nan)
)
summary = (
    sim_results_df
    .groupby(["arm", "iter", "method", "covar_strength", "noise_level"])
    .agg(
        interval_width_mean=("interval_width", "mean"),
        interval_width_se=("interval_width", lambda x: x.std(ddof=1) / np.sqrt(x.count()))
    )
    .reset_index()
)
summary = (
    summary
    .melt(id_vars=["arm", "iter", "method", "covar_strength", "noise_level"],
          value_vars=[
              "interval_width_mean", "interval_width_se"
          ],
          var_name="name", value_name="value")
    .assign(
        metric=lambda d: d["name"].str.extract(r"^(interval_width)")[0],
        stat=lambda d: d["name"].str.extract(r"_(mean|se)$")[0]
    )
    .drop(columns="name")
    .pivot(index=["arm", "iter", "method", "covar_strength", "noise_level", "metric"], columns="stat", values="value")
    .reset_index()
)
summary["ci_low"]  = summary["mean"] - 2.33 * summary["se"]
summary["ci_high"] = summary["mean"] + 2.33 * summary["se"]
summary = (
    summary
    .assign(
        cstrength_label = lambda df: df["covar_strength"].apply(lambda y: {0.1: "Signal: Low", 0.5: "Signal: Medium", 1.0: "Signal: High"}[y]),
        nstrength_label = lambda df: df["noise_level"].apply(lambda y: {"low": "Noise: Low (d = 5)", "medium": "Noise: Medium (d = 25)", "high": "Noise: High (d = 50)"}[y])
    )
    .assign(
        cstrength_label = lambda df: pd.Categorical(df["cstrength_label"], categories=["Signal: Low", "Signal: Medium", "Signal: High"], ordered=True),
        nstrength_label = lambda df: pd.Categorical(df["nstrength_label"], categories=["Noise: Low (d = 5)", "Noise: Medium (d = 25)", "Noise: High (d = 50)"], ordered=True)
    )
)

# Plot it
(
    pn.ggplot(
        summary[(summary["iter"] % 200 == 0) & (summary["metric"] == "interval_width")],
        pn.aes(x="iter", y="mean", ymin="ci_low", ymax="ci_high", color="method", fill="method")
    )
    + pn.geom_point(size = 0.2)
    # + pn.geom_errorbar()
    + pn.geom_line(alpha = 0.3, size=0.5)
    + theme_daniel()
    + pn.coord_cartesian(ylim=(0, 4))
    + pn.facet_wrap("~ cstrength_label + nstrength_label")
    + pn.labs(x="t", y=u"Mean CS width", color="", fill = "")
).save(
    filename=base_dir / "figures" / "figure2.png",
    width=8,
    height=6,
    dpi=500
)

# Figures 4 and 5 -------------------------------------------------------------

# Reward function for simulations
#
# We demonstrate this with an experiment simulating a control arm and four
# treatment arms with ATEs of 0.1, 0.12, 0.3, and 0.32, respectively, over a
# fixed sample size of 20,000. We expect the bandit algorithm to allocate most of
# the sample to arms 3 and 4, leaving arms 1 and 2 under-powered.
def reward_fn(arm: int) -> float:
    values = {
        0: generator.binomial(1, 0.5),  # Control arm
        1: generator.binomial(1, 0.6),  # ATE = 0.1
        2: generator.binomial(1, 0.62), # ATE = 0.12
        3: generator.binomial(1, 0.8),  # ATE = 0.3
        4: generator.binomial(1, 0.82)  # ATE = 0.32
    }
    return Reward(outcome=values[arm])

# Simulation results over 1,0000 runs
# 
# We can more precisely quantify the improvements by running 1,000 simulations,
# comparing Type 2 error and confidence band width between the vanilla MAD
# algorithm and the modified algorithm. Each simulation runs for 20,000
# iterations with early stopping. If the modified algorithm stops early, the
# vanilla algorithm will also stop early to maintain equal sample sizes in each
# simulation.
def compare(i):
    mad_modified = MADMod(
        bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=int(3e4),
        decay=lambda x: 1. / (x ** (1. / 8.))
    )
    mad_modified.fit(cs_precision=0.1, verbose=False, early_stopping=True)

    # Run the vanilla algorithm
    mad_vanilla = MAD(
        bandit=TSBernoulli(k=5, control=0, reward=reward_fn),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=mad_modified._bandit._t
    )
    mad_vanilla.fit(verbose=False, early_stopping=False)

    # Calculate the Type 2 error and the Confidence Sequence width

    ## For modified algorithm
    mad_mod_n = (
        pd
        .DataFrame([
            {"arm": k, "n": last(mad_modified._n[k])}
            for k in range(mad_modified._bandit.k())
            if k != mad_modified._bandit.control()
        ])
        .assign(
            n_pct=lambda x: x["n"].apply(lambda y: y/np.sum(x["n"]))
        )
    )
    mad_mod_df = (
        mad_modified
        .estimates()
        .assign(
            idx=i,
            method="modified",
            width=lambda x: x["ub"] - x["lb"],
            error=lambda x: ((0 > x["lb"]) & (0 < x["ub"]))
        )
        .merge(mad_mod_n, on="arm", how="left")
    )

    ## For vanilla algorithm
    mad_van_n = (
        pd
        .DataFrame([
            {"arm": k, "n": last(mad_vanilla._n[k])}
            for k in range(mad_vanilla._bandit.k())
            if k != mad_vanilla._bandit.control()
        ])
        .assign(
            n_pct=lambda x: x["n"].apply(lambda y: y/np.sum(x["n"]))
        )
    )
    mad_van_df = (
        mad_vanilla
        .estimates()
        .assign(
            idx=i,
            method="mad",
            width=lambda x: x["ub"] - x["lb"],
            error=lambda x: ((0 > x["lb"]) & (0 < x["ub"]))
        )
        .merge(mad_van_n, on="arm", how="left")
    )

    out = {
        "metrics": pd.concat([mad_mod_df, mad_van_df]),
        "reward": {
            "modified": np.sum(mad_modified._rewards),
            "mad": np.sum(mad_vanilla._rewards)
        }
    }
    return out

# Execute in parallel with joblib
comparison_results_list = [
    x for x in
    joblib.Parallel(return_as="generator", n_jobs=-1)(
        joblib.delayed(compare)(i) for i in range(100)
    )
]

# Compare performance on key metrics across simulations
metrics_df = pd.melt(
    (
        pd
        .concat([x["metrics"] for x in comparison_results_list])
        .reset_index(drop=True)
        .assign(error=lambda x: x["error"].apply(lambda y: int(y)))
    ),
    id_vars=["arm", "method"],
    value_vars=["width", "error", "n", "n_pct"],
    var_name="meas",
    value_name="value"
)
metrics_df["method"] = (
    metrics_df["method"]
    .apply(lambda x: {"modified": "MADMod", "mad": "MAD"}[x])
)
metrics_summary = (
    metrics_df
    .groupby(["arm", "method", "meas"], as_index=False).agg(
        mean=("value", "mean"),
        std=("value", "std"),
        n=("value", "count")
    )
    .assign(
        se=lambda x: x["std"] / np.sqrt(x["n"]),
        t_val=lambda x: t.ppf(0.975, x["n"] - 1),
        ub=lambda x: x["mean"] + x["t_val"] * x["se"],
        lb=lambda x: x["mean"] - x["t_val"] * x["se"]
    )
    .drop(columns=["se", "t_val"])
)

# Plot Figure 4 ---------------------------------

facet_labels = {
    "error": "Type 2 error",
    "width": "Interval width",
    "n": "Sample size",
    "n_pct": "Sample size %"
}
(
    pn.ggplot(
        metrics_summary[metrics_summary["meas"].isin(["error", "width"])],
        pn.aes(
            x="factor(arm)",
            y="mean",
            ymin="lb",
            ymax="ub",
            color="method"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.2), size=0.7)
    + pn.geom_linerange(position=pn.position_dodge(width=0.2))
    + pn.facet_wrap(
        "~ meas",
        labeller=lambda x: facet_labels[x],
        scales="free"
    )
    + theme_daniel()
    + pn.labs(x="Arm", y="", color="Method")
).save(
    base_dir / "figures" / "figure4.png",
    width=5,
    height=2,
    dpi=500
)

# Plot Figure 5 ---------------------------------

def custom_breaks(lim, n=4):
    unit = 0.1 if lim[1] <= 1 else 1000
    raw = np.linspace(lim[0], lim[1], n)
    br = np.ceil(raw / unit) * unit
    if unit == 0.1:
        return [round(float(v), 1) for v in sorted(set(br))]
    return [int(v) for v in sorted(set(br))]

(
    pn.ggplot(
        metrics_summary[metrics_summary["meas"].isin(["n", "n_pct"])],
        pn.aes(
            x="factor(arm)",
            y="mean",
            ymin="lb",
            ymax="ub",
            color="method"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.2), size=0.7)
    + pn.geom_linerange(position=pn.position_dodge(width=0.2))
    + pn.facet_wrap(
        "~ meas",
        labeller=lambda x: facet_labels[x],
        scales="free"
    )
    + theme_daniel()
    + pn.scale_y_continuous(
        breaks=lambda limits: list(np.arange(
            math.floor(limits[0] / (0.2 if limits[1] <= 1 else 3000))
            * (0.2 if limits[1] <= 1 else 3000),
            math.ceil(limits[1] / (0.2 if limits[1] <= 1 else 3000))
            * (0.2 if limits[1] <= 1 else 3000)
            + (0.2 if limits[1] <= 1 else 3000),
            (0.2 if limits[1] <= 1 else 3000)
        ))
    )
    + pn.labs(x="Arm", y="", color="Method")
).save(
    base_dir / "figures" / "figure5.png",
    width=5,
    height=2,
    dpi=500
)

# Appendix Table 1 ------------------------------------------------------------

# Well-specified outcome models

def reward_fn(arm: int) -> Tuple[float, pd.DataFrame]:
    ate = {
        0: 0.0,
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4,
        5: 0.5
    }
    # Draw X values randomly (here using standard normal distribution)
    X1 = np.random.randn()
    X2 = np.random.randn()
    X3 = np.random.randn()
    # Get the corresponding ATE from the dictionary
    ate = ate[arm]
    # Compute Y_i using the given model
    mean = 0.5 + ate + 0.3 * X1 + 1.0 * X2 - 0.5 * X3
    Y_i = generator.normal(mean, 1)
    X_df = pd.DataFrame({"X_1": [X1], "X_2": [X2], "X_3": [X3]})
    return Reward(outcome=float(Y_i), covariates=X_df)

def reward_vanilla(arm: int) -> float:
    return Reward(outcome=reward_fn(arm=arm).outcome)

def compare_type1_error(i, reward, t_star, ate_truth, verbose=False):
    # No multiple comparison adjustment
    mad_no_adjust = MAD(
        bandit=TSNormal(k=6, control=0, reward=reward),
        alpha=0.05,
        delta=lambda x: 1. / (x ** 0.24),
        t_star=t_star,
        model=FastOLSModel,
        pooled=True,
        n_warmup=50
    )
    mad_no_adjust.fit(verbose=verbose, early_stopping=False, mc_adjust=None)

    type1_error = pd.concat([
        pd.DataFrame({
            "arm": [k]*1,
            "error": [
                np.any(
                    [
                        not (x - y) < ate_truth[k] < (x + y) for x, y
                        in zip(mad_no_adjust._ate[k], mad_no_adjust._cs_radius[k])
                        if not np.isnan(x) and not np.isinf(y)
                    ]
                )
            ],
            "adjustment_method": ["None"],
            "method": ["MADMod"]*1,
            "idx": [i]*1
        })
        for k in range(1, 6)
    ]).reset_index(drop=True)
    
    return type1_error

n_sim = 1000
type1_error_sim = [
    x for x in
    tqdm(
        joblib.Parallel(return_as="generator", n_jobs=-1)(
            joblib.delayed(compare_type1_error)(
                i,
                reward=reward_fn,
                t_star=int(1e4),
                ate_truth={
                    0: 0.0,
                    1: 0.1,
                    2: 0.2,
                    3: 0.3,
                    4: 0.4,
                    5: 0.5
                }
            ) for i in range(n_sim)
        ),
        total=n_sim
    )
]

# Calculate marginal coverage of true parameters
coverage_error_df = pd.concat(type1_error_sim, ignore_index=True)

individual_type1_error = (
    coverage_error_df
    .groupby(["method", "adjustment_method", "arm"], as_index=False)
    .agg(coverage=("error", "mean"), n=("error", "count"))
)
individual_type1_error["se"] = (
    np.sqrt(individual_type1_error["coverage"]
    * (1 - individual_type1_error["coverage"])
    / individual_type1_error["n"])
)
individual_type1_error["ci_lower"] = (
    individual_type1_error["coverage"]
    - 1.96 * individual_type1_error["se"]
)
individual_type1_error["ci_upper"] = (
    individual_type1_error["coverage"]
    + 1.96 * individual_type1_error["se"]
)
individual_type1_error["coverage_type"] = "Marginal"
well_specified_coverage = (
    individual_type1_error
    .assign(arm=lambda df: df["arm"].apply(lambda arm: f"Arm {arm}"))
    .assign(
        coverage_error=lambda df: df.apply(
            lambda d: f"{round(d['coverage'], 3)} ({round(d['ci_lower'], 3)}, {round(d['ci_upper'], 3)})",
            axis=1
        )
    )
    .filter(items=["arm", "coverage_error", "n"])
    .assign(model = "well-specified")
)

# Mis-specified outcome models

def reward_fn(arm: int) -> Tuple[float, pd.DataFrame]:
    ate = {
        0: 0.0,
        1: 0.1,
        2: 0.2,
        3: 0.3,
        4: 0.4,
        5: 0.5
    }
    # Draw X values randomly (here using standard normal distribution)
    X1 = np.random.randn()
    X2 = np.random.randn()
    X3 = np.random.randn()
    # Get the corresponding ATE from the dictionary
    ate = ate[arm]
    # Compute Y_i using the given model
    mean = 0.5 + ate + 0.3 * X1**2 + 1.0 * X2*X3 - 0.5 * np.exp(X3)
    Y_i = generator.normal(mean, 1)
    X_df = pd.DataFrame({"X_1": [X1], "X_2": [X2], "X_3": [X3]})
    return Reward(outcome=float(Y_i), covariates=X_df)

def reward_vanilla(arm: int) -> float:
    return Reward(outcome=reward_fn(arm=arm).outcome)

n_sim = 1000
type1_error_sim_mis = [
    x for x in
    tqdm(
        joblib.Parallel(return_as="generator", n_jobs=-1)(
            joblib.delayed(compare_type1_error)(
                i,
                reward=reward_fn,
                t_star=int(1e4),
                ate_truth={
                    0: 0.0,
                    1: 0.1,
                    2: 0.2,
                    3: 0.3,
                    4: 0.4,
                    5: 0.5
                }
            ) for i in range(n_sim)
        ),
        total=n_sim
    )
]

# Calculate marginal coverage of true parameters
coverage_error_mis_df = pd.concat(type1_error_sim_mis, ignore_index=True)

individual_type1_error = (
    coverage_error_mis_df
    .groupby(["method", "adjustment_method", "arm"], as_index=False)
    .agg(coverage=("error", "mean"), n=("error", "count"))
)
individual_type1_error["se"] = (
    np.sqrt(individual_type1_error["coverage"]
    * (1 - individual_type1_error["coverage"])
    / individual_type1_error["n"])
)
individual_type1_error["ci_lower"] = (
    individual_type1_error["coverage"]
    - 1.96 * individual_type1_error["se"]
)
individual_type1_error["ci_upper"] = (
    individual_type1_error["coverage"]
    + 1.96 * individual_type1_error["se"]
)
individual_type1_error["coverage_type"] = "Marginal"
misspecified_coverage = (
    individual_type1_error
    .assign(arm=lambda df: df["arm"].apply(lambda arm: f"Arm {arm}"))
    .assign(
        coverage_error=lambda df: df.apply(
            lambda d: f"{round(d['coverage'], 3)} ({round(d['ci_lower'], 3)}, {round(d['ci_upper'], 3)})",
            axis=1
        )
    )
    .filter(items=["arm", "coverage_error", "n"])
    .assign(model = "mis-specified")
)

# Table 1 ---------------------------------------

table1 = pd.concat([well_specified_coverage, misspecified_coverage])
table1.to_csv(base_dir / "figures" / "table1.csv", index=False)
