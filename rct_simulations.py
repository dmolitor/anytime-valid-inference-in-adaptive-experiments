from pathlib import Path
base_dir = Path(__file__).resolve().parent

import numpy as np
import pandas as pd
import plotnine as pn
from tqdm import tqdm
from typing import List, Tuple

from src.bandit import Reward, TSBernoulli, TSNormal, UCB
from src.mad import MAD, MADModified, MADCovariateAdjusted
from src.model import LassoModel, OLSModel
from src.utils import last

generator = np.random.default_rng(seed=123)

# Import RCT data
rct_data = pd.read_csv(
    base_dir / "data" / "rct_sim" / "SDC - Data - Recoded.csv"
)
intervention_labels = (
    pd.read_csv(
        base_dir / "data" / "rct_sim" / "SDC - Data - Intervention Names.csv"
    )
    .rename(
        columns={
            "Intervention_Name_Data": "label",
            "Intervention_Name_Manuscript": "label_clean"
        }
    )
)
pa_rct_results = pd.merge(
    (
        pd
        .read_csv(
            base_dir / "data" / "rct_sim" / "results_pa.csv"
        )
        [["term", "estimate", "conf.low", "conf.high", "n"]]
        .rename(
            columns={
                "term": "label",
                "estimate": "ate",
                "conf.low": "lb",
                "conf.high": "ub"
            }
        )
        .assign(which="RCT")
    ),
    intervention_labels,
    on="label",
    how="left"
)
outcomes = {
    "PA": "Partisan Animosity",
    "ADA": "Support for Undemocratic Practices",
    "SPV": "Support for Partisan Violence"
}
interventions = dict(enumerate([
    "Null_Control",
    "Alternative_Control",
    "Befriending_Meditation",
    "Chatbot_Quiz",
    "Civity_Storytelling",
    "Common_Identity",
    "Contact_Project",
    "Counterfactual_Selves",
    "Democratic_Fear",
    "Economic_Interests",
    "Empathy_Beliefs",
    "Epistemic_Rescue",
    "Harmful_Experiences",
    "Inparty_Elites",
    "Learning_Goals",
    "Media_Trust",
    "Misperception_Competition",
    "Misperception_Democratic",
    "Misperception_Film",
    "Misperception_Suffering",
    "Moral_Differences",
    "Outparty_Friendship",
    "Partisan_Threat",
    "Party_Overlap",
    "System_Justification",
    "Utah_Cues",
    "Violence_Efficacy"
]))

class SimData:
    def __init__(
        self,
        outcomes: List[str],
        interventions: List[str],
        data: pd.DataFrame
    ):
        self._data = {}
        for outcome in outcomes:
            for intervention in interventions:
                outcome_data = (
                    data
                    .dropna(subset=[outcome])
                    .filter(
                        items=[
                            outcome,
                            "Condition",
                            "Gender",
                            "Age",
                            "Race",
                            "Education",
                            "Inparty_Person",
                            "PI_Pre",
                            "Supplier"
                        ],
                        axis=1
                    )
                )
                intervention_data = (
                    outcome_data[outcome_data["Condition"] == intervention]
                )
                self._data[f"{outcome}_{intervention}"] = intervention_data
    
    def get_data(self, outcome: str, intervention: str):
        """
        Return data for a specific Intervention x Outcome combination.
        """
        return self._data[f"{outcome}_{intervention}"]
    
    def sample_data(self, outcome: str, intervention: str):
        """
        Sample a random row from the data corresponding to the specified
        outcome and intervention combination. This will allow us to effectively
        bootstrap outcome x intervention data.
        """
        data = self.get_data(outcome=outcome, intervention=intervention)
        row = data.sample(n=1)
        return row

# Simulate experiment

# A function to prep the modeling data
def prep_data(X: pd.DataFrame) -> pd.DataFrame:
    X["Education"] = pd.Categorical(
        X["Education"],
        categories=["HS or less", "Some college", "Bachelor", "Postgraduate"]
    )
    X["Gender"] = pd.Categorical(
        X["Gender"],
        categories=["Male", "Female", "Other"]
    )
    X["Race"] = pd.Categorical(
        X["Race"],
        categories=["White", "Black", "Asian", "Hispanic", "Other"]
    )
    X["Inparty_Person"] = pd.Categorical(
        X["Inparty_Person"],
        categories=["Democrat", "Republican"]
    )
    X["Supplier"] = pd.Categorical(
        X["Supplier"],
        categories=["Dynata", "Luth", "Bovitz"]
    )
    for var in ["Education", "Gender", "Race", "Inparty_Person", "Supplier"]:
        dummies = pd.get_dummies(X[var], prefix=var, drop_first=True, dtype=int)
        X = pd.concat([X, dummies], axis=1).drop(var, axis=1)
    X = X.drop("Condition", axis=1)
    return X

## RCT data
sim_data = SimData(
    outcomes=outcomes,
    interventions=list(interventions.values()),
    data=rct_data
)

## Simulation: MADMod vs RCT with extreme adaptivity --------------------------

def reward_fn_covar(arm: int) -> Tuple[float, pd.DataFrame]:
    row = sim_data.sample_data(outcome="PA", intervention=interventions[arm])
    outcome = row["PA"].iloc[0]
    covariates = prep_data(row.drop("PA", axis=1))
    return Reward(outcome=outcome, covariates=covariates)

def reward_fn(arm: int) -> float:
    reward = reward_fn_covar(arm)
    return Reward(outcome=reward.outcome)

mad = MAD(
    bandit=UCB(
        k=len(interventions),
        control=0,
        reward=reward_fn,
        optimize="min"
    ),
    alpha=0.05,
    # delta=lambda x: 1./(x**(0.24*(1-(1/(x**(1/12)))))),
    delta=lambda x: 1./(x**0.2),
    t_star=int(32e3)
)
mad.fit(verbose=True, mc_adjust=None)

mad_covar_adj = MAD(
    bandit=UCB(
        k=len(interventions),
        control=0,
        reward=reward_fn_covar,
        optimize="min"
    ),
    alpha=0.05,
    delta=lambda x: 1./(x**0.2),
    t_star=int(32e3),
    model=OLSModel,
    pooled=True,
    n_warmup=100
)
mad_covar_adj.fit(verbose=True, mc_adjust=None)

# Plot results
pa_results_covar = (
    pd
    .merge(
        mad_covar_adj.estimates(),
        pd.merge(
            pd.DataFrame(interventions.items(), columns=["arm", "label"]),
            intervention_labels,
            on="label",
            how="left"
        ),
        on="arm",
        how="left"
    )
    .assign(which="MADCovar")
)
pa_results_mad = (
    pd
    .merge(
        mad.estimates(),
        pd.merge(
            pd.DataFrame(interventions.items(), columns=["arm", "label"]),
            intervention_labels,
            on="label",
            how="left"
        ),
        on="arm",
        how="left"
    )
    .assign(which="MAD")
)
ate_comparison_df = pd.concat(
    [
        pa_rct_results[pa_rct_results["label"] != "Null_Control"],
        pa_results_covar,
        pa_results_mad
    ],
    axis=0
)
# Plot ATE comparison
(
    pn.ggplot(
        ate_comparison_df,
        pn.aes(
            x="reorder(label_clean, -ate)",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.5))
    + pn.geom_linerange(position=pn.position_dodge(width=0.5))
    + pn.geom_hline(yintercept=0, linetype="dashed", color="black")
    + pn.coord_flip()
    + pn.labs(x="", y="ATE", title="Outcome: Partisan animosity", color="")
    + pn.theme_538()
).save(
    base_dir / "figures" / "sim_ate_comparison.png",
    width=8,
    height=8,
    dpi=300
)
# Plot N comparison
mad_covar_n = (
    pd
    .DataFrame(
        {
            k: last(mad_covar_adj._n[k])
            for k in range(mad_covar_adj._bandit.k())
        }.items(),
        columns=["arm", "n"]
    )
    .assign(
        label=lambda x: x["arm"].apply(lambda y: interventions[y])
    )
    .assign(which="MADCovar")
    .merge(intervention_labels, on="label", how="left")
)
mad_n = (
    pd
    .DataFrame(
        {
            k: last(mad._n[k])
            for k in range(mad._bandit.k())
        }.items(),
        columns=["arm", "n"]
    )
    .assign(
        label=lambda x: x["arm"].apply(lambda y: interventions[y])
    )
    .assign(which="MAD")
    .merge(intervention_labels, on="label", how="left")
)
rct_n = pa_rct_results[["label", "label_clean", "n", "which"]]
n_df = pd.concat([mad_covar_n, mad_n, rct_n], axis=0)
(
    pn.ggplot(n_df, pn.aes(x="reorder(label_clean, -n)", y="n", fill="which"))
    + pn.geom_bar(stat="identity", position=pn.position_dodge())
    + pn.coord_flip()
    + pn.theme_538()
    + pn.labs(x="", y="N", title="Outcome: Partisan animosity", fill="")
).save(
    base_dir / "figures" / "sim_n_comparison.png",
    width=8,
    height=8,
    dpi=300
)

## Simulation: MAD vs RCT with more balanced adaptivity -----------------------

mad = MAD(
    bandit=UCB(
        k=len(interventions),
        control=0,
        reward=reward_fn,
        optimize="min"
    ),
    alpha=0.05,
    delta=lambda x: 1./(x**0.05),
    t_star=int(32e3)
)
mad.fit(verbose=True, mc_adjust=None)

mad_covar_adj = MADCovariateAdjusted(
    bandit=UCB(
        k=len(interventions),
        control=0,
        reward=reward_fn_covar,
        optimize="min"
    ),
    model=OLSModel,
    pooled=True,
    n_warmup=100, 
    alpha=0.05,
    delta=lambda x: 1./(x**0.05),
    t_star=int(32e3)
)
mad_covar_adj.fit(verbose=True, mc_adjust=None)

# Plot results
pa_results_covar = (
    pd
    .merge(
        mad_covar_adj.estimates(),
        pd.merge(
            pd.DataFrame(interventions.items(), columns=["arm", "label"]),
            intervention_labels,
            on="label",
            how="left"
        ),
        on="arm",
        how="left"
    )
    .assign(which="MADCovar")
)
pa_results_mad = (
    pd
    .merge(
        mad.estimates(),
        pd.merge(
            pd.DataFrame(interventions.items(), columns=["arm", "label"]),
            intervention_labels,
            on="label",
            how="left"
        ),
        on="arm",
        how="left"
    )
    .assign(which="MAD")
)
ate_comparison_df = pd.concat(
    [
        pa_rct_results[pa_rct_results["label"] != "Null_Control"],
        pa_results_covar,
        pa_results_mad
    ],
    axis=0
)
# Plot ATE comparison
(
    pn.ggplot(
        ate_comparison_df,
        pn.aes(
            x="reorder(label_clean, -ate)",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.5))
    + pn.geom_linerange(position=pn.position_dodge(width=0.5))
    + pn.geom_hline(yintercept=0, linetype="dashed", color="black")
    + pn.coord_flip()
    + pn.labs(x="", y="ATE", title="Outcome: Partisan animosity", color="")
    + pn.theme_538()
).save(
    base_dir / "figures" / "sim_ate_comparison_low_adapt.png",
    width=8,
    height=8,
    dpi=300
)
# Plot N comparison
mad_covar_n = (
    pd
    .DataFrame(
        {
            k: last(mad_covar_adj._n[k])
            for k in range(mad_covar_adj._bandit.k())
        }.items(),
        columns=["arm", "n"]
    )
    .assign(
        label=lambda x: x["arm"].apply(lambda y: interventions[y])
    )
    .assign(which="MADCovar")
    .merge(intervention_labels, on="label", how="left")
)
mad_n = (
    pd
    .DataFrame(
        {
            k: last(mad._n[k])
            for k in range(mad._bandit.k())
        }.items(),
        columns=["arm", "n"]
    )
    .assign(
        label=lambda x: x["arm"].apply(lambda y: interventions[y])
    )
    .assign(which="MAD")
    .merge(intervention_labels, on="label", how="left")
)
rct_n = pa_rct_results[["label", "label_clean", "n", "which"]]
n_df = pd.concat([mad_covar_n, mad_n, rct_n], axis=0)
(
    pn.ggplot(n_df, pn.aes(x="reorder(label_clean, -n)", y="n", fill="which"))
    + pn.geom_bar(stat="identity", position=pn.position_dodge())
    + pn.coord_flip()
    + pn.theme_538()
    + pn.labs(x="", y="N", title="Outcome: Partisan animosity", fill="")
).save(
    base_dir / "figures" / "sim_n_comparison_low_adapt.png",
    width=8,
    height=8,
    dpi=300
)

## Simulation: MADMod vs MAD vs RCT -------------------------------------------

mad_covar_adj = MAD(
    bandit=UCB(
        k=len(interventions),
        control=0,
        reward=reward_fn_covar,
        optimize="min"
    ),
    alpha=0.05,
    delta=lambda x: 1./(x**0.05),
    t_star=int(32e3),
    model=OLSModel,
    pooled=True,
    n_warmup=100
)
mad_covar_adj.fit(verbose=True, mc_adjust=None)

mad_mod = MADModified(
    bandit=UCB(
        k=len(interventions),
        control=0,
        reward=reward_fn_covar,
        optimize="min"
    ),
    alpha=0.05,
    delta=lambda x: 1./(x**0.05),
    t_star=int(32e3),
    decay=lambda x: 1./(x**(1./8.)),
    model=OLSModel,
    pooled=True,
    n_warmup=100
)
mad_mod.fit(verbose=True, mc_adjust=None)

# Plot results
pa_results_covar = (
    pd
    .merge(
        mad_covar_adj.estimates(),
        pd.merge(
            pd.DataFrame(interventions.items(), columns=["arm", "label"]),
            intervention_labels,
            on="label",
            how="left"
        ),
        on="arm",
        how="left"
    )
    .assign(which="MADCovar")
)
pa_results_mad_mod = (
    pd
    .merge(
        mad_mod.estimates(),
        pd.merge(
            pd.DataFrame(interventions.items(), columns=["arm", "label"]),
            intervention_labels,
            on="label",
            how="left"
        ),
        on="arm",
        how="left"
    )
    .assign(which="MADMod")
)
ate_comparison_df = pd.concat(
    [
        pa_rct_results[pa_rct_results["label"] != "Null_Control"],
        pa_results_covar,
        pa_results_mad_mod
    ],
    axis=0
)
# Plot ATE comparison
(
    pn.ggplot(
        ate_comparison_df,
        pn.aes(
            x="reorder(label_clean, -ate)",
            y="ate",
            ymin="lb",
            ymax="ub",
            color="which"
        )
    )
    + pn.geom_point(position=pn.position_dodge(width=0.5))
    + pn.geom_linerange(position=pn.position_dodge(width=0.5))
    + pn.geom_hline(yintercept=0, linetype="dashed", color="black")
    + pn.coord_flip()
    + pn.labs(x="", y="ATE", title="Outcome: Partisan animosity", color="")
    + pn.theme_538()
).save(
    base_dir / "figures" / "sim_ate_comparison_low_adapt.png",
    width=8,
    height=8,
    dpi=300
)
# Plot N comparison
mad_covar_n = (
    pd
    .DataFrame(
        {
            k: last(mad_covar_adj._n[k])
            for k in range(mad_covar_adj._bandit.k())
        }.items(),
        columns=["arm", "n"]
    )
    .assign(
        label=lambda x: x["arm"].apply(lambda y: interventions[y])
    )
    .assign(which="MADCovar")
    .merge(intervention_labels, on="label", how="left")
)
mad_mod_n = (
    pd
    .DataFrame(
        {
            k: last(mad_mod._n[k])
            for k in range(mad_mod._bandit.k())
        }.items(),
        columns=["arm", "n"]
    )
    .assign(
        label=lambda x: x["arm"].apply(lambda y: interventions[y])
    )
    .assign(which="MADMod")
    .merge(intervention_labels, on="label", how="left")
)
rct_n = pa_rct_results[["label", "label_clean", "n", "which"]]
n_df = pd.concat([mad_covar_n, mad_mod_n, rct_n], axis=0)
(
    pn.ggplot(n_df, pn.aes(x="reorder(label_clean, -n)", y="n", fill="which"))
    + pn.geom_bar(stat="identity", position=pn.position_dodge())
    + pn.coord_flip()
    + pn.theme_538()
    + pn.labs(x="", y="N", title="Outcome: Partisan animosity", fill="")
).save(
    base_dir / "figures" / "sim_n_comparison_low_adapt.png",
    width=8,
    height=8,
    dpi=300
)