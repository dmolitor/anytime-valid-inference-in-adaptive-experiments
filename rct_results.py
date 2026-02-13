from pathlib import Path
base_dir = Path(__file__).resolve().parent

from io import StringIO
import numpy as np
import pandas as pd
import plotnine as pn
from pyssed import MAD
import statsmodels.formula.api as smf
from typing import List

from src.av_utils import sequential_f_cs
from src.bandit import (
    Reward,
    TSNormal,
)
from src.model import FastOLSModel
from src.utils import last

generator = np.random.default_rng(seed=123)

# Define my own slightly customized theme
def theme_daniel():
    return (
        pn.theme_light()
        + pn.theme(
            legend_key=pn.element_blank(),
            panel_border=pn.element_blank()
        )
    )

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
        self._data_raw = {}
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
                self._data_raw[outcome] = outcome_data
    
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
    
    def sample_data_outcome(self, outcome: str, n: int):
        """
        Sample a random row from the data corresponding to the specified
        outcome. This will allow us to effectively bootstrap outcome data.
        """
        data = self._data_raw[outcome]
        rows = data.sample(n=n, replace=True)
        return rows

class RCT:
    def __init__(self, alpha: float, n: int, outcome: str = "PA"):
        self._alpha = alpha
        self._covariates = None
        self._estimates = None
        self._n = n
        self._outcome = outcome
    
    def estimates(self) -> pd.DataFrame:
        return self._estimates

    def fit(self):
        model_data = (
            rct_data.dropna(subset=self._outcome)
            .sample(n=self._n, replace=True)
        )
        self._covariates = model_data.copy()
        model_data["Condition"] = pd.Categorical(
            model_data.copy()["Condition"],
            categories=list(interventions.values())
        )
        # Fit OLS model
        model = (
            smf
            .ols(
                "PA ~ 1 + Condition + Gender + Age + Race + Education + Inparty_Person + PI_Pre + Supplier",
                data=model_data
            )
            .fit()
        )
        # Extract coefficients
        coefs = pd.read_html(
            StringIO(model.summary(alpha=self._alpha).tables[1].as_html()),
            header=0
        )[0]
        coefs.columns = [
            "label",
            "ate",
            "std.error",
            "statistic",
            "p.value",
            "lb",
            "ub"
        ]
        # Calculate anytime-valid CSs and p-values for the model coefficients
        coefs = sequential_f_cs(
            delta=coefs["ate"].values,
            se=coefs["std.error"].values,
            n=int(model.nobs),
            n_params=len(model.params),
            Z=np.linalg.inv(model.cov_params().values),
            alpha=self._alpha,
            phi=1,
            term=coefs["label"].values
        )
        # Tidy coefficients
        coefs = (
            coefs[coefs["term"].str.startswith("Condition")]
            .assign(
                label=lambda df: (
                    df["term"]
                    .str
                    .replace(r"^Condition\[T\.", "", regex=True)
                    .str
                    .replace(r"\]$", "", regex=True)
                ),
                which="RCT*"
            )
            .rename(columns={"estimate": "ate", "cs_lower": "lb", "cs_upper": "ub"})
            [["label", "ate", "lb", "ub", "which"]]
        )
        # Calculate anytime-valid CSs and p-values for all 
        self._estimates = coefs.reset_index(drop=True)

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

## Simulation data
sim_data = SimData(
    outcomes=outcomes,
    interventions=list(interventions.values()),
    data=rct_data
)

## RCT simulation
rct = RCT(alpha=0.05, n=int(32e3))
rct.fit()
pa_results_rct = (
    rct
    .estimates()
    .merge(
        intervention_labels,
        on="label",
        how="left"
    )
    .merge(
        pd.read_csv(base_dir / "data" / "rct_sim" / "results_pa.csv"),
        left_on="label",
        right_on="term",
        how="left"
    )
)
rct_n = (
    pd
    .DataFrame(rct._covariates["Condition"].value_counts())
    .reset_index()
    .rename(columns={"Condition": "label", "count": "n"})
    .assign(which="RCT*")
    .merge(intervention_labels, on="label", how="left")
)

## Reward functions for simulations
def reward_fn_covar(arm: int) -> Reward:
    row = sim_data.sample_data(outcome="PA", intervention=interventions[arm])
    outcome = row["PA"].iloc[0]
    covariates = prep_data(row.drop("PA", axis=1))
    return Reward(outcome=outcome, covariates=covariates)

def reward_fn(arm: int) -> float:
    reward = reward_fn_covar(arm)
    return Reward(outcome=reward.outcome)

## Simulation: Focus on adaptivity; best arm identification and exploitation ##
## ------------------------------------------------------------------------- ##

mad = MAD(
    bandit=TSNormal(
        k=len(interventions),
        control=0,
        reward=reward_fn,
        optimize="min"
    ),
    alpha=0.05,
    delta=lambda x: 1./(x**0.1),
    t_star=int(32e3)
)
mad.fit(verbose=True, mc_adjust=None)

mad_covar_adj = MAD(
    bandit=TSNormal(
        k=len(interventions),
        control=0,
        reward=reward_fn_covar,
        optimize="min"
    ),
    alpha=0.05,
    delta=lambda x: 1./(x**0.1),
    t_star=int(32e3),
    model=FastOLSModel,
    pooled=True,
    n_warmup=50
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
        pa_results_rct.assign(
            ate=lambda df: df["estimate"],
            lb=np.nan,
            ub=np.nan
        )[pa_results_rct.index % 2 == 0],
        pa_results_covar[pa_results_covar.index % 2 == 0],
        pa_results_mad[pa_results_mad.index % 2 == 0]
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
    + pn.geom_point(position=pn.position_dodge(width=0.6))
    + pn.geom_linerange(position=pn.position_dodge(width=0.6))
    + pn.geom_hline(yintercept=0, linetype="dashed", color="black")
    + pn.coord_flip()
    + pn.scale_color_manual(
        values={"MAD":"#F8766D", "MADCovar":"#00BA38", "RCT*":"#619CFF"}
    )
    + pn.labs(x="", y="ATE", title="Outcome: Partisan animosity", color="")
    + theme_daniel()
).save(
    base_dir / "figures" / "figure3.png",
    width=8,
    height=5,
    dpi=300
)