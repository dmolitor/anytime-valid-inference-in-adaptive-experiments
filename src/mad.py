from .bandit import Bandit, Reward
import numpy as np
import pandas as pd
import plotnine as pn
from .model import Model
from tqdm import tqdm
from typing import Any, Callable, Tuple
from .utils import (
    check_shrinkage_rate,
    cs_radius,
    last,
    prep_dummies,
    weighted_probs
)

generator = np.random.default_rng(seed=123)

class MADBase:
    """
    No need to read through the code for this class!!! It just adds some
    methods for plotting the results and spitting out a nicely formatted
    summary. The meat of the algorithm is in the `MAD` class below.
    """
    def __init__(
        self,
        bandit: Bandit,
        alpha: float,
        delta: Callable[[int], float],
        t_star: int,
        model: Model = None,
        n_warmup: int = 1,
        pooled: bool = True
    ):
        self._alpha = alpha
        self._ate = []
        self._bandit = bandit
        self._covariates = pd.DataFrame()
        self._cs_radius = []
        self._cs_width = []
        self._cs_width_benchmark = []
        self._delta = delta
        self._eliminated = {k: False for k in range(bandit.k())}
        self._is_stat_sig = {k: False for k in range(bandit.k())}
        self._ite = {"control": [], "treat": []}
        self._ite_var = {"control": [], "treat": []}
        self._model = model
        self._n = []
        self._n_warmup = n_warmup
        self._pooled = pooled
        self._probs = {k: [] for k in range(bandit.k())}
        self._rewards = []
        self._stat_sig_counter = []
        self._t_no_ite = 0
        self._t_star = t_star
        for _ in range(bandit.k()):
            self._ate.append([])
            self._cs_radius.append([])
            self._cs_width.append(0)
            self._cs_width_benchmark.append(0)
            self._ite["control"].append([])
            self._ite["treat"].append([])
            self._ite_var["control"].append([])
            self._ite_var["treat"].append([])
            self._n.append([0])
            self._stat_sig_counter.append(0)
    
    def _compute_ate(
        self,
        arm: int,
        mc_adjust: str = None
    ) -> Tuple[float, float]:
        """
        Compute the ATE and corresponding CS as laid out in Liang and Bojinov
        """
        # Get the total number of treatment arms
        n_treatments = self._bandit.k() - 1
        # Get all the ITEs and their variance estimates for the current arm
        # as well as the control arm
        control_ites = self._ite["control"][arm]
        control_vars = self._ite_var["control"][arm]
        treat_ites = self._ite["treat"][arm]
        treat_vars = self._ite_var["treat"][arm]
        ites = np.append(control_ites, treat_ites)
        vars = np.append(control_vars, treat_vars)
        assert len(ites) == len(vars), "Mismatch in dimensions of ITEs and Variances"
        # If there aren't enough ITEs to calculate the ATE just mark the ATE
        # as missing (np.nan) and mark the CS as infinite
        if len(control_ites) < 1 or len(treat_ites) < 1:
            avg_treat_effect = np.nan
            conf_seq_radius = np.inf
        else:
            # Calculate the ATE from the ITEs:
            avg_treat_effect = np.mean(ites)
            # The Confidence Sequence calculation. Check out `utils.cs_radius()`
            # for calculation details:
            conf_seq_radius = cs_radius(
                var=vars,
                t=len(ites),
                t_star=self._t_star,
                alpha=self._alpha,
                mc_adjust=mc_adjust,
                n_arms=n_treatments
            )
            alt_conf_seq_radius = cs_radius(
                var=vars,
                t=self._bandit._t,
                t_star=self._t_star,
                alpha=self._alpha,
                mc_adjust=mc_adjust,
                n_arms=n_treatments
            )
        return avg_treat_effect, conf_seq_radius
    
    def _compute_ite(self, selected_arm: Any, outcome: float, propensity: float) -> float:
        """
        Unbiased individual treatment effect estimator
        """
        control_ite = -outcome/propensity
        treat_ite = outcome/propensity
        var = (outcome**2)/(propensity**2)
        for arm in self._bandit._active_arms:
            if selected_arm == self._bandit.control():
                ite = control_ite
                ite_var = var
                self._ite["control"][arm].append(ite)
                self._ite_var["control"][arm].append(ite_var)
            elif selected_arm == arm:
                ite = treat_ite
                ite_var = var
                self._ite["treat"][arm].append(ite)
                self._ite_var["treat"][arm].append(ite_var)
            else:
                ite = 0.0
                ite_var = 0.0
                self._ite["treat"][arm].append(ite)
                self._ite_var["treat"][arm].append(ite_var)
    
    def _compute_ite_pooled(
        self,
        outcome: float,
        covariates: pd.DataFrame,
        selected_arm: Any,
        control_arm: Any,
        propensity: float,
        n_warmup: int
    ) -> float:
        """
        Unbiased individual treatment effect estimator
        """
        X = self._covariates
        y = np.array(self._rewards)
        assert len(X) == len(y), "Mismatch in dimensions of X and y"
        # If there isn't enough data to fit, return an empty array
        if (len(X) < n_warmup):
            return np.empty(0, dtype = [("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
        # Setup data for modeling
        X = prep_dummies(X, self._bandit._active_arms, add_constant=True)
        # Train model
        model = self._model(y=y, X=X)
        model.fit()
        # Counterfactual covariates under the control arm
        covar_control = prep_dummies(
            covariates,
            self._bandit._active_arms,
            add_constant=True,
            arm=control_arm
        )
        pred_control = model.predict(covar_control)[0]
        for arm in self._bandit._active_arms:
            covar_treat = prep_dummies(
                covariates,
                self._bandit._active_arms,
                add_constant=True,
                arm=arm
            )
            # Generate counterfactual prediction
            pred_treat = model.predict(covar_treat)[0]
            # ITE calculation
            pred_ite = pred_treat - pred_control
            if selected_arm == control_arm:
                ite = pred_ite - ((outcome - pred_control)/propensity)
                ite_var = ((outcome - pred_control)**2)/(propensity**2)
                self._ite["control"][arm].append(ite)
                self._ite_var["control"][arm].append(ite_var)
            elif selected_arm == arm:
                ite = pred_ite + ((outcome - pred_treat)/propensity)
                ite_var = ((outcome - pred_treat)**2)/(propensity**2)
                self._ite["treat"][arm].append(ite)
                self._ite_var["treat"][arm].append(ite_var)
            else:
                ite = pred_ite
                ite_var = 0.0
                self._ite["treat"][arm].append(ite)
                self._ite_var["treat"][arm].append(ite_var)

    def _compute_ite_split(
        self,
        outcome: float,
        covariates: pd.DataFrame,
        selected_arm: Any,
        control_arm: Any,
        propensity: float,
        n_warmup: int
    ) -> float:
        """
        Unbiased individual treatment effect estimator using separate models for each treatment arm.
        """
        X = self._covariates
        y = np.array(self._rewards)
        assert len(X) == len(y), "Mismatch in dimensions of X and y"
        # Check for sufficient data
        if len(X) < n_warmup:
            return np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
        # Train control arm model
        control_X = X[X["arm"] == control_arm].assign(const=1.0).drop("arm", axis=1)
        control_y = y[control_X.index.to_numpy()]
        if len(control_X) < 1:
            return np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
        model_control = self._model(y=control_y, X=control_X)
        model_control.fit()
        # Set up counterfactual covariates with constant
        covar_control = covariates.assign(const=1.0)
        pred_control = model_control.predict(covar_control)[0]
        # Generate ITE estimates for each arm
        estimates = []
        for arm in self._bandit._active_arms:
            # Train treatment model for the current arm
            treat_X = X[X["arm"] == arm].assign(const=1.0).drop("arm", axis=1)
            treat_y = y[treat_X.index.to_numpy()]
            if len(treat_X) < 1:
                return np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
            model_treat = self._model(y=treat_y, X=treat_X)
            model_treat.fit()
            # Generate counterfactual prediction under treatment
            pred_treat = model_treat.predict(covar_control)[0]
            pred_ite = pred_treat - pred_control
            # ITE calculation based on which arm is selected
            if selected_arm == control_arm:
                ite = pred_ite - ((outcome - pred_control) / propensity)
                ite_var = ((outcome - pred_control) ** 2) / (propensity ** 2)
                self._ite["control"][arm].append(ite)
                self._ite_var["control"][arm].append(ite_var)
            elif selected_arm == arm:
                ite = pred_ite + ((outcome - pred_treat) / propensity)
                ite_var = ((outcome - pred_treat) ** 2) / (propensity ** 2)
                self._ite["treat"][arm].append(ite)
                self._ite_var["treat"][arm].append(ite_var)
            else:
                ite = pred_ite
                ite_var = 0.0
                self._ite["treat"][arm].append(ite)
                self._ite_var["treat"][arm].append(ite_var)
            estimates.append((ite, ite_var, arm))
    
    def estimates(self) -> pd.DataFrame:
        """
        A dataframe of treatment effect estimates and confidence bands
        """
        results = {"arm": [], "ate": [], "lb": [], "ub": []}
        for arm in range(len(self._ate)):
            if arm == self._bandit.control():
                continue
            ate = last(self._ate[arm])
            radius = last(self._cs_radius[arm])
            lb = ate - radius
            ub = ate + radius
            results["arm"].append(arm)
            results["ate"].append(ate)
            results["lb"].append(lb)
            results["ub"].append(ub)
        return pd.DataFrame(results)

    def fit(
        self,
        early_stopping: bool = True,
        cs_precision: float = 0.1,
        mc_adjust: str = "Bonferroni",
        verbose: bool = True,
        **kwargs,
    ) -> None:
        """
        Fit the full MAD algorithm for the full time horizon or until there are
        no treatment arms remaining
        """
        if verbose:
            iter_range = tqdm(range(self._t_star), total = self._t_star)
        else:
            iter_range = range(self._t_star)
        for _ in iter_range:
            self.pull(
                early_stopping=early_stopping,
                cs_precision=cs_precision,
                mc_adjust=mc_adjust,
                **kwargs
            )
            # If all treatment arms have been eliminated, end the algorithm
            if early_stopping and all(
                value
                for key, value in self._eliminated.items()
                if key != self._bandit.control()
            ):
                if verbose: 
                    print("Stopping early!")
                break

    def plot_ate(self) -> pn.ggplot:
        """
        Plot the ATEs and CSs for each arm at the end of the experiment
        """
        estimates_df = self.estimates()
        plt = (
            pn.ggplot(
                estimates_df,
                pn.aes(x="factor(arm)", y="ate", ymin="lb", ymax="ub")
            )
            + pn.geom_point()
            + pn.geom_errorbar(width=0.001)
            + pn.labs(x="Arm", y="ATE")
            + pn.theme_538()
        )
        return plt
    
    def plot_ate_path(self) -> pn.ggplot:
        """
        Plot the ATE and CS paths for each arm of the experiment
        """
        arms = list(range(len(self._ate)))
        arms.remove(self._bandit.control())
        estimates = []
        for arm in arms:
            ates = self._ate[arm]
            radii = self._cs_radius[arm]
            ubs = np.nan_to_num([x + y for (x, y) in zip(ates, radii)], nan=np.inf)
            lbs = np.nan_to_num([x - y for (x, y) in zip(ates, radii)], nan=-np.inf)
            estimates_df = pd.DataFrame({
                "arm": [arm]*len(ates),
                "ate": ates,
                "lb": lbs,
                "ub": ubs,
                "t": range(1, len(ates) + 1)
            })
            estimates.append(estimates_df)
        estimates = (
            pd
            .concat(estimates, axis=0)
            .reset_index(drop=True)
            .dropna(subset="ate")
        )
        plt = (
            pn.ggplot(
                data=estimates,
                mapping=pn.aes(
                    x="t",
                    y="ate",
                    ymin="lb",
                    ymax="ub",
                    color="factor(arm)",
                    fill="factor(arm)"
                )
            )
            + pn.geom_line(size=0.3, alpha = 0.8)
            + pn.geom_ribbon(alpha=0.05)
            + pn.facet_wrap(
                "~ arm",
                ncol=2,
                labeller=pn.labeller(arm=lambda v: f"Arm {v}")
            )
            + pn.theme_538()
            + pn.theme(legend_position="none")
            + pn.labs(y="ATE", color="Arm", fill="Arm")
        )
        return plt
    
    def plot_ites(self, arm: Any, type: str = "boxplot", **kwargs) -> pn.ggplot:
        """
        Parameters
        ----------

        arm : Any
            The label of the arm for which to plot ITE estimates.
        type : str
            The type of plot. Must be one of 'boxplot', 'density', or 'histogram'.
        **kwargs
            Keyword arguments to pass directly to the `geom_{plot_type}()` call.
        """
        ites = pd.concat([
            pd.DataFrame({
                "ITE": self._ite["control"][arm]
            }).assign(Group="Control"),
            pd.DataFrame({
                "ITE": self._ite["treat"][arm]
            }).assign(Group="Treatment")
        ])
        if type.lower() == "boxplot":
            plt = (
                pn.ggplot(ites, pn.aes(y="ITE"))
                + pn.geom_boxplot(**kwargs)
                + pn.theme_538()
                + pn.facet_wrap("~ Group", scales="free", nrow=1)
            )
        elif type.lower() == "density":
            plt = (
                pn.ggplot(ites, pn.aes(x="ITE", color="Group"))
                + pn.geom_density(**kwargs)
                + pn.theme_538()
            )
        elif type.lower() == "histogram":
            plt = (
                pn.ggplot(ites, pn.aes(x="ITE"))
                + pn.geom_histogram(**kwargs)
                + pn.theme_538()
                + pn.facet_wrap("~ Group", scales="free", ncol=1)
            )
        else:
            ValueError("Type must be one of ['boxplot', 'density', 'histogram']")
        return plt
    
    def plot_n(self) -> pn.ggplot:
        """
        Plot the total sample size (N) in each arm
        """
        sample_df = pd.DataFrame({
            k: [last(self._n[k])] for k in range(self._bandit.k())
        })
        sample_df = pd.melt(sample_df, value_name="n", var_name="arm")
        plt = (
            pn.ggplot(sample_df, pn.aes(x="factor(arm)", y="n"))
            + pn.geom_bar(stat="identity")
            + pn.labs(x="Arm", y="N")
            + pn.theme_538()
        )
        return plt
    
    def plot_probabilities(self) -> pn.ggplot:
        """
        Plot the arm assignment probabilities across time
        """
        probs_df = pd.melt(
            frame=pd.DataFrame(self._probs),
            value_name="probability",
            var_name="arm"
        )
        probs_df["t"] = probs_df.groupby("arm").cumcount() + 1
        plt = (
            pn.ggplot(
                probs_df,
                pn.aes(
                    x="t",
                    y="probability",
                    color="factor(arm)",
                    group="factor(arm)"
                )
            )
            + pn.geom_line()
            + pn.theme_538()
            + pn.labs(x="t", y="Probability", color="Arm")
        )
        return plt
    
    def plot_sample(self) -> pn.ggplot:
        """
        Plot sample assignment to arms across time
        """
        sample_assignment = pd.concat([
            pd.DataFrame({
                "arm": [arm]*len(self._n[arm]),
                "t": np.array(range(len(self._n[arm]))),
                "n": self._n[arm]
            }) for arm in range(len(self._ate))
        ])
        plt = (
            pn.ggplot(
                data=sample_assignment,
                mapping=pn.aes(
                    x="t", y="n", color="factor(arm)", group="factor(arm)"
                )
            )
            + pn.geom_line()
            + pn.facet_wrap(
                "~ arm",
                ncol=2,
                labeller=pn.labeller(arm=lambda v: f"Arm {v}")
            )
            + pn.theme_538()
            + pn.theme(legend_position="none")
            + pn.labs(y="N", color="Arm", fill="Arm")
        )
        return plt
    
    def summary(self) -> None:
        """
        Print a summary of treatment effect estimates and confidence bands
        """
        print("Treatment effect estimates:")
        for arm in range(len(self._ate)):
            if arm == self._bandit.control():
                continue
            ate = last(self._ate[arm])
            radius = last(self._cs_radius[arm])
            lb = ate - radius
            ub = ate + radius
            print(f"- Arm {arm}: {round(ate, 3)} ({round(lb, 5)}, {round(ub, 5)})")


class MAD(MADBase):
    """
    A class implementing Liang and Bojinov's Mixture-Adaptive Design (MAD).
    
    Parameters
    ----------
    bandit : Bandit 
        This object must implement several crucial
        methods/attributes. For more details on how to create a custom Bandit
        object, see the documentation of the Bandit class.
    alpha : float
        The size of the statistical test (testing for non-zero treatment effects)
    delta : Callable[[int], float]
        A function that generates the real-valued sequence delta_t in Liang
        and Bojinov (Definition 4 - Mixture Adaptive Design). This sequence
        should converge to 0 slower than 1/t^(1/4) where t denotes the time
        frame in {0, ... n}. This function should intake an integer (t) and
        output a float (the corresponding delta_t)
    t_star : int
        The time-step at which we want to optimize the CSs to be tightest.
        E.g. Liang and Bojinov set this to the max horizon of their experiment
    """
    def __init__(
        self,
        bandit: Bandit,
        alpha: float,
        delta: Callable[[int], float],
        t_star: int,
        model: Model = None,
        n_warmup: int = 1,
        pooled: bool = True
    ):
        super().__init__(
            bandit=bandit,
            alpha=alpha,
            delta=delta,
            t_star=t_star,
            model=model,
            n_warmup=n_warmup,
            pooled=pooled
        )
    
    def pull(
        self,
        early_stopping: bool = True, 
        cs_precision: float = 0.1,
        mc_adjust: str = "Bonferroni"
    ) -> None:
        """
        Perform one full iteration of the MAD algorithm.

        Parameters
        ----------
        early_stopping : bool
            Whether or not to stop the experiment early when all the arms have
            statistically significant ATEs.
        cs_precision : float
            This parameter controls how precise we want to make our Confidence
            Sequences (CSs). If `cs_precision = 0` then the experiment will stop
            immediately as soon as all arms are statistically significant.
            If `cs_precision = 0.2` then the experiment will run until all
            CSs are at least 20% tighter (shorter) than they
            were when they became statistically significant. If
            `cs_precision = 0.4` the experiment will run until all CSs are at
            least 40% tighter, and so on.
        mc_adjust : str
            The type of multiple comparison correction to apply to the
            constructed CSs. Default is Bonferroni
        """
        # The index of the control arm of the bandit, typically 0
        control = self._bandit.control()
        # The number of bandit arms
        k = self._bandit.k()
        # The CURRENT time step of the bandit
        t = self._bandit.t()
        # The random exploration mixing rate at time step t; For more details
        # on this mixing sequence delta_t look at Liang and Bojinov on Page 13
        # directly above Theorem 1 and in the first new paragraph on Page 11.
        d_t = self._delta(t)
        # This function just ensures that the mixing rate d_t follows the
        # requirements stated in the paper. Can't shrink faster than 1/(t^(1/4)).
        check_shrinkage_rate(t, d_t)
        # Get the arm assignment probabilities from the underlying bandit algo
        arm_probs = self._bandit.probabilities()
        # For each arm, calculate the MAD probabilities based on the bandit probs.
        # This equation is Definition 4 in the paper and it's multi-arm corollary
        # is defined in the third paragraph in Appendix C (Page 34). The
        # multi-class definition is what I'm using here (just a generalization
        # of the 2-class case).
        probs = [d_t/k + (1 - d_t)*p for p in arm_probs.values()]
        # Record these probabilities for plotting later
        for key, value in enumerate(probs):
            self._probs[key].append(value)
        # Then select the arm as a draw from multinomial with these probabilities
        selected_index = generator.multinomial(1, pvals=probs).argmax()
        selected_arm = list(arm_probs.keys())[selected_index]
        # This is not essential for the MAD algorithm. I'm simply tracking how
        # much sample has been assigned to each arm.
        for arm in range(len(self._ate)):
            if arm == selected_arm:
                self._n[arm].append((last(self._n[arm]) + 1))
            else:
                self._n[arm].append((last(self._n[arm])))
        # Propensity score; obviously just the probability of the selected arm
        propensity = probs[selected_index]
        # Reward (and covariates) resulting from the selected arm
        reward: Reward = self._bandit.reward(selected_arm)
        # If covariates are NOT provided use standard IPW estimator
        if reward.covariates is None:
            reward = reward.outcome
            # Calculate the individual treatment effect estimate (ITE) and its
            # variance upper bound. This is effectively just (reward / propensity).
            # See `utils.ite()` for exactly what it's doing.
            self._compute_ite(
                selected_arm=selected_arm,
                outcome=reward,
                propensity=propensity
            )
            # Record the observed reward
            self._rewards.append(reward)
        # If covariates ARE provided, use AIPW estimator
        else:
            covariates = reward.covariates
            reward = reward.outcome
            # Estimate AIPW with pooled estimator
            if self._pooled:
                self._compute_ite_pooled(
                    outcome=reward,
                    covariates=covariates,
                    selected_arm=selected_arm,
                    control_arm=control,
                    propensity=propensity,
                    n_warmup=self._n_warmup
                )
            # Estimate AIPW with separate models per treatment arm
            else:
                self._compute_ite_split(
                    outcome=reward,
                    covariates=covariates,
                    selected_arm=selected_arm,
                    control_arm=control,
                    propensity=propensity,
                    n_warmup=self._n_warmup
                )
            # Record rewards and covariates
            self._rewards.append(reward)
            covariates["arm"] = selected_arm
            self._covariates = (
                pd
                .concat([self._covariates, covariates], axis=0)
                .reset_index(drop=True)
            )
        # Now, for each arm, calculate its ATE and corresponding Confidence Sequence (CS)
        # value for the current time step t.
        for arm in arm_probs.keys():
            # Calculate the ATE and corresponding CS for the current arm
            avg_treat_effect, conf_seq_radius = self._compute_ate(
                arm=arm,
                mc_adjust=mc_adjust
            )
            # Record the ATE and CS for the current arm
            self._ate[arm].append(avg_treat_effect)
            self._cs_radius[arm].append(conf_seq_radius)
            self._cs_width[arm] = 2.0*conf_seq_radius
            # This isn't really the MAD design. This just controls early stopping
            # once all the arms are eliminated (aka all arms have significant ATEs).
            # If the arm's ATE is undefined (insufficient sample size) skip
            if np.isnan(avg_treat_effect) or np.isinf(conf_seq_radius):
                continue
            # Is the arm statistically significant?
            stat_sig = np.logical_or(
                0 <= avg_treat_effect - conf_seq_radius,
                0 >= avg_treat_effect + conf_seq_radius
            )
            # Mark arm's statistical significance
            self._is_stat_sig[arm] = stat_sig
            # If the CS is statistically significant for the first time
            # we will attempt to increase precision by decreasing the
            # interval width by X% relative to its current width
            if stat_sig:
                self._stat_sig_counter[arm] += 1
                if self._stat_sig_counter[arm] == 1:
                    self._cs_width_benchmark[arm] = self._cs_width[arm]
            if early_stopping and arm != control:
                if stat_sig:
                    # Now, eliminate the arm iff it is <= (1-X)% of it's width when
                    # initially marked as statistically significant
                    threshold = (1-cs_precision)*self._cs_width_benchmark[arm]
                    if self._cs_width[arm] <= threshold:
                        self._eliminated[arm] = True
                else:
                    # If the arm has been significant but is not any more,
                    # un-eliminate the arm
                    if self._eliminated[arm]:
                        self._eliminated[arm] = False
        return None

class MADModified(MADBase):
    """
    This class implements my minor (but important) changes to the MAD design.
    All the code in the `pull()` method is identical to that in the MAD class
    above. I've only documented the chunks where the code is different, which
    is where I implement my modifications to the MAD algorithm.
    
    Parameters
    ----------
    bandit : Bandit 
        This object must implement several crucial
        methods/attributes. For more details on how to create a custom Bandit
        object, see the documentation of the Bandit class.
    alpha : float
        The size of the statistical test (testing for non-zero treatment effects)
    delta : Callable[[int], float]
        A time-decreasing function that generates the real-valued sequence
        delta_t in Liang and Bojinov (Definition 4 - Mixture Adaptive Design).
        This sequence should converge to 0 slower than 1/t^(1/4) where t denotes
        the time frame in {0, ... n}. This function should intake an integer (t)
        and output a float (the corresponding delta_t).
    t_star : int
        The time-step at which we want to optimize the CSs to be tightest.
        E.g. Liang and Bojinov set this to the max horizon of their experiment
    decay : Callable[[int], float]
        `decay()` should intake the current time step t and output a value
        in [0, 1] where 1 represents no decay and 0 represents complete decay.
        This function is similar to the `delta()` argument above. However,
        `delta()` determines the amount of random exploration in the MAD
        algorithm at time t. In contrast, `decay()` is a time-decreasing
        function that determines how quickly the bandit assignment
        probabilities for arm k decay to 0 once arm k's ATE (ATE_k) is
        statistically significant. Setting a constant `decay = lambda _: 1` makes
        this method identical to the vanilla MAD design. In contrast,
        `decay = lambda _: 0` is the same as setting the bandit probabilities
        for arm k to 0 as soon as it has a significant ATE.
    
    Attributes
    ----------
    _weights : Dict[int, float]
        These weights are calculated by `decay()`. Each arm has a weight. When
        an arm has a weight of 1, its bandit assignment probabilities are not
        adjusted. Once an arm is statistically significant, its weight begins
        to decay to 0, and so does its bandit assignment probabilities. As an
        arm's weight decays to 0, it "shifts" its probability onto currently
        under-powered arms. This iterative procedure continues to focus more
        sample on under-powered arms until either all arms have significant ATEs
        or the experiment has ended.
    """
    def __init__(
        self,
        bandit: Bandit,
        alpha: float,
        delta: Callable[[int], float],
        t_star: int,
        decay: Callable[[int], float] = lambda x: 1/np.sqrt(x),
        model: Model = None,
        n_warmup: int = 1,
        pooled: bool = True
    ):
        super().__init__(
            bandit=bandit,
            alpha=alpha,
            delta=delta,
            t_star=t_star,
            model=model,
            n_warmup=n_warmup,
            pooled=pooled
        )
        self._decay = decay
        self._eliminated_t = {k: None for k in range(bandit.k())}
        self._is_stat_sig = {k: False for k in range(bandit.k())}
        self._weights = {k: 1. for k in range(bandit.k())}

    def pull(
        self,
        early_stopping: bool = True,
        cs_precision: float = 0.1,
        mc_adjust: str = "Bonferroni"
    ) -> None:
        """
        Perform one full iteration of the modified MAD algorithm
        """
        control = self._bandit.control()
        k = self._bandit.k()
        t = self._bandit.t()
        d_t = self._delta(t)
        check_shrinkage_rate(t, d_t)
        arm_probs = self._bandit.probabilities()

        # Re-weight the bandit probabilities according to each arm's assigned
        # weights. Take a look at `utils.weighted_probs()`; it has a very
        # detailed explanation of how the re-weighting works.
        arm_probs = weighted_probs(arm_probs, self._weights)

        probs = [d_t/k + (1 - d_t)*p for p in arm_probs.values()]
        for key, value in enumerate(probs):
            self._probs[key].append(value)
        selected_index = generator.multinomial(1, pvals=probs).argmax()
        selected_arm = list(arm_probs.keys())[selected_index]
        for arm in range(len(self._ate)):
            if arm == selected_arm:
                self._n[arm].append((last(self._n[arm]) + 1))
            else:
                self._n[arm].append((last(self._n[arm])))
        propensity = probs[selected_index]
        reward: Reward = self._bandit.reward(selected_arm)
        if reward.covariates is None:
            reward = reward.outcome
            self._compute_ite(
                selected_arm=selected_arm,
                outcome=reward,
                propensity=propensity
            )
            self._rewards.append(reward)
        else:
            covariates = reward.covariates
            reward = reward.outcome
            if self._pooled:
                self._compute_ite_pooled(
                    outcome=reward,
                    covariates=covariates,
                    selected_arm=selected_arm,
                    control_arm=control,
                    propensity=propensity,
                    n_warmup=self._n_warmup
                )
            else:
                self._compute_ite_split(
                    outcome=reward,
                    covariates=covariates,
                    selected_arm=selected_arm,
                    control_arm=control,
                    propensity=propensity,
                    n_warmup=self._n_warmup
                )
            self._rewards.append(reward)
            covariates["arm"] = selected_arm
            self._covariates = (
                pd
                .concat([self._covariates, covariates], axis=0)
                .reset_index(drop=True)
            )
        for arm in arm_probs.keys():
            # Calculate the ATE and corresponding CS for the current arm
            avg_treat_effect, conf_seq_radius = self._compute_ate(
                arm=arm,
                mc_adjust=mc_adjust
            )
            self._ate[arm].append(avg_treat_effect)
            self._cs_radius[arm].append(conf_seq_radius)
            self._cs_width[arm] = 2.0*conf_seq_radius
            if np.isnan(avg_treat_effect) or np.isinf(conf_seq_radius):
                    continue
            stat_sig = np.logical_or(
                0 <= avg_treat_effect - conf_seq_radius,
                0 >= avg_treat_effect + conf_seq_radius
            )
            self._is_stat_sig[arm] = stat_sig
            if stat_sig:
                self._stat_sig_counter[arm] += 1
                if self._stat_sig_counter[arm] == 1:
                    self._cs_width_benchmark[arm] = self._cs_width[arm]
            if early_stopping and arm != control:
                if stat_sig:
                    threshold = (1-cs_precision)*self._cs_width_benchmark[arm]
                    if self._cs_width[arm] <= threshold and not self._eliminated[arm]:
                        self._eliminated[arm] = True

                        # We need to know for HOW LONG has the arm been eliminated.
                        # The longer the arm is eliminated (statistically significant)
                        # the closer its corresponding weight gets to 0
                        self._eliminated_t[arm] = t

                else:
                    if self._eliminated[arm]:
                        self._eliminated[arm] = False
                        self._eliminated_t[arm] = None
                
                # Now, update the arm's assigned weight. If the arm has been
                # eliminated this weight will decay to 0. Every time the arm
                # goes from stat. sig. to non-stat. sig., this weight resets
                # to 1 (the default value for non-stat. sig. arms) and restarts
                # its decay path to 0.
                if not self._eliminated[arm]:
                    self._weights[arm] = 1.
                else:
                    self._weights[arm] = np.max([
                        self._decay(t + 1 - self._eliminated_t[arm]),
                        1e-10
                    ]).astype(float)
        return None

class MADCovariateAdjusted(MADBase):
    """
    A class implementing a modification of Liang and Bojinov's Mixture-Adaptive
    Design (MAD). Instead of relying on an IPW estimator, we utilize an AIPW
    estimator that harnesses the power of outcome models to reduce the variance
    of our ATE estimates.
    
    Parameters
    ----------
    bandit : Bandit 
        This object must implement several crucial
        methods/attributes. For more details on how to create a custom Bandit
        object, see the documentation of the Bandit class.
    model : Model
        This model object must implement several crucial methods. For more
        details on how to create a custom Model object, see the documentation
        of the Model class.
    pooled : bool
        Flag to select the modeling strategy. When True, a single pooled model
        is fit to estimate E[Y | X=x, W=w, F] across all treatment arms. When
        False, separate models are fit for each treatment arm
        (i.e., E[Y | X=x, W=k, F] for each k in {0, ..., K}).
    n_warmup : int
        The number of "warmup" observations to gather before calculating the
        ATE and CS. Default is 1.
    alpha : float
        The size of the statistical test (testing for non-zero treatment effects)
    delta : Callable[[int], float]
        A function that generates the real-valued sequence delta_t in Liang
        and Bojinov (Definition 4 - Mixture Adaptive Design). This sequence
        should converge to 0 slower than 1/t^(1/4) where t denotes the time
        frame in {0, ... n}. This function should intake an integer (t) and
        output a float (the corresponding delta_t)
    t_star : int
        The time-step at which we want to optimize the CSs to be tightest.
        E.g. Liang and Bojinov set this to the max horizon of their experiment
    """
    def __init__(
        self,
        bandit: Bandit,
        model: Model,
        pooled: bool = True,
        n_warmup: int = 1,
        alpha: float = 0.05,
        delta: Callable[[int], float] = lambda x: 1./(x**0.24),
        t_star: int = int(1e3)
    ):
        super().__init__(
            bandit=bandit,
            alpha=alpha,
            delta=delta,
            t_star=t_star
        )
        self._covariates = pd.DataFrame()
        self._ite = {
            "control": np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")]),
            "treat": np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
        }
        self._model = model
        self._n_warmup = n_warmup
        self._pooled = pooled
    
    def _ite_pooled(
        self,
        outcome: float,
        covariates: pd.DataFrame,
        selected_arm: Any,
        control_arm: Any,
        propensity: float,
        n_warmup: int
    ) -> float:
        """
        Unbiased individual treatment effect estimator:
        
        TODO: replace the modeling stuff below to accept a user-provided fitting function
        """
        X = self._covariates
        y = self._rewards
        assert len(X) == len(y), "Mismatch in dimensions of X and y"
        # If there isn't enough data to fit, return an empty array
        if (len(X) < n_warmup):
            return np.empty(0, dtype = [("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
        # Setup data for modeling
        X = prep_dummies(X, self._bandit._active_arms, add_constant=True)
        # Train model
        model = self._model(y=y, X=X)
        model.fit()
        # Counterfactual covariates under the control arm
        covar_control = prep_dummies(
            covariates,
            self._bandit._active_arms,
            add_constant=True,
            arm=control_arm
        )
        pred_control = model.predict(covar_control)[0]
        estimates = []
        for arm in self._bandit._active_arms:
            covar_treat = prep_dummies(
                covariates,
                self._bandit._active_arms,
                add_constant=True,
                arm=arm
            )
            # Generate counterfactual prediction
            pred_treat = model.predict(covar_treat)[0]
            # ITE calculation
            pred_ite = pred_treat - pred_control
            if selected_arm == control_arm:
                ite = pred_ite - ((outcome - pred_control)/propensity)
                ite_var = ((outcome - pred_control)**2)/(propensity**2)
            elif selected_arm == arm:
                ite = pred_ite + ((outcome - pred_treat)/propensity)
                ite_var = ((outcome - pred_treat)**2)/(propensity**2)
            else:
                ite = pred_ite
                ite_var = 0.0
            # Append to array
            estimates.append((ite, ite_var, arm))
        estimates = np.array(
            estimates,
            dtype = [("ite", "f8"), ("ite_var", "f8"), ("arm", "O")]
        )
        return estimates

    def _ite_split(
        self,
        outcome: float,
        covariates: pd.DataFrame,
        selected_arm: Any,
        control_arm: Any,
        propensity: float,
        n_warmup: int
    ) -> float:
        """
        Unbiased individual treatment effect estimator using separate models for each treatment arm.
        """
        X = self._covariates
        y = self._rewards
        assert len(X) == len(y), "Mismatch in dimensions of X and y"
        # Check for sufficient data
        if len(X) < n_warmup:
            return np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
        # Train control arm model
        control_X = X[X["arm"] == control_arm].assign(const=1.0).drop("arm", axis=1)
        control_y = y[control_X.index.to_numpy()]
        if len(control_X) < 1:
            return np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
        model_control = self._model(y=control_y, X=control_X)
        model_control.fit()
        # Set up counterfactual covariates with constant
        covar_control = covariates.assign(const=1.0)
        pred_control = model_control.predict(covar_control)[0]
        # Generate ITE estimates for each arm
        estimates = []
        for arm in self._bandit._active_arms:
            # Train treatment model for the current arm
            treat_X = X[X["arm"] == arm].assign(const=1.0).drop("arm", axis=1)
            treat_y = y[treat_X.index.to_numpy()]
            if len(treat_X) < 1:
                return np.empty(0, dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")])
            model_treat = self._model(y=treat_y, X=treat_X)
            model_treat.fit()
            # Generate counterfactual prediction under treatment
            pred_treat = model_treat.predict(covar_control)[0]
            pred_ite = pred_treat - pred_control
            # ITE calculation based on which arm is selected
            if selected_arm == control_arm:
                ite = pred_ite - ((outcome - pred_control) / propensity)
                ite_var = ((outcome - pred_control) ** 2) / (propensity ** 2)
            elif selected_arm == arm:
                ite = pred_ite + ((outcome - pred_treat) / propensity)
                ite_var = ((outcome - pred_treat) ** 2) / (propensity ** 2)
            else:
                ite = pred_ite
                ite_var = 0.0
            estimates.append((ite, ite_var, arm))
        estimates = np.array(
            estimates,
            dtype=[("ite", "f8"), ("ite_var", "f8"), ("arm", "O")]
        )
        return estimates

    def pull(
        self,
        early_stopping: bool = True, 
        cs_precision: float = 0.1,
        mc_adjust: str = "Bonferroni"
    ) -> None:
        """
        Perform one full iteration of the MAD algorithm.

        Parameters
        ----------
        early_stopping : bool
            Whether or not to stop the experiment early when all the arms have
            statistically significant ATEs.
        cs_precision : float
            This parameter controls how precise we want to make our Confidence
            Sequences (CSs). If `cs_precision = 0` then the experiment will stop
            immediately as soon as all arms are statistically significant.
            If `cs_precision = 0.2` then the experiment will run until all
            CSs are at least 20% tighter (shorter) than they
            were when they became statistically significant. If
            `cs_precision = 0.4` the experiment will run until all CSs are at
            least 40% tighter, and so on.
        mc_adjust : str
            The type of multiple comparison correction to apply to the
            constructed CSs. Default is Bonferroni
        """
        control = self._bandit.control()
        k = self._bandit.k()
        t = self._bandit.t()
        d_t = self._delta(t)
        check_shrinkage_rate(t, d_t)
        arm_probs = self._bandit.probabilities()
        probs = [d_t/k + (1 - d_t)*p for p in arm_probs.values()]
        for key, value in enumerate(probs):
            self._probs[key].append(value)
        selected_index = generator.multinomial(1, pvals=probs).argmax()
        selected_arm = list(arm_probs.keys())[selected_index]
        for arm in range(len(self._ate)):
            if arm == selected_arm:
                self._n[arm].append((last(self._n[arm]) + 1))
            else:
                self._n[arm].append((last(self._n[arm])))
        propensity = probs[selected_index]
        # Observe reward (outcome) and corresponding covariates
        reward, covariates = self._bandit.reward(selected_arm)
        if self._pooled:
            ite_est = self._ite_pooled(
                outcome=reward,
                covariates=covariates,
                selected_arm=selected_arm,
                control_arm=control,
                propensity=propensity,
                n_warmup=self._n_warmup
            )
        else:
            ite_est = self._ite_split(
                outcome=reward,
                covariates=covariates,
                selected_arm=selected_arm,
                control_arm=control,
                propensity=propensity,
                n_warmup=self._n_warmup
            )
        self._rewards = np.append(self._rewards, reward)
        covariates["arm"] = selected_arm
        self._covariates = (
            pd
            .concat([self._covariates, covariates], axis=0)
            .reset_index(drop=True)
        )
        if len(ite_est) > 0:
            if selected_arm == control:
                self._ite["control"] = np.append(self._ite["control"], ite_est)
            else:
                self._ite["treat"] = np.append(self._ite["treat"], ite_est)
        else:
            self._t_no_ite += 1
        for arm in arm_probs.keys():
            avg_treat_effect, conf_seq_radius = self._compute_ate(
                arm=arm,
                mc_adjust=mc_adjust
            )
            # Record the ATE and CS for the current arm
            self._ate[arm].append(avg_treat_effect)
            self._cs_radius[arm].append(conf_seq_radius)
            self._cs_width[arm] = 2.0*conf_seq_radius
            # This isn't really the MAD design. This just controls early stopping
            # once all the arms are eliminated (aka all arms have significant ATEs).
            # If the arm's ATE is undefined (insufficient sample size) skip
            if np.isnan(avg_treat_effect) or np.isinf(conf_seq_radius):
                continue
            # Is the arm statistically significant?
            stat_sig = np.logical_or(
                0 <= avg_treat_effect - conf_seq_radius,
                0 >= avg_treat_effect + conf_seq_radius
            )
            # Mark arm's statistical significance
            self._is_stat_sig[arm] = stat_sig
            # If the CS is statistically significant for the first time
            # we will attempt to increase precision by decreasing the
            # interval width by X% relative to its current width
            if stat_sig:
                self._stat_sig_counter[arm] += 1
                if self._stat_sig_counter[arm] == 1:
                    self._cs_width_benchmark[arm] = self._cs_width[arm]
            if early_stopping and arm != control:
                if stat_sig:
                    # Now, eliminate the arm iff it is <= (1-X)% of it's width when
                    # initially marked as statistically significant
                    threshold = (1-cs_precision)*self._cs_width_benchmark[arm]
                    if self._cs_width[arm] <= threshold:
                        self._eliminated[arm] = True
                else:
                    # If the arm has been significant but is not any more,
                    # un-eliminate the arm
                    if self._eliminated[arm]:
                        self._eliminated[arm] = False
        return None