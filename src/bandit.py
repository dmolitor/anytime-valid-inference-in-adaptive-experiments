from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from scipy.stats import invgamma
from typing import Callable, Dict

class Bandit(ABC):
    """
    An abstract class for Bandit algorithms. Each bandit algorithm must define
    the abstract methods defined below.
    """
    @abstractmethod
    def control(self) -> int:
        """
        Returns the index of the arm that is the control arm. E.g. if the
        bandit is a 3-arm bandit with the first arm being the control arm,
        this should return the value 0.
        """

    @abstractmethod
    def eliminate_arm(self, arm: int) -> None:
        """
        Eliminate an arm of the bandit. In other words, the specified arm
        should no longer ever be assigned as a treatment and all other
        methods should change as necessary.

        Parameters:
        -----------
        arm: int - The index of the eliminated bandit arm
        """
    
    @abstractmethod
    def k(self) -> int:
        """This method that returns the number of arms in the bandit"""
    
    @abstractmethod
    def probabilities(self) -> Dict[int, float]:
        """
        Returns a dictionary with the arm indices as keys and 
        selection probabilities for each arm as values. For example,
        if the bandit algorithm is UCB with three arms, and the third arm has
        the maximum confidence bound, then this should return the following
        dictionary: {0: 0., 1: 0., 1: 1.}, since UCB is deterministic.
        """
    
    @abstractmethod
    def reactivate_arm(self, arm: int):
        """
        This method is a pseudo-inverse of the `eliminate_arm` method. In other
        words this specifies an arm that, at one point, was deactivated, but
        now should be added back to the pool of active treatment arms. It
        should alter all other methods as necessary.
        """

    @abstractmethod
    def reward(self, arm: int) -> "Reward":
        """
        Returns the reward for a selected arm. Returned object must
        be of class `Reward`.
        
        Parameters:
        -----------
        arm: int - The index of the selected bandit arm
        """
    
    @abstractmethod
    def t(self) -> int:
        """
        This method returns the current time step of the bandit, and then
        increments the time step by 1. E.g. if the bandit has completed
        9 iterations, this should return the value 10. Time step starts
        at 1, not 0.
        """

class AB(Bandit):
    """
    A class for implementing and A/B-style experiment
    """
    def __init__(self, k: int, control: int, reward: Callable[[int], float]):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._t = 1
        self._reward_fn = reward
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in `len(self._active_arms)` and `self.k()`"
        return {x: 1/self.k() for x in self._active_arms}
    
    def reactivate_arm(self, arm: int) -> None:
        self._active_arms.append(arm)
        self._active_arms.sort()
        self._k += 1
    
    def reward(self, arm: int) -> float:
        reward: Reward = self._reward_fn(arm)
        # Assert that the reward is an object of class `Reward`
        if not isinstance(reward, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        return reward

    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class ControlAugmentedTSBernoulli(Bandit):
    """
    Control-augmented Thompson Sampling for Bernoulli data using a single draw per arm.
    
    For non-control arms, we use a Beta(1,1) prior that is updated with each observed reward.
    The algorithm works as follows:
    
    1. For each non-control arm, sample one value from its Beta posterior.
    2. Let b be the non-control arm with the best sample (maximum if optimize="max", minimum if "min").
    3. Compute d = (cumulative assignments for b) - (cumulative assignments for control).
    4. Compute q = min(max(d / batch_size, 0), Z) — this is the catch-up fraction.
    5. Set the control arm’s probability to p_control = q + R * (1 - q).
    6. Assign the best non-control arm the remaining probability: (1 - R) * (1 - q), and all other arms get 0.
    7. Normalize the probabilities so they sum to 1.
    
    Parameters:
      k : int
          Total number of arms.
      control : int
          The index of the control arm.
      reward : Callable[[int], Reward]
          A function that takes an arm index and returns a Reward object.
      optimize : str, default "max"
          Whether to maximize ("max") or minimize ("min") the outcomes.
      Z : float in (0,1), default 0.5
          Maximum proportion of the next batch allocated to control as catch-up.
      R : float in (0,1), default 0.0
          Fraction of the remaining probability allocated to control.
      batch_size : int, default 1
          The size of the next batch (n) for which catch-up is calculated.
    """
    
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], "Reward"],
        optimize: str = "max",
        Z: float = 0.5,
        R: float = 0.0
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._optimize = optimize
        self._reward_fn = reward
        self._t = 1
        self._Z = Z
        self._R = R
        
        # For non-control arms, initialize Beta(1,1) priors.
        # For the control arm, we do not update its parameters.
        self._params = {}
        for arm in range(k):
            if arm == control:
                self._params[arm] = {"alpha": None, "beta": None}
            else:
                self._params[arm] = {"alpha": 1, "beta": 1}
        
        # Track cumulative assignments for each arm.
        self._rewards = {arm: [] for arm in range(k)}
    
    def probabilities(self) -> Dict[int, float]:
        # Sample one draw from each non-control arm's Beta posterior.
        non_control = [arm for arm in self._active_arms if arm != self._control]
        ts_samples = {}
        for arm in non_control:
            a = self._params[arm]["alpha"]
            b = self._params[arm]["beta"]
            ts_samples[arm] = np.random.beta(a, b)
        
        # Determine the best non-control arm based on the single draw.
        if self._optimize == "max":
            best_arm = max(ts_samples, key=ts_samples.get)
        elif self._optimize == "min":
            best_arm = min(ts_samples, key=ts_samples.get)
        else:
            raise ValueError("`optimize` must be either 'max' or 'min'")
        
        # Compute cumulative assignment counts.
        n_best = len(self._rewards[best_arm])
        n_control = len(self._rewards[self._control])
        d = n_best - n_control
        
        # Compute catch-up fraction q.
        q = min(max(d, 0), self._Z)
        
        # Compute augmented probability for control:
        p_control = q + self._R * (1 - q)
        # The best non-control arm gets the remaining probability.
        p_best = (1 - self._R) * (1 - q)
        
        # Build the probability dictionary.
        probs = {self._control: p_control}
        for arm in non_control:
            probs[arm] = p_best if arm == best_arm else 0.0
        
        # Normalize to ensure the probabilities sum to 1.
        total = sum(probs.values())
        if total > 0:
            probs = {arm: p / total for arm, p in probs.items()}
        return probs
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        if arm in self._active_arms:
            self._active_arms.remove(arm)
            self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def reactivate_arm(self, arm: int):
        if arm not in self._active_arms:
            self._active_arms.append(arm)
            self._active_arms.sort()
            self._k += 1
    
    def reward(self, arm: int) -> "Reward":
        reward_obj = self._reward_fn(arm)
        if not isinstance(reward_obj, Reward):
            raise ValueError("The provided reward function must return a Reward object")
        outcome = reward_obj.outcome
        self._rewards[arm].append(outcome)
        # Update Beta parameters for non-control arms.
        if arm != self._control:
            if outcome == 1:
                self._params[arm]["alpha"] += 1
            else:
                self._params[arm]["beta"] += 1
        return reward_obj
    
    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class ControlAugmentedTSNormal(Bandit):
    """
    Control-augmented Thompson Sampling for normally distributed data with unknown mean and variance.
    Uses a Normal–Inverse-Gamma prior for each non-control arm:
      mu | sigma^2 ~ N(prior_mu, sigma^2/prior_kappa)
      sigma^2 ~ InvGamma(prior_alpha, prior_beta)
      
    The control arm is excluded from posterior updating; only its cumulative count is tracked.
    
    Parameters:
      k : int
          Total number of arms.
      control : int
          Index of the control arm.
      reward : Callable[[int], Reward]
          Function taking an arm index and returning a Reward object.
      optimize : str, default "max"
          Whether to select the arm with the maximum (or minimum) posterior sample.
      Z : float in (0,1), default 0.5
          Maximum fraction of the next batch allocated to control as catch-up.
      R : float in (0,1), default 0.0
          Fraction of the remaining probability allocated to control.
      batch_size : int, default 1
          The batch size for which the catch-up fraction is calculated.
      prior_mu : float, default 0.0
          Prior mean for non-control arms.
      prior_kappa : float, default 1.0
          Prior “sample size” for the mean.
      prior_alpha : float, default 1.0
          Prior alpha for the inverse gamma (controls variance).
      prior_beta : float, default 1.0
          Prior beta for the inverse gamma (controls variance).
    """
    
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], "Reward"],
        optimize: str = "max",
        Z: float = 0.5,
        R: float = 0.0
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._optimize = optimize  # "max" or "min"
        self._reward_fn = reward
        self._t = 1
        self._Z = Z
        self._R = R
        
        # Initialize posterior parameters for non-control arms; control arm's parameters remain unused.
        self._params = {}
        for arm in range(k):
            if arm == control:
                self._params[arm] = None  # Control arm is not updated.
            else:
                self._params[arm] = {
                    "mu": 0.0,
                    "kappa": 1.0,
                    "alpha": 1.0,
                    "beta": 1.0
                }
        self._rewards = {x: [] for x in range(k)}
    
    def probabilities(self) -> Dict[int, float]:
        """
        Computes the augmented assignment probabilities as follows:
          1. For each non-control arm, draw a single sample from its Normal–Inverse-Gamma posterior.
          2. Identify the best non-control arm based on the sample (max or min depending on optimize).
          3. Let d = (cumulative assignments for best arm) - (cumulative assignments for control).
          4. Compute q = min(max(d / batch_size, 0), Z).
          5. Set control arm probability to p_control = q + R * (1 - q).
          6. Assign the best non-control arm p_best = (1 - R) * (1 - q); all other non-control arms get 0.
          7. Normalize so probabilities sum to 1.
        """
        # Sample one draw for each non-control arm.
        non_control = [arm for arm in self._active_arms if arm != self._control]
        ts_samples = {}
        for arm in non_control:
            params = self._params[arm]
            # Sample sigma^2 from the inverse gamma distribution.
            sigma2_sample = invgamma.rvs(a=params["alpha"], scale=params["beta"])
            # Sample mu from Normal with variance sigma2_sample/kappa.
            mu_sample = np.random.normal(loc=params["mu"], scale=np.sqrt(sigma2_sample / params["kappa"]))
            ts_samples[arm] = mu_sample
        
        # If there are no non-control arms, assign probability 1 to control.
        if not ts_samples:
            return {self._control: 1.0}
        
        # Identify the best non-control arm.
        if self._optimize == "max":
            best_arm = max(ts_samples, key=ts_samples.get)
        elif self._optimize == "min":
            best_arm = min(ts_samples, key=ts_samples.get)
        else:
            raise ValueError("`optimize` must be either 'max' or 'min'")
        
        # Compute cumulative assignment counts.
        n_best = len(self._rewards[best_arm])
        n_control = len(self._rewards[self._control])
        d = n_best - n_control
        
        # Compute catch-up fraction q.
        q = min(max(d, 0), self._Z)
        
        # Compute augmented probability for control.
        p_control = q + self._R * (1 - q)
        # Allocate the remaining probability to the best non-control arm.
        p_best = (1 - self._R) * (1 - q)
        
        # Build the probability dictionary.
        probs = {self._control: p_control}
        for arm in non_control:
            probs[arm] = p_best if arm == best_arm else 0.0
        
        # Normalize to sum to 1.
        total = sum(probs.values())
        if total > 0:
            probs = {arm: p / total for arm, p in probs.items()}
        return probs
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        if arm in self._active_arms:
            self._active_arms.remove(arm)
            self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def reactivate_arm(self, arm: int) -> None:
        if arm not in self._active_arms:
            self._active_arms.append(arm)
            self._active_arms.sort()
            self._k += 1
    
    def reward(self, arm: int) -> "Reward":
        """
        Processes a new observation for the specified arm.
        For non-control arms, updates the Normal–Inverse-Gamma posterior.
        For the control arm, only records the assignment.
        """
        reward_obj = self._reward_fn(arm)
        if not isinstance(reward_obj, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        r = reward_obj.outcome
        self._rewards[arm].append(r)
        
        # Update posterior for non-control arms.
        if arm != self._control:
            params = self._params[arm]
            mu_old = params["mu"]
            kappa_old = params["kappa"]
            alpha_old = params["alpha"]
            beta_old = params["beta"]
            
            kappa_new = kappa_old + 1
            mu_new = (kappa_old * mu_old + r) / kappa_new
            alpha_new = alpha_old + 0.5
            beta_new = beta_old + 0.5 * (r - mu_old)**2 * (kappa_old / kappa_new)
            
            self._params[arm] = {
                "mu": mu_new,
                "kappa": kappa_new,
                "alpha": alpha_new,
                "beta": beta_new
            }
        return reward_obj
    
    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class TopNTSBernoulli(Bandit):
    """
    Top-N Thompson Sampling for Bernoulli data.
    
    This algorithm uses a Beta(1,1) prior for each arm.
    It generalizes Top-2 TS to Top-N TS: it samples N distinct candidate arms 
    (via independent TS draws) and then selects one of these candidates according 
    to a probability vector beta.
    
    Parameters:
      k : int
          Total number of arms.
      control : int
          Index of the control arm (required by the Bandit interface, though not used by the algorithm).
      reward : Callable[[int], Reward]
          A function that takes an arm index and returns a Reward object.
      optimize : str, default "max"
          Whether to pick the candidate with the maximum sample ("max") or minimum ("min").
      N : int, default 2
          The number of top candidates to consider (must be at least 2).
      beta : list or np.array of floats, length N
          A probability vector that sums to 1, used to select among the N candidates.
    """
    
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], "Reward"],
        optimize: str = "max",
        N: int = 2,
        beta: np.ndarray = None
    ):
        if N < 2:
            raise ValueError("N must be at least 2")
            
        self._active_arms = [x for x in range(k)]
        self._control = control  # Not used in selection but required by interface.
        self._k = k
        self._optimize = optimize
        self._reward_fn = reward
        self._t = 1
        
        # Set up Top-N parameters.
        self._N = N
        # If no beta vector is provided, default to equal probabilities.
        if beta is None:
            self._beta = np.ones(N) / N
        else:
            beta = np.array(beta)
            if not np.isclose(beta.sum(), 1):
                raise ValueError("The beta vector must sum to 1")
            if len(beta) != N:
                raise ValueError("Length of beta vector must equal N")
            self._beta = beta
        
        # Initialize each arm's Beta prior for Bernoulli outcomes.
        self._params = {arm: {"alpha": 1, "beta": 1} for arm in range(k)}
        # Record observed outcomes per arm.
        self._rewards = {arm: [] for arm in range(k)}
    
    def probabilities(self) -> Dict[int, float]:
        """
        Implements Top-N Thompson Sampling:
          1. Sample candidate arms: repeatedly draw one sample from each arm's Beta posterior,
             and record the best arm (by max or min) if it is not already chosen.
          2. Once N distinct candidates are collected, sample an index from the provided 
             beta vector and select the corresponding candidate.
          3. Return a dictionary mapping each arm to its assignment probability 
             (1.0 for the selected arm, 0.0 for all others).
        """
        candidates = []
        # Generate N distinct candidates.
        while len(candidates) < self._N:
            # Draw one sample for each active arm.
            samples = {
                arm: np.random.beta(self._params[arm]["alpha"], self._params[arm]["beta"])
                for arm in self._active_arms
            }
            if self._optimize == "max":
                candidate = max(samples, key=samples.get)
            elif self._optimize == "min":
                candidate = min(samples, key=samples.get)
            else:
                raise ValueError("`optimize` must be either 'max' or 'min'")
            if candidate not in candidates:
                candidates.append(candidate)
            # If the same candidate is drawn repeatedly, the loop continues until a new one appears.
        
        # Sample an index from the beta vector.
        chosen_index = np.random.choice(self._N, p=self._beta)
        selected_arm = candidates[chosen_index]
        
        # Build the probability dictionary: assign probability 1 to the selected arm.
        probs = {arm: 0.0 for arm in self._active_arms}
        probs[selected_arm] = 1.0
        return probs
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        if arm in self._active_arms:
            self._active_arms.remove(arm)
            self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def reactivate_arm(self, arm: int) -> None:
        if arm not in self._active_arms:
            self._active_arms.append(arm)
            self._active_arms.sort()
            self._k += 1
    
    def reward(self, arm: int) -> "Reward":
        """
        Obtains a reward for the specified arm and updates its Beta prior.
        """
        reward_obj = self._reward_fn(arm)
        if not isinstance(reward_obj, Reward):
            raise ValueError("The provided reward function must return a Reward object")
        outcome = reward_obj.outcome
        self._rewards[arm].append(outcome)
        # Update the Beta parameters: for a success, increment alpha; for a failure, increment beta.
        if outcome == 1:
            self._params[arm]["alpha"] += 1
        else:
            self._params[arm]["beta"] += 1
        return reward_obj
    
    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class TopNTSNormal(Bandit):
    """
    Top-N Thompson Sampling for normally distributed data with unknown mean and variance.
    
    Each arm uses a Normal–Inverse-Gamma prior:
      mu | sigma^2 ~ N(prior_mu, sigma^2/prior_kappa)
      sigma^2 ~ InvGamma(prior_alpha, prior_beta)
    
    On each decision step, the algorithm:
      1. Draws one sample from each arm's posterior.
      2. Collects N distinct candidate arms based on the best (max or min) sample.
      3. Chooses one candidate according to a provided probability vector β.
    
    Parameters:
      k : int
          Total number of arms.
      control : int
          Index of the control arm (required by the interface but not used in selection).
      reward : Callable[[int], Reward]
          Function taking an arm index and returning a Reward object.
      optimize : str, default "max"
          Whether to select the candidate with the maximum ("max") or minimum ("min") sample.
      N : int, default 2
          The number of top candidates to consider (must be at least 2).
      beta : list or np.array of floats, length N
          A probability vector (summing to 1) used to select among the N candidates.
    """
    
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], "Reward"],
        optimize: str = "max",
        N: int = 2,
        beta: np.ndarray = None,
    ):
        if N < 2:
            raise ValueError("N must be at least 2")
            
        self._active_arms = [x for x in range(k)]
        self._control = control  # Required by the interface.
        self._k = k
        self._optimize = optimize
        self._reward_fn = reward
        self._t = 1
        
        # Top-N parameters.
        self._N = N
        if beta is None:
            self._beta = np.ones(N) / N
        else:
            beta = np.array(beta)
            if not np.isclose(beta.sum(), 1):
                raise ValueError("The beta vector must sum to 1")
            if len(beta) != N:
                raise ValueError("Length of beta vector must equal N")
            self._beta = beta
        
        # Initialize each arm's Normal–Inverse-Gamma prior.
        self._params = {
            arm: {
                "mu": 0.0,
                "kappa": 1.0,
                "alpha": 1.0,
                "beta": 1.0
            }
            for arm in range(k)
        }
        # Record observed outcomes per arm.
        self._rewards = {arm: [] for arm in range(k)}
    
    def probabilities(self) -> Dict[int, float]:
        """
        Implements Top-N Thompson Sampling for Normal data:
          1. Repeatedly sample one draw from each arm's posterior.
          2. Collect N distinct candidate arms based on the best sample 
             (according to 'optimize': "max" or "min").
          3. Draw an index from the β vector and select the corresponding candidate.
          4. Return a dictionary that assigns probability 1 to the chosen arm.
        """
        candidates = []
        while len(candidates) < self._N:
            samples = {}
            for arm in self._active_arms:
                params = self._params[arm]
                # Sample variance from the inverse gamma distribution.
                sigma2_sample = invgamma.rvs(a=params["alpha"], scale=params["beta"])
                # Sample mean from the normal distribution with variance sigma2/kappa.
                mu_sample = np.random.normal(loc=params["mu"], scale=np.sqrt(sigma2_sample / params["kappa"]))
                samples[arm] = mu_sample
            
            if self._optimize == "max":
                candidate = max(samples, key=samples.get)
            elif self._optimize == "min":
                candidate = min(samples, key=samples.get)
            else:
                raise ValueError("`optimize` must be either 'max' or 'min'")
            if candidate not in candidates:
                candidates.append(candidate)
            # If the same candidate is sampled repeatedly, the loop continues until N distinct ones are found.
        
        chosen_index = np.random.choice(self._N, p=self._beta)
        selected_arm = candidates[chosen_index]
        
        probs = {arm: 0.0 for arm in self._active_arms}
        probs[selected_arm] = 1.0
        return probs
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        if arm in self._active_arms:
            self._active_arms.remove(arm)
            self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def reactivate_arm(self, arm: int) -> None:
        if arm not in self._active_arms:
            self._active_arms.append(arm)
            self._active_arms.sort()
            self._k += 1
    
    def reward(self, arm: int) -> "Reward":
        """
        Obtains a reward for the specified arm and updates its Normal–Inverse-Gamma posterior.
        """
        reward_obj = self._reward_fn(arm)
        if not isinstance(reward_obj, Reward):
            raise ValueError("The provided reward function must return a Reward object")
        r = reward_obj.outcome
        self._rewards[arm].append(r)
        
        params = self._params[arm]
        mu_old = params["mu"]
        kappa_old = params["kappa"]
        alpha_old = params["alpha"]
        beta_old = params["beta"]
        
        kappa_new = kappa_old + 1
        mu_new = (kappa_old * mu_old + r) / kappa_new
        alpha_new = alpha_old + 0.5
        beta_new = beta_old + 0.5 * (r - mu_old)**2 * (kappa_old / kappa_new)
        
        self._params[arm] = {
            "mu": mu_new,
            "kappa": kappa_new,
            "alpha": alpha_new,
            "beta": beta_new
        }
        return reward_obj
    
    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class TSNormal(Bandit):
    """
    Thompson Sampling for normally distributed data with unknown mean and variance.
    Uses a Normal–Inverse-Gamma prior for each arm:
      mu | sigma^2 ~ N(prior_mu, sigma^2/prior_kappa)
      sigma^2 ~ InvGamma(prior_alpha, prior_beta)
    """
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], "Reward"],
        optimize: str = "max"
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._optimize = optimize  # "max" (or "min" for minimization problems)
        self._reward_fn = reward
        self._t = 1
        # Each arm's posterior parameters are stored in a dict with keys:
        # "mu", "kappa", "alpha", "beta"
        self._params = {
            x: {"mu": 0.0, "kappa": 1.0, "alpha": 1.0, "beta": 1.0}
            for x in range(k)
        }
        self._rewards = {x: [] for x in range(k)}
    
    def calculate_probs(self) -> Dict[int, float]:
        sample_size = 1
        samples = []
        for idx in self._active_arms:
            params = self._params[idx]
            # Sample sigma^2 from the inverse gamma distribution
            sigma2_sample = invgamma.rvs(a=params["alpha"], scale=params["beta"])
            # Sample mu from the normal with variance sigma2_sample/kappa
            mu_sample = np.random.normal(loc=params["mu"], scale=np.sqrt(sigma2_sample / params["kappa"]))
            samples.append(mu_sample)
        samples = np.array(samples).reshape(1, -1)  # shape: (1, number of active arms)
        if self._optimize == "max":
            optimal_indices = np.argmax(samples, axis=1)
        elif self._optimize == "min":
            optimal_indices = np.argmin(samples, axis=1)
        else:
            raise ValueError("`optimize` must be one of: ['max', 'min']")
        win_counts = {
            idx: np.sum(optimal_indices == i) / sample_size
            for i, idx in enumerate(self._active_arms)
        }
        return win_counts

    def control(self) -> int:
        return self._control

    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1

    def k(self) -> int:
        return self._k

    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in active arms and k"
        return self.calculate_probs()

    def reactivate_arm(self, arm: int) -> None:
        if arm not in self._active_arms:
            self._active_arms.append(arm)
            self._active_arms.sort()
            self._k += 1

    def reward(self, arm: int) -> "Reward":
        reward_obj = self._reward_fn(arm)
        if not isinstance(reward_obj, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        r = reward_obj.outcome
        self._rewards[arm].append(r)
        # Retrieve current posterior parameters for the arm
        params = self._params[arm]
        mu_old = params["mu"]
        kappa_old = params["kappa"]
        alpha_old = params["alpha"]
        beta_old = params["beta"]
        # Update the parameters using the Normal–Inverse-Gamma conjugate update formulas
        kappa_new = kappa_old + 1
        mu_new = (kappa_old * mu_old + r) / kappa_new
        alpha_new = alpha_old + 0.5
        beta_new = beta_old + 0.5 * (r - mu_old)**2 * (kappa_old / kappa_new)
        self._params[arm] = {
            "mu": mu_new,
            "kappa": kappa_new,
            "alpha": alpha_new,
            "beta": beta_new
        }
        return reward_obj

    def t(self) -> int:
        step = self._t
        self._t += 1
        return step


class TSBernoulli(Bandit):
    """
    A class for implementing Thompson Sampling on Bernoulli data
    """
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], float],
        optimize: str = "max"
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._means = {x: 0. for x in range(k)}
        self._optimize = optimize
        self._params = {x: {"alpha": 1, "beta": 1} for x in range(k)}
        self._rewards = {x: [] for x in range(k)}
        self._reward_fn = reward
        self._t = 1
    
    def calculate_probs(self) -> Dict[int, float]:
        sample_size = 1
        samples = np.column_stack([
            np.random.beta(
                a=self._params[idx]["alpha"],
                b=self._params[idx]["beta"],
                size=sample_size
            )
            for idx in self._active_arms
        ])
        if self._optimize == "max":
            optimal_indices = np.argmax(samples, axis=1)
        elif self._optimize == "min":
            optimal_indices = np.argmin(samples, axis=1)
        else:
            raise ValueError("`self._optimal` must be one of: ['max', 'min']")
        win_counts = {
            idx: np.sum(optimal_indices == i) / sample_size
            for i, idx in enumerate(self._active_arms)
        }
        return win_counts

    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in `len(self._active_arms)` and `self.k()`"
        probs = self.calculate_probs()
        return probs
    
    def reactivate_arm(self, arm: int) -> None:
        self._active_arms.append(arm)
        self._active_arms.sort()
        self._k += 1
    
    def reward(self, arm: int) -> float:
        reward: Reward = self._reward_fn(arm)
        # Assert that the reward is an object of class `Reward`
        if not isinstance(reward, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        self._rewards[arm].append(reward.outcome)
        if reward.outcome == 1:
            self._params[arm]["alpha"] += 1
        else:
            self._params[arm]["beta"] += 1
        self._means[arm] = (
            self._params[arm]["alpha"]
            /(self._params[arm]["alpha"] + self._params[arm]["beta"])
        )
        return reward

    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class UCB(Bandit):
    """
    A class for implementing the UCB algorithm.
    """
    def __init__(
        self,
        k: int,
        control: int,
        reward: Callable[[int], float],
        optimize: str = "max"
    ):
        self._active_arms = [x for x in range(k)]
        self._control = control
        self._k = k
        self._optimize = optimize  # "max" for UCB1, "min" for lower confidence bound
        self._reward_fn = reward
        self._t = 1
        # Track counts, cumulative rewards, means, and reward history for each arm.
        self._counts = {x: 0 for x in range(k)}
        self._sums = {x: 0.0 for x in range(k)}
        self._means = {x: 0.0 for x in range(k)}
        self._rewards = {x: [] for x in range(k)}
    
    def calculate_probs(self) -> Dict[int, float]:
        # If any arm hasn't been pulled yet, assign equal probability among them.
        zero_count_arms = [arm for arm in self._active_arms if self._counts[arm] == 0]
        if zero_count_arms:
            prob = 1.0 / len(zero_count_arms)
            probs = {
                arm: prob if arm in zero_count_arms else 0.0
                for arm in self._active_arms
            }
            return probs
        # Compute UCB (or lower confidence bound for "min" optimization).
        ucb_values = {}
        for arm in self._active_arms:
            avg = self._sums[arm] / self._counts[arm]
            bonus = np.sqrt((2 * np.log(self._t)) / self._counts[arm])
            if self._optimize == "max":
                ucb_values[arm] = avg + bonus
            elif self._optimize == "min":
                ucb_values[arm] = avg - bonus
            else:
                raise ValueError("`optimize` must be one of: ['max', 'min']")
        if self._optimize == "max":
            best_arm = max(ucb_values, key=ucb_values.get)
        elif self._optimize == "min":
            best_arm = min(ucb_values, key=ucb_values.get)
        probs = {
            arm: 1.0 if arm == best_arm else 0.0
            for arm in self._active_arms
        }
        return probs
    
    def control(self) -> int:
        return self._control
    
    def eliminate_arm(self, arm: int) -> None:
        self._active_arms.remove(arm)
        self._k -= 1
    
    def k(self) -> int:
        return self._k
    
    def probabilities(self) -> Dict[int, float]:
        assert self.k() == len(self._active_arms), "Mismatch in active arms and k"
        probs = self.calculate_probs()
        return probs
    
    def reactivate_arm(self, arm: int) -> None:
        if arm not in self._active_arms:
            self._active_arms.append(arm)
            self._active_arms.sort()
            self._k += 1
    
    def reward(self, arm: int) -> float:
        reward: Reward = self._reward_fn(arm)
        # Assert that the reward is an object of class `Reward`
        if not isinstance(reward, Reward):
            raise ValueError("The provided reward function must return `Reward` objects")
        self._rewards[arm].append(reward.outcome)
        self._counts[arm] += 1
        self._sums[arm] += reward.outcome
        self._means[arm] = self._sums[arm] / self._counts[arm]
        return reward
    
    def t(self) -> int:
        step = self._t
        self._t += 1
        return step

class Reward:
    """
    A simple class for reward functions. It has two attributes: `self.outcome`
    and `self.covariates`. Each reward function should return a reward object.
    For covariate adjusted algorithms, the reward should contain both the
    outcome as well as the corresponding covariates. For non-covariate adjusted
    algorithms, only the outcome should be specified.
    """
    def __init__(self, outcome: float, covariates: pd.DataFrame | None = None):
        self.outcome = outcome
        self.covariates = covariates