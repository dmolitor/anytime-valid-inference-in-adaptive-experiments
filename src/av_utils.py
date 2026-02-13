import numpy as np
import pandas as pd

def projection_matrix(X):
    """
    Compute the projection matrix for the matrix X.

    Parameters
    ----------
    X : numpy.ndarray
        A 2D array representing the design matrix.

    Returns
    -------
    numpy.ndarray
        The projection matrix Q @ Q.T, where Q is obtained from the QR decomposition of X.
    """
    Q, _ = np.linalg.qr(X)
    return Q @ Q.T


def MPxM(M, X):
    """
    Compute the matrix (Q^T M)^T (Q^T M) = M^T Q Q^T M,
    where Q is the orthogonal matrix from the QR decomposition of X.

    Parameters
    ----------
    M : numpy.ndarray
        A matrix (or vector) M.
    X : numpy.ndarray
        A matrix X from which the orthogonal matrix Q is computed.

    Returns
    -------
    numpy.ndarray
        The product (Q^T M)^T (Q^T M), equivalent to M^T Q Q^T M.
    """
    Q, _ = np.linalg.qr(X)
    QtM = Q.T @ M
    return QtM.T @ QtM


def MP1M(M):
    """
    Compute the outer product of the column-sum of M with itself divided by the number of rows.

    Parameters
    ----------
    M : numpy.ndarray
        A 2D array or a vector. If a vector is provided, it is treated as a single-column matrix.

    Returns
    -------
    numpy.ndarray
        The matrix (Mbar @ Mbar.T) / n, where Mbar is the sum of the columns of M and n is the number of rows.
    """
    # Ensure M is at least 2D (if M is 1D, treat it as a row vector and then transpose)
    M = np.atleast_2d(M)
    n = M.shape[0]
    Mbar = np.sum(M, axis=0)
    return np.outer(Mbar, Mbar) / n


def log_bf(t2, nu, phi, z2):
    """
    Compute the log Bayes factor for the univariate sequential t-test.

    Parameters
    ----------
    t2 : float
        Squared t-statistic.
    nu : float or int
        Degrees of freedom.
    phi : float
        Tuning parameter.
    z2 : float
        The reciprocal of the design factor (i.e. sigma^2/(se^2)).

    Returns
    -------
    float
        The computed log Bayes factor.
    """
    r = phi / (phi + z2)
    return 0.5 * np.log(r) + 0.5 * (nu + 1) * (np.log(1 + t2 / nu) - np.log(1 + r * t2 / nu))


def log_bf_multivariate(delta, n, p, d, Phi, ZtZ, s2):
    """
    Compute the log Bayes factor for the multivariate sequential F-test.

    Parameters
    ----------
    delta : numpy.ndarray
        Vector of parameter estimates (d-dimensional).
    n : int
        Sample size.
    p : int
        Number of nuisance parameters.
    d : int
        Dimension of delta (number of parameters of interest).
    Phi : numpy.ndarray or float
        Tuning parameter matrix (if d > 1) or scalar (if d == 1).
    ZtZ : numpy.ndarray or float
        The d x d matrix (or scalar) representing the inverse of the variance-covariance matrix for delta.
    s2 : float
        Estimate of the error variance.

    Returns
    -------
    float
        The log Bayes factor for the multivariate case.
    """
    if d > 1:
        normalizing_constant = 0.5 * np.log(np.linalg.det(Phi)) - 0.5 * np.log(np.linalg.det(Phi + ZtZ))
        sol = np.linalg.solve(ZtZ + Phi, ZtZ)
    else:
        normalizing_constant = 0.5 * np.log(Phi) - 0.5 * np.log(Phi + ZtZ)
        sol = ZtZ / (ZtZ + Phi)
    denom = s2 * (n - p - d)
    # Ensure delta is a column vector.
    delta = np.atleast_2d(delta)
    if delta.shape[0] == 1:
        delta = delta.T
    term1 = np.log(1 + (delta.T @ ZtZ @ delta) / denom)
    term2 = np.log(1 + (delta.T @ (ZtZ - ZtZ @ sol) @ delta) / denom)
    # Convert 1x1 arrays to scalars
    term1 = term1.item()
    term2 = term2.item()
    return normalizing_constant + 0.5 * (n - p) * (term1 - term2)


def sequential_t_p_value(estimate, se, df, phi=1):
    """
    Calculate a sequential t-test p-value for a given estimate and standard error.

    Parameters
    ----------
    estimate : float
        The point estimate (e.g., a marginal effect).
    se : float
        The standard error associated with the estimate.
    df : int or float
        Degrees of freedom.
    phi : float, optional
        Tuning parameter (default is 1).

    Returns
    -------
    dict
        Dictionary with keys:
            't_value': The computed t-statistic.
            'p_value': The sequential t-test p-value.
    """
    t_val = estimate / se
    z2 = 1 / (se ** 2)
    bf_log = log_bf(t_val ** 2, df, phi, z2)
    pval = min(1, np.exp(-bf_log))
    return {"t_value": t_val, "p_value": pval}


def sequential_t_cs(estimate, se, df, alpha=0.05, phi=1):
    """
    Calculate a sequential t-test confidence sequence (CS) for a given estimate.

    Parameters
    ----------
    estimate : float
        The point estimate (e.g., a marginal effect).
    se : float
        The standard error of the estimate.
    df : int or float
        Degrees of freedom.
    alpha : float, optional
        Significance level (default is 0.05).
    phi : float, optional
        Tuning parameter (default is 1).

    Returns
    -------
    dict
        Dictionary containing:
            't_value': The computed t-statistic.
            'p_value': The sequential p-value.
            'cs_lower': The lower bound of the confidence sequence.
            'cs_upper': The upper bound of the confidence sequence.
    """
    t_val = estimate / se
    z2 = 1 / (se ** 2)
    r = phi / (phi + z2)
    eps = 1e-8
    term = (r * alpha ** 2) ** (1 / (df + 1))
    numerator = 1 - term
    denominator = max(term - r, eps)
    radii = se * np.sqrt(df * (numerator / denominator))
    lower = estimate - radii
    upper = estimate + radii
    p_value = min(1, np.exp(-log_bf(t_val ** 2, df, phi, z2)))
    return {"t_value": t_val, "p_value": p_value, "cs_lower": lower, "cs_upper": upper}


def sequential_f_p_value(delta, n, n_params, Z, phi=1):
    """
    Calculate the sequential F-test p-value for a vector of estimates.

    Parameters
    ----------
    delta : numpy.ndarray
        Vector of estimates for the parameters of interest (length d).
    n : int
        Sample size.
    n_params : int
        Total number of parameters in the model (including nuisance parameters).
    Z : numpy.ndarray
        The d x d matrix (typically the inverse of the variance-covariance matrix for delta).
    phi : float, optional
        Tuning parameter (default is 1).

    Returns
    -------
    float
        The sequential F-test p-value.
    """
    delta = np.asarray(delta)
    d = delta.size
    p = n_params - d
    Phi = np.eye(d) * phi
    pval = min(1, np.exp(-log_bf_multivariate(delta, n, p, d, Phi, Z, 1)))
    return pval


def sequential_f_cs(delta, se, n, n_params, Z, alpha=0.05, phi=1, term=None, contrast=None):
    """
    Calculate a sequential F-test confidence sequence (CS) for multivariate parameters.

    Parameters
    ----------
    delta : numpy.ndarray
        Vector of estimates (e.g., marginal effects) for parameters of interest (length d).
    se : numpy.ndarray
        Corresponding standard errors for the estimates.
    n : int
        Total sample size.
    n_params : int
        Total number of parameters in the full model (including nuisance parameters).
    Z : numpy.ndarray
        The d x d matrix obtained by inverting the variance-covariance matrix for delta.
    alpha : float, optional
        Significance level (default is 0.05).
    phi : float, optional
        Tuning parameter (default is 1).
    term : list of str, optional
        Names for the estimates; if None, defaults to ["X1", "X2", ..., "Xd"].
    contrast : any, optional
        Contrast description (passed through).

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
            'term', 'contrast', 'estimate', 'std_error', 't_stat', 'p_value',
            'cs_lower', 'cs_upper', 'f_p_value'
    """
    delta = np.asarray(delta)
    se = np.asarray(se)
    d = delta.size
    if term is None:
        term = [f"X{i+1}" for i in range(d)]
    p = n_params - d
    nu = n - p - 1
    tstats = delta / se
    tstats2 = tstats ** 2
    z2 = 1 / (se ** 2)
    spvalues = np.minimum(1, np.exp(-log_bf(tstats2, nu, phi, z2)))
    r = phi / (phi + z2)
    eps = 1e-8
    alpha_factor = (r * alpha ** 2) ** (1 / (nu + 1))
    numerator = 1 - alpha_factor
    denominator = np.maximum(alpha_factor - r, eps)
    radii = se * np.sqrt(nu * (numerator / denominator))
    cs_lower = delta - radii
    cs_upper = delta + radii
    f_seq_p = sequential_f_p_value(delta, n, n_params, Z, phi)
    
    result_df = pd.DataFrame({
        "term": term,
        "contrast": contrast,
        "estimate": delta,
        "std_error": se,
        "t_stat": tstats,
        "p_value": spvalues,
        "cs_lower": cs_lower,
        "cs_upper": cs_upper,
        "f_p_value": f_seq_p
    })
    return result_df

def sequential_asymptotic_cs(delta, n, propensity, lambda_param=100, alpha=0.05, sigma_hat=None, term=None, contrast=None):
    """
    Calculate the asymptotic confidence sequence (CS) for a vector of ATE estimates.

    The half-width (r_n) is computed using the asymptotic formula:
    
        r_n = sigma_hat / sqrt(n * propensity * (1 - propensity)) * 
              sqrt(((lambda_param + n) / n) * log((lambda_param + n) / (lambda_param * alpha**2)))
    
    The confidence sequence for each ATE estimate is then given by:
        cs_lower = delta - r_n
        cs_upper = delta + r_n

    Parameters
    ----------
    delta : array-like
        Vector of ATE estimates (e.g., marginal effects for treatment).
    n : int
        Total sample size.
    propensity : float
        Treatment assignment probability (e.g., mean of a 0/1 treatment variable).
    lambda_param : float, optional
        Tuning parameter (default is 100).
    alpha : float, optional
        Significance level (default is 0.05).
    sigma_hat : float, optional
        Estimated residual standard error. If not provided, assumed to be 1.
    term : array-like of str, optional
        Optional names for the estimates. If not provided, defaults to ["ATE_1", "ATE_2", ...].
    contrast : any, optional
        Optional contrast information.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns:
            - term: Names for the ATE estimates.
            - contrast: The provided contrast information.
            - estimate: The original ATE estimates (delta).
            - half_width: The computed half-width (r_n) for the confidence sequence.
            - cs_lower: Lower bound of the confidence sequence.
            - cs_upper: Upper bound of the confidence sequence.
            - n: Total sample size.
            - propensity: Treatment assignment probability.
            - lambda: The tuning parameter.
            - alpha: Significance level.
            - sigma_hat: The estimated residual standard error.
    """
    # If sigma_hat is not provided, set to 1.
    if sigma_hat is None:
        sigma_hat = 1
    # Convert delta to a NumPy array (if it is not already)
    delta = np.asarray(delta)
    # Generate default term names if not provided.
    if term is None:
        term = [f"ATE_{i+1}" for i in range(delta.size)]
    # Compute the half-width using the asymptotic formula.
    half_width = (
        sigma_hat
        / np.sqrt(n * propensity * (1 - propensity))
        * np.sqrt(
            ((lambda_param + n) / n)
            * np.log((lambda_param + n) / (lambda_param * alpha**2))
        )
    )
    # Construct confidence sequence lower and upper bounds.
    cs_lower = delta - half_width
    cs_upper = delta + half_width
    # Construct the result DataFrame.
    result_df = pd.DataFrame({
        "term": term,
        "contrast": contrast,
        "estimate": delta,
        "half_width": half_width,
        "cs_lower": cs_lower,
        "cs_upper": cs_upper,
        "n": n,
        "propensity": propensity,
        "lambda": lambda_param,
        "alpha": alpha,
        "sigma_hat": sigma_hat
    })
    return result_df