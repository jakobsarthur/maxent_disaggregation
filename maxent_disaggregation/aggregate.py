from scipy.stats import truncnorm, lognorm
from scipy.optimize import least_squares
import warnings
import numpy as np


def sample_aggregate(
    n: int,
    mean: float = None,
    sd: float = None,
    low_bound: float = 0,
    high_bound: float = np.inf,
    log: bool = True,
) -> np.ndarray:
    """

    Generate random aggregate values based on the information provided.
    The distribution from which to sample is determined internally based on the information
    provided by the user."

    Parameters
    ----------
    n : int
        The number of samples to generate.
    mean:
        The best guess of the aggregate value.
    sd:
        The standard deviation of the aggregate value.
    low_bound:
        The lower boundary of the aggregate value.
    high_bound:
        The upper boundary of the aggregate value.
    log:
        If True, the lognormal distribution is used for the aggregate value when a mean
        and a standard deviation are provided. If False, samples are drawn from a truncated
        normal distribution, which is the maximum entropy solution but produces a biased mean.
        Default is True
    """

    # harmonize input of sd
    if sd is not None and np.isnan(sd):
        sd = None

    # Determine distribution and sample
    # Normal distribution
    if (
        mean is not None
        and sd is not None
        and (low_bound == -np.inf or low_bound is None)
        and (high_bound == np.inf or high_bound is None)
    ):
        # Normal distribution
        return np.random.normal(loc=mean, scale=sd, size=n)
    
    # Truncated normal or lognormal
    elif mean is not None and sd is not None:
        if log == False:
            # Truncated normal from observed parameters
            sample = sample_truncnorm(mean, sd, low_bound, high_bound, size=n)
            return sample
        else:
            # use lognormal
            if low_bound < 0:
                warnings.warn(
                    "You provided a negative lower bound but the lognormal distribution cannot be used with negative values. Setting low_bound to 0. Alternatively set log=False."
                )
                low_bound = 0
            if high_bound != np.inf:
                warnings.warn(
                    "You provided a finite high bound, currently this not supported for the lognormal distribution. High bound is ignored. Alternatively set log=False."
                )
            # Lognormal distribution
            sigma = np.sqrt(np.log(1 + (sd / mean) ** 2))
            mu = np.log(mean) - 0.5 * sigma**2
            return lognorm.rvs(s=sigma, scale=np.exp(mu), size=n)

    elif mean is not None and sd is None and low_bound == 0 and (high_bound == np.inf or high_bound is None):
        # Exponential
        return np.random.exponential(scale=mean, size=n)
    elif (
        mean is None
        and sd is None
        and np.isfinite(low_bound)
        and np.isfinite(high_bound)
    ):
        # Uniform
        return np.random.uniform(low=low_bound, high=high_bound, size=n)
    elif mean is not None and sd is None and low_bound not in [0, None]:
        raise ValueError("Case with mean, no sd, and non-zero lower bound, or non-finite high bound is not implemented at the moment.")
    elif mean is not None and sd is None and np.isfinite(high_bound):
        raise ValueError("Case with mean, no sd, and non-zero lower bound, or non-finite high bound is not implemented at the moment.")
    else:
        raise ValueError("Combination of inputs not implemented. Please check the input values.")



def sample_truncnorm(obs_mean, obs_std, a, b, size=1000):
    """
    Draw random samples from a truncated normal distribution
    given observed mean, standard deviation, and bounds.

    Parameters
    ----------
    obs_mean : float
        Observed mean used to infer the underlying normal distribution's location.
    obs_std : float
        Observed standard deviation used to infer the underlying normal distribution's scale.
    a : float
        Lower truncation bound (expressed in the same units as obs_mean/obs_std).
    b : float
        Upper truncation bound (expressed in the same units as obs_mean/obs_std).
    size : int, optional
        Number of random samples to draw. Default is 1000.

    Returns
    -------
    numpy.ndarray
        1-D array of random variates drawn from the truncated normal distribution.

    Notes
    -----
    This function relies on estimate_truncnormparams(obs_mean, obs_std, a, b) to compute
    parameters (mu, sigma, alpha, beta) suitable for scipy.stats.truncnorm.rvs, where
    mu and sigma are the location and scale of the underlying normal distribution and
    alpha, beta are the standardized truncation limits accepted by scipy.stats.truncnorm.


    Examples
    --------
    >>> samples = sample_truncnorm(10.0, 2.0, 5.0, 15.0, size=500)
    >>> samples.shape(500,)
    """
    mu, sigma, alpha, beta = estimate_truncnormparams(obs_mean, obs_std, a, b)
    return truncnorm.rvs(alpha, beta, loc=mu, scale=sigma, size=size)


def estimate_truncnormparams(obs_mean, obs_std, a, b, mu_init=None, sigma_init=None):
    """
    Estimate the Gaussian  parameters of a truncated normal distribution given observed
    statistics. This function finds the parameters (mu, sigma) of a truncated normal
    distribution that best match the observed mean and standard deviation, given truncation bounds.

    Parameters
    ----------
    obs_mean : float
        The observed mean of the truncated distribution.
    obs_std : float
        The observed standard deviation of the truncated distribution.
    a : float
        The lower truncation bound.
    b : float
        The upper truncation bound.
    mu_init : float, optional
        Initial guess for the location parameter (mu). Defaults to obs_mean.
    sigma_init : float, optional
        Initial guess for the scale parameter (sigma). Defaults to obs_std.
    
    Returns
    -------
    mu_opt : float
        Optimal location parameter of the underlying normal distribution.
    sigma_opt : float
        Optimal scale parameter of the underlying normal distribution.
    alpha_opt : float
        Standardized lower truncation bound: (a - mu_opt) / sigma_opt.
    beta_opt : float
        Standardized upper truncation bound: (b - mu_opt) / sigma_opt.
    
    Notes
    -----
    The function uses least squares optimization to minimize the difference between
    the theoretical and observed moments of the truncated normal distribution.
    The scale parameter (sigma) is constrained to be positive (>= 1e-10).
    
    Examples
    --------
    >>> mu, sigma, alpha, beta = estimate_truncnormparams(5.0, 1.5, 0, 10)
    >>> print(f"Estimated mu: {mu:.2f}, sigma: {sigma:.2f}")
    """

    
    def objective(params):
        mu, sigma = params
        alpha = (a - mu) / sigma
        beta = (b - mu) / sigma
        tn = truncnorm(alpha, beta, loc=mu, scale=sigma)
        return [tn.mean() - obs_mean, tn.std() - obs_std]
    
    if mu_init is None:
        mu_init = obs_mean
    if sigma_init is None:
        sigma_init = obs_std

    res = least_squares(
        objective, 
        x0=[mu_init, sigma_init],
        bounds=([-np.inf, 1e-10], [np.inf, np.inf])
    )
    
    mu_opt, sigma_opt = res.x
    alpha_opt = (a - mu_opt) / sigma_opt
    beta_opt = (b - mu_opt) / sigma_opt
    
    return mu_opt, sigma_opt, alpha_opt, beta_opt
