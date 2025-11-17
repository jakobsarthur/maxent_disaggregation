from scipy.stats import truncnorm, lognorm
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

    if (
        mean is not None
        and sd is not None
        and (low_bound == -np.inf or low_bound is None)
        and (high_bound == np.inf or high_bound is None)
    ):
        # Normal distribution
        return np.random.normal(loc=mean, scale=sd, size=n)
    elif mean is not None and sd is not None:
        if log == False:
            # Truncated normal
            a, b = (low_bound - mean) / sd, (high_bound - mean) / sd
            return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n)
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




