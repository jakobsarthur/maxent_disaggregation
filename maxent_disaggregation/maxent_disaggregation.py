import numpy as np
from scipy.stats import truncnorm, lognorm
from .shares import sample_shares
import warnings


def maxent_disagg(
    n: int,
    mean_0: float,
    shares: np.ndarray | list,
    sd_0: float = None,
    min_0: float = 0,
    max_0: float = np.inf,
    sds: np.ndarray | list = None,
    log: bool = False,
    grad_based: bool = False,
    return_shares: bool = False,
    return_aggregate: bool = False,
    max_iter: int = 1e3,
    **kwargs,
) -> np.ndarray:
    """
    Generate random disaggregates based on the maximum entropy principle.
    Creates a random sample of disaggregates based on the information provided.
    The aggregate and the shares are sampled independently. The distribution
    from which to sample is determined internally based on the information
    provided by the user.


    Parameters
    ----------
    n : int
        The number of samples to generate.
    mean_0:
        The best guess of the aggregate value.
    shares:
        The best guesses for the shares. The sum of the shares should be 1 (unless there are NA's).
    sd_0:
        The standard deviation of the aggregate value.
    min:
        The lower boundary of the aggregate value.
    max:
        The upper boundary of the aggregate value.
    sds:
        The standard deviations of the shares. Set to None if not available.

    Returns
    -------
    sample_disagg : np.ndarray
        A 2D array of shape (n, len(shares)) containing the generated samples.
    """

    # Check if shares and sds are numpy arrays or lists
    if type(shares) != np.ndarray:
        if type(shares) == list:
            shares = np.array(shares)
        else:
            raise ValueError("Shares should be a numpy array or a list.")
    if type(sds) != np.ndarray and sds is not None:
        if type(sds) == list:
            sds = np.array(sds)
        else:
            raise ValueError("Sds should be a numpy array or a list.")

    # check shares contain at least one non-NA value
    if np.all(np.isnan(shares)):
        raise ValueError("Shares should contain at least one non-NA value.")
    # check shares sum to 1
    elif not np.any(np.isnan(shares)):
        if not np.isclose(np.sum(shares), 1):
            raise ValueError("Shares should sum to 1 unless there are NA values.")
    # check shares and sds have the same length
    if sds is not None:
        if len(shares) != len(sds):
            raise ValueError("Shares and sds should have the same length.")

    # The code below is only necessary if we want to accept arrays as input
    # if sd_0 is not None:
    #     if not type(mean_0)==type(sd_0)==type(min)==type(max):
    #         raise ValueError('All arguments should be of the same type.')
    # else:
    #     if not type(mean_0)==type(min)==type(max):
    #         raise ValueError('All arguments should be of the same type.')

    samples_agg = sample_aggregate(
        n=n, mean=mean_0, sd=sd_0, low_bound=min_0, high_bound=max_0, log=log
    )
    samples_shares, gamma = sample_shares(
        n=n,
        shares=shares,
        sds=sds,
        grad_based=grad_based,
        max_iter=max_iter,
        **kwargs,
    )
    # Check if the shares sum to 1
    if not np.isclose(np.sum(samples_shares, axis=1), 1).all():
        raise ValueError("Shares do not sum to 1! Check your shares and sds.")
    sample_disagg = samples_shares * samples_agg[:, np.newaxis]
    if return_aggregate and return_shares:
        return sample_disagg, samples_agg, samples_shares, gamma
    return sample_disagg, gamma


def sample_aggregate(
    n: int,
    mean: float,
    sd: float = None,
    low_bound: float = 0,
    high_bound: float = np.inf,
    log: bool = False,
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
    """

    if (
        mean is not None
        and sd is not None
        and low_bound == -np.inf
        and high_bound == np.inf
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

    elif mean is not None and sd is None and low_bound == 0 and high_bound == np.inf:
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
    else:
        raise ValueError("Case not implemented atm.")
