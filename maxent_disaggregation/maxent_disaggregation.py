import numpy as np
from scipy.stats import truncnorm
from .shares import sample_shares


def maxent_disagg(
    n: int,
    mean_0: float,
    sd_0: float = None,
    min_0: float = 0,
    max_0: float = np.inf,
    shares: np.ndarray = None,
    sds: np.ndarray = None,
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
    sd_0:
        The standard deviation of the aggregate value.
    min:
        The lower boundary of the aggregate value.
    max:
        The upper boundary of the aggregate value.
    shares:
        The best guesses for the shares. The sum of the shares should be 1 (unless there are NA's).
    sds:
        The standard deviations of the shares. Set to None if not available.

    Returns
    -------
    sample_disagg : np.ndarray
        A 2D array of shape (n, len(shares)) containing the generated samples.
    """
    
    # The code below is only necessary if we want to accept arrays as input
    # if sd_0 is not None:
    #     if not type(mean_0)==type(sd_0)==type(min)==type(max):
    #         raise ValueError('All arguments should be of the same type.')
    # else:
    #     if not type(mean_0)==type(min)==type(max):
    #         raise ValueError('All arguments should be of the same type.')

    samples_agg = sample_aggregate(n=n, mean=mean_0, sd=sd_0, min=min_0, max=max_0)
    samples_shares = sample_shares(n=n, shares=shares, sds=sds)
    sample_disagg = samples_shares * samples_agg[:, np.newaxis]
    return sample_disagg


def sample_aggregate(
    n: int,
    mean: float,
    sd: float = None,
    min: float = 0,
    max: float = np.inf,
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
    min:
        The lower boundary of the aggregate value.
    max:
        The upper boundary of the aggregate value.
    """


    
    if mean is not None and sd is not None and min == -np.inf and max == np.inf:
        # Normal distribution
        return np.random.normal(loc=mean, scale=sd, size=n)
    elif mean is not None and sd is not None:
        # Truncated normal
        a, b = (min - mean) / sd, (max - mean) / sd
        return truncnorm.rvs(a, b, loc=mean, scale=sd, size=n)
    elif mean is not None and sd is None and min == 0 and max == np.inf:
        # Exponential
        return np.random.exponential(scale=mean, size=n)
    elif mean is None and sd is None and np.isfinite(min) and np.isfinite(max):
        # Uniform
        return np.random.uniform(low=min, high=max, size=n)
    else:
        raise ValueError('Case not implemented atm.')


def sample_shares(n, shares, sds=None):
    if sds is None:
        sds = np.zeros_like(shares)
    samples = np.random.normal(loc=shares, scale=sds, size=(n, len(shares)))
    samples = np.clip(samples, 0, 1)
    samples /= samples.sum(axis=1, keepdims=True)
    return samples
