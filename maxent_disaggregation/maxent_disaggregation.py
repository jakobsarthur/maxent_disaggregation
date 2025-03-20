import numpy as np


def maxent_disagg(
    n: int,
    mean_0: float,
    sd_0: float = None,
    min: float = 0,
    max: float = np.inf,
    shares: np.array = None,
    sds: np.array = None,
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
    sample_agg = ragg(n=n, mean=mean_0, sd=sd_0, min=min, max=max)
    sample_shares = rshares(n=n, shares=shares, sds=sds)
    sample_disagg = sample_shares * sample_agg[:, np.newaxis]
    return sample_disagg


def ragg(n, mean, sd=None, min=0, max=np.inf):
    if sd is None:
        sd = mean * 0.1  # Assuming a default standard deviation if not provided
    samples = np.random.normal(loc=mean, scale=sd, size=n)
    samples = np.clip(samples, min, max)
    return samples


def rshares(n, shares, sds=None):
    if sds is None:
        sds = np.zeros_like(shares)
    samples = np.random.normal(loc=shares, scale=sds, size=(n, len(shares)))
    samples = np.clip(samples, 0, 1)
    samples /= samples.sum(axis=1, keepdims=True)
    return samples
