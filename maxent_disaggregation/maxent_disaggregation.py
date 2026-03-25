import numpy as np
from .shares import sample_shares
from .aggregate import sample_aggregate


def maxent_disagg(
    n: int,
    mean_0: float,
    shares: np.ndarray | list,
    sd_0: float = None,
    min_0: float = 0,
    max_0: float = np.inf,
    sds: np.ndarray | list = None,
    log: bool = True,
    grad_based: bool = False,
    return_shares: bool = False,
    return_aggregate: bool = False,
    max_iter: int = 1e3,
    suppress_warnings: bool = False,
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
        The best guesses for the shares. The sum of the shares should be 1 (unless there are NA's). Use np.nan for NA's.
    sd_0:
        The standard deviation of the aggregate value. Set to None or NA if not available.
    min:
        The lower boundary of the aggregate value.
    max:
        The upper boundary of the aggregate value.
    sds:
        The standard deviations of the shares. Set to None if not available. Use np.nan for NA's.
    log:
        If True, the lognormal distribution is used for the aggregate value when a mean
        and a standard deviation are provided. If False, samples are drawn from a truncated
        normal distribution with optimised Gaussian parameters to fit the observed mean and standard deviation. 
        Note that this is the general maximum entropy solution for bounded data. 
        Default is True to use lognormal.
    suppress_warnings : bool, optional
        If True, suppress warnings about sample means and standard deviations deviating
        from the specified values. Default is False.

    Returns
    -------
    sample_disagg : np.ndarray
        A 2D array of shape (n, len(shares)) containing the generated samples.
    """


    # Check if shares and sds are numpy arrays or lists
    if  type(shares) != np.ndarray:
        if type(shares) == list:
            shares = np.array(shares)
        else:
            raise ValueError("Shares should be a numpy array or a list. If no shares are known, set them them to np.nan")
    if sds is not None and type(sds) != np.ndarray:
        if type(sds) == list:
            sds = np.array(sds)
        else:
            raise ValueError("Sds should be a numpy array or a list, or None.")

    # check shares sum to 1
    if not np.any(np.isnan(shares)):
        if not np.isclose(np.sum(shares), 1):
            raise ValueError("Shares should sum to 1 unless there are NA values.")
    # Or are less than 1 if NA values are present
    else:
        if not np.nansum(shares) < 1:
            raise ValueError("Shares should sum to less than 1 if NA values are present.")
    # check shares and sds have the same length
    if sds is not None:
        if len(shares) != len(sds):
            raise ValueError("Shares and sds should have the same length.")

    # Checks on sd_0
    if sd_0 is not None and sd_0 < 0:
        raise ValueError("sd_0 should be non-negative, or None/NA if not available.")
    if sd_0 == 0:
        raise ValueError("sd_0 should be positive, or None/NA if not available.")
    


    # Checks on min and max
    if min_0 >= max_0:
        raise ValueError("min_0 should be less than max_0.")
    if min_0 is None:
        min_0 = -np.inf
    if max_0 is None:
        max_0 = np.inf
    
    if mean_0 < min_0 or mean_0 > max_0:
        raise ValueError("mean_0 should be between min_0 and max_0.")


    samples_agg = sample_aggregate(
        n=n, mean=mean_0, sd=sd_0, low_bound=min_0, high_bound=max_0, log=log,
        suppress_warnings=suppress_warnings,
    )
    samples_shares, gamma = sample_shares(
        n=n,
        shares=shares,
        sds=sds,
        grad_based=grad_based,
        max_iter=max_iter,
        suppress_warnings=suppress_warnings,
        **kwargs,
    )
    # Check if the shares sum to 1
    if not np.isclose(np.sum(samples_shares, axis=1), 1).all():
        raise ValueError("Shares do not sum to 1! Check your shares and sds.")
    sample_disagg = samples_shares * samples_agg[:, np.newaxis]
    if return_aggregate and return_shares:
        return sample_disagg, samples_agg, samples_shares, gamma
    return sample_disagg, gamma