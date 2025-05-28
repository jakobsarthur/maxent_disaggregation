import warnings
import numpy as np
from scipy.stats import truncnorm, lognorm
from .shares import sample_shares
import matplotlib.pyplot as plt
import corner


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
    log:
        If True, the lognormal distribution is used for the aggregate value when a mean
        and a standard deviation are provided. If False, samples are drawn from a truncated
        normal distribution, which is the maximum entropy solution but produces a biased mean.
        Default is True

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
    else:
        raise ValueError("Case not implemented atm.")


def plot_samples_hist(
    samples,
    mean_0=None,
    sd_0=None,
    shares=None,
    sds=None,
    logscale=False,
    plot_agg=True,
    plot_sample_mean=True,
    title=None,
    xlabel=None,
    ylabel=None,
    legend_labels=None,
    save=False,
    filename=None,
):
    """
    Plot histograms of samples.
    :param samples: 2D array of samples
    :param shares: list of shares for each sample
    :param sds: list of standard deviations for each sample
    """

    max_height = 0  # Track the maximum height of all histograms

    for i in range(samples.shape[1]):
        if sds is not None:
            std = sds[i]
        else:
            std = sds
        if shares is not None:
            share = shares[i]
        else:
            share = shares
        if legend_labels is not None:
            label = legend_labels[i]
        else:
            label = f"Share {i+1} input = {share}, SD = {std}"
        x = plt.hist(
            samples[:, i],
            bins=50,
            alpha=0.5,
            label=label,
            density=True,
        )
        max_height = max(max_height, max(x[0]))  # Update the maximum height

        if plot_sample_mean:
            plt.axvline(
                x=samples[:, i].mean(),
                color=x[2][0].get_facecolor(),
                linestyle="--",
                label=f"Share {i+1} sample mean",
            )
    if plot_agg:
        if legend_labels is not None:
            label = legend_labels[-1]
        else:
            label = f"Aggregate input= {mean_0}, SD = {sd_0}"
        x = plt.hist(
            samples.sum(axis=1),
            bins=50,
            alpha=0.5,
            label=label,
            density=True,
        )
        max_height = max(max_height, max(x[0]))  # Update the maximum height
        if plot_sample_mean:
            plt.axvline(
                x=samples.sum(axis=1).mean(),
                color=x[2][0].get_facecolor(),
                linestyle="--",
                label="Aggregate sample mean",
            )

    if logscale:
        plt.xscale("log")
    
    # Set the y-axis limit slightly above the maximum height
    plt.ylim(0, max_height * 1.1)

    plt.legend(frameon=True, fontsize=8,)
    if xlabel is None:
        xlabel = "Value"
    if ylabel is None:
        ylabel = "Probability density"
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if title==None:
        title = "MaxEnt Disaggregation"
    plt.title(title)

    if save:
        if filename is None:
            raise ValueError("Filename must be provided if save is True.")
        plt.savefig(filename)
    
    plt.show()


def plot_covariances(
    samples,
    title=None,
    labels=None,
    save=False,
    filename=None,
):
    """
    Plot correlations between samples.
    :param samples: 2D array of samples
    :param shares: list of shares for each sample
    :param sds: list of standard deviations for each sample
    """
  
    if not labels:
        labels = [f"Share {i+1}" for i in range(samples.shape[1])]

    corner.corner(samples,
             labels=labels,
             quantiles=[0.16, 0.5, 0.84],
             show_titles=True,
             title_kwargs={"fontsize": 12},
             label_kwargs={"fontsize": 12},
             smooth=True,
             smooth1d=True,
             fill_contours=False,
             levels=(0.68, 0.95),
             bins=50,
             plot_datapoints=True,
             color="C0",
            )

    if title==None:
        title = "MaxEnt Disaggregation Covariances"
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save:
        if filename is None:
            raise ValueError("Filename must be provided if save is True.")
        plt.savefig(filename)
    
    plt.show()