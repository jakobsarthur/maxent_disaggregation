import warnings
import numpy as np
from scipy.stats import dirichlet, gamma
import scipy.stats as stats
from .maxent_direchlet import find_gamma_maxent, dirichlet_entropy

# set the warning filter to always show warnings
warnings.simplefilter("always", UserWarning)


def generalized_dirichlet(n, shares, sds):
    """
    Generate random samples from a Generalised Dirichlet distribution
    with given shares and standard deviations.

    Reference:
    ----------------
    Plessis, Sylvain, Nathalie Carrasco, and Pascal Pernot.
    “Knowledge-Based Probabilistic Representations of Branching Ratios in
    Chemical Networks: The Case of Dissociative Recombinations.”
    The Journal of Chemical Physics 133, no. 13 (October 7, 2010): 134110.
    https://doi.org/10.1063/1.3479907.

    Parameters:
    ------------
    n (int): Number of samples to generate.
    shares (array-like): best-guess (mean) values for the shares. Must sum to 1!y.
    sds (array-like): Array of standard deviations for the shares.

    Returns:
    ------------
    tuple: A tuple containing:
        - sample (ndarray): An array of shape (n, lentgh(shares)) containing the generated samples.
        - None: Placeholder for compatibility with other functions (always returns None).
    """

    alpha2 = (shares / sds) ** 2
    beta2 = shares / (sds) ** 2
    k = len(alpha2)
    x = np.zeros((n, k))
    for i in range(k):
        x[:, i] = gamma.rvs(alpha2[i], scale=1 / beta2[i], size=n)
    sample = x / x.sum(axis=1, keepdims=True)
    return sample, None


def dirichlet_max_ent(n: int, shares: np.ndarray | list, **kwargs):
    """
    Generate samples from a Dirichlet distribution with maximum entropy.
    This function computes the gamma parameter that maximizes the entropy
    of the Dirichlet distribution given the input shares. It then generates
    `n` samples from the resulting Dirichlet distribution.
    Parameters:
        n (int): The number of samples to generate.
        shares (array-like): The input shares (probabilities) that define
            the Dirichlet distribution.
        **kwargs: Additional keyword arguments passed to the `find_gamma_maxent`
            function.
    Returns:
        tuple: A tuple containing:
            - sample (ndarray): An array of shape (n, len(shares)) containing
              the generated samples.
            - gamma_par (float): The computed gamma parameter that maximizes
              the entropy of the Dirichlet distribution.
    """

    gamma_par = find_gamma_maxent(shares, eval_f=dirichlet_entropy, **kwargs)
    sample = sample_dirichlet(shares * gamma_par, size=n, **kwargs)
    return sample, gamma_par


def sample_shares(
    n: int,
    shares: np.ndarray | list,
    sds: np.ndarray | list = None,
    max_iter: int = 1e3,
    grad_based: bool = False,
    threshold_shares: float = 0.1,
    threshold_sd: float = 0.2,
    **kwargs,
):
    """
    Samples from a distribution of shares based on given means and standard deviations.

    This function generates samples of shares using either a generalized Dirichlet
    distribution, a maximum entropy Dirichlet distribution, or a combination of both,
    depending on the availability of mean and standard deviation inputs.

    Parameters:
    ----------
    n : int
        Number of samples to generate.
    shares : np.ndarray | list
        Array or list of mean values for the shares. These should sum to 1 if fully specified.
    sds : np.ndarray | list, optional
        Array or list of standard deviations for the shares. If not provided, defaults to NaN.
    max_iter : float, optional
        Maximum number of iterations for optimization algorithms. Default is 1e3.
    grad_based : bool, optional
        Whether to use gradient-based optimization for maximum entropy Dirichlet sampling.
        Default is False.
    threshold_shares : float, optional
        Threshold for the relative difference between the sample mean and the
        specified shares. If the difference exceeds this threshold, a warning is raised.
        Default is 0.1 (10%).
    threshold_sd : float, optional
        Threshold for the relative difference between the sample standard deviation and
        the specified sds. If the difference exceeds this threshold, a warning is raised.
        Default is 0.2 (20%).
    **kwargs : dict
        Additional keyword arguments passed to the underlying sampling functions.

    Returns:
    -------
    sample : np.ndarray
        A 2D array of shape (n, K), where K is the number of shares, containing the sampled values.
    gamma_par : np.ndarray
        Parameters of the Dirichlet or generalized Dirichlet distribution used for sampling.

    Notes:
    -----
    - If both means and standard deviations are provided for all shares, the generalized
      Dirichlet distribution is used.
    - If only means are provided, the maximum entropy Dirichlet distribution is used.
    - If a mix of known and unknown means/standard deviations is provided, a hierarchical
      approach is used to sample the shares.
    - The function raises warnings if standard deviations are provided without corresponding
      mean values, as this is not recommended.

    Raises:
    ------
    ValueError
        If `na_action` is not "fill" or "remove".
        If the shares do not sum to 1 and `na_action` is not set to "fill".
    AssertionError
        If the resulting sample shape does not match the expected dimensions.
    """
    if sds is None:
        sds = np.full_like(shares, np.nan)
    shares = np.asarray(shares)
    sds = np.asarray(sds)

    K = len(shares)
    have_mean_only = np.isfinite(shares) & ~np.isfinite(sds)
    have_sd_only = np.isfinite(sds) & ~np.isfinite(shares)
    if np.sum(have_sd_only) > 0:
        warnings.warn(
            "You have standard deviations for shares without a best guess estimate. This is not recommended, please check your inputs. These will be treated as missing values and ignored for the calculation."
        )
    have_both = np.isfinite(shares) & np.isfinite(sds)

    if np.all(have_both):
        # use generalized dirichlet
        sample, gamma_par = generalized_dirichlet(n, shares, sds)
    elif np.all(have_mean_only):
        # maximize entropy for dirichlet
        sample, gamma_par = dirichlet_max_ent(
            n, shares, grad_based=grad_based, **kwargs
        )
    elif np.isfinite(shares).sum()==0:
        # no information on the shares, use uniform dirichlet
        shares = np.asarray([1/len(shares)]*len(shares))
        gamma_par = len(shares)  # the maximum entropy solution has a concentration parameter equal to the number of shares
        sample = sample_dirichlet(shares=shares, gamma_par=gamma_par, size=n)
        # break out because it does not need to check the means and sd's 
        return sample, None
    
    else:
        # option 1: has partial mean and no sd: assign unkown means (1-sum(known means))/N_unknown_means and use max ent dirichlet
        if np.sum(have_mean_only) > 0 and np.sum(have_both) == 0:
            remaining_share = (1 - np.sum(shares[have_mean_only])) / np.sum(
                ~have_mean_only
            )
            shares[~have_mean_only] = remaining_share
            sample, gamma_par = dirichlet_max_ent(
                n, shares, grad_based=grad_based, **kwargs
            )
        # option 2: has partial both mean and sd: use double layer dirichlet
        elif np.sum(have_both) > 0:
            sum_shares = shares[have_both].sum()
            haveboth_indices = np.where(have_both)[0]
            firstlayer_shares = [1 - sum_shares, sum_shares]
            firstlayer_sample = sample_dirichlet(firstlayer_shares, size=n, **kwargs)

            if np.sum(have_mean_only) == 0:
                # use maxent dirichlet for unkown shares
                remaining_share = (1 - sum_shares) / np.sum(~have_both)
                unknown_shares = [remaining_share] * np.sum(~have_both)
                unknown_shares = unknown_shares / np.sum(unknown_shares)
                unknown_sample, gamma_par = dirichlet_max_ent(
                    n, unknown_shares, grad_based=grad_based, **kwargs
                )
                unknown_indices = np.where(~have_both)[0]
                first_sample = firstlayer_sample[:, 0].reshape(-1, 1) * unknown_sample

            else:
                # use maxent dirichlet for unkown shares and the known shares without sd
                remaining_share = (
                    1 - sum_shares - shares[have_mean_only].sum()
                ) / np.isnan(shares).sum()
                unknown_indices = np.where(~have_both)[0]
                unknown_shares = [remaining_share] * np.isnan(shares).sum()
                mean_only_shares = list(shares[have_mean_only]) + unknown_shares
                mean_only_shares = mean_only_shares / np.sum(mean_only_shares)
                meanonly_indices = np.where(have_mean_only)[0]
                mean_only_indices = np.concatenate((meanonly_indices, unknown_indices))
                mean_only_sample, gamma_par = dirichlet_max_ent(
                    n, mean_only_shares, grad_based=grad_based, **kwargs
                )
                first_sample = firstlayer_sample[:, 0].reshape(-1, 1) * mean_only_sample

            # use generalized dirichlet for known shares
            known_shares = shares[have_both] / np.sum(shares[have_both])
            known_sds = sds[have_both] / np.sum(shares[have_both])
            known_sample, _ = generalized_dirichlet(n, known_shares, known_sds)
            second_sample = firstlayer_sample[:, 1].reshape(-1, 1) * known_sample

            # combine the two samples
            sample = np.hstack((first_sample, second_sample))
            indices = np.concatenate((unknown_indices, haveboth_indices))
            # reorder the sample to match the original shares
            sample[:, indices] = sample

            assert sample.shape == (
                n,
                K,
            ), f"sample shape is {sample.shape} but should be {(n, K)}"

    # check if sample means and standard deviations deviate more than the threshold values:
    sample_mean = np.mean(sample, axis=0)
    sample_sd = np.std(sample, axis=0)
    diff_mean = np.abs(sample_mean - shares) / shares
    diff_sd = np.abs(sample_sd - sds) / sds
    means_above_threshold = diff_mean > threshold_shares
    indices_above_threshold = np.where(means_above_threshold)[0]
    if np.any(means_above_threshold):
        warnings.warn(
            f"The generated samples for the shares have a mean that is more than {threshold_shares*100}% different from the specified shares. Please check your inputs. Reasons for this could be large relative uncertainties for the shares, or a small number of samples. To surpress this warning you can set a higher threshold_shares."
        )
        print(
            f"Shares above threshold: {diff_mean[means_above_threshold]},\
             shares: {shares[means_above_threshold]}, sample_mean: {sample_mean[means_above_threshold]},\
             indices: {indices_above_threshold}"
        )
    sds_above_threshold = diff_sd > threshold_sd
    indices_above_threshold = np.where(sds_above_threshold)[0]
    if np.any(sds_above_threshold):
        warnings.warn(
            f"The generated samples for the shares have a standard deviation that is more than {threshold_sd*100}% different from the specified sd's. Please note that the specified sd's might be incompetibale with the other constraints. Please check your inputs. To surpress this warning you can set a higher threshold_sd."
        )
        print(
            f"Sds above threshold: {diff_sd[sds_above_threshold]},\
             sds: {sds[sds_above_threshold]}, sample_sd: {sample_sd[sds_above_threshold]},\
             indices: {indices_above_threshold}"
        )

    return sample, gamma_par


def sample_dirichlet(
    shares,
    size=None,
    gamma_par=None,
    threshold_dirichlet=0.01,
    force_nonzero_samples=True,
    **kwargs,
    ):
    """
    A wrapper function to sample from a Dirichlet distribution with a
    given set of shares and gamma concentration parameter.

    It differs from the default Dirichlet distroibution in that when the

    For each variable \eqn{i} whose mean value (\eqn{\alpha_i = \gamma_{par} \cdot share_i})
    that is below a `threshold`, a fallback parametrization of the Gamma distribution
    (which is used for sampling from the Dirichlet distribution) is applied to avoid
    zero or near-zero sampling. This is especially useful for very
    small shape parameters, which can cause numerical issues in in the dirichlet sampling.
    The following pragmatic workaround is used that sets:
        - \eqn{\alpha_i = 1} (shape) for shares below `threshold`
        - \eqn{rate = 1 / \alpha_i} ensuring less extreme values.
    For more details, see the discussion in [rgamma()] under "small shape values" and
    the references there. This approach helps mitigate issues where numeric precision
    can push small Gamma-distributed values to zero (see
    https://stat.ethz.ch/R-manual/R-devel/library/stats/html/GammaDist.html).
    Note however that fix changes the expectation values (means) of the sampled parameters
    such that they can deviate from the inputed shares. If this is undesired
    set force_nonzero_samples=False.



    Parameters:
    -----------
    size : int
        The number of samples to generate.
    shares : array-like
        The input shares (probabilities) that define the Dirichlet distribution.
    gamma_par : float
        The gamma parameter that scales the shares for the Dirichlet distribution.
    threshold : float
        The threshold below which the shares are adjusted to avoid zero sampling.
    force_nonzero_samples : bool
        If True, forces non-zero samples by adjusting alphas and rate/scale within
        the gamma distribution. This may lead to biased means of the samples.
        If False, uses the original scipy implementation of the Dirichlet distribution.
        Note that in the case of very small alphas, this may lead to a large number zeros
        in the samples due to numerical issues. The means are unbiased though.

    Methods:
    --------
    sample():
        Generates samples from the Dirichlet distribution.
    """

    if gamma_par is None:
        alpha = np.asarray(shares)
    else:
        alpha = np.asarray(shares) * gamma_par

    if not force_nonzero_samples:
        print("Using scipy dirichlet!!!!!")
        return stats.dirichlet.rvs(alpha, size=size)
    else:
        l = len(alpha)
        rate = np.ones(l)
        rate[alpha < threshold_dirichlet] = 1 / alpha[alpha < threshold_dirichlet]
        alpha[alpha < threshold_dirichlet] = 1
        x = gamma.rvs(alpha, scale=1 / rate, size=(size, l))
        sample = x / x.sum(axis=1, keepdims=True)
        return sample
