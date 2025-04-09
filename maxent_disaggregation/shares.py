import numpy as np
from scipy.stats import dirichlet, gamma
from .maxent_direchlet import find_gamma_maxent2, dirichlet_entropy
import warnings


def generalized_dirichlet(n, shares, sds):
    """
    Generate random samples from a Generalised Dirichlet distribution with given shares and standard deviations.
    
    Reference:
    ----------------
    Plessis, Sylvain, Nathalie Carrasco, and Pascal Pernot. “Knowledge-Based Probabilistic Representations of Branching Ratios in Chemical Networks: The Case of Dissociative Recombinations.”
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


def dirichlet_max_ent(n, shares, **kwargs):

    gamma_par = find_gamma_maxent2(shares, eval_f=dirichlet_entropy, **kwargs)
    sample = dirichlet.rvs(shares * gamma_par, size=n)
    return sample, gamma_par


def sample_shares(
    n, shares, 
    sds=None, 
    na_action="fill", 
    max_iter=1e3, 
    grad_based=False, 
    **kwargs
):
    if sds is None:
        sds = np.full_like(shares, np.nan)

    if na_action == "remove":
        sds = sds[~np.isnan(shares)]
        shares = shares[~np.isnan(shares)]
    elif na_action == "fill":
        shares[np.isnan(shares)] = (1 - np.nansum(shares)) / np.isnan(shares).sum()
    else:
        raise ValueError('na_action must be either "remove" or "fill"!')

    if not np.isclose(np.sum(shares), 1):
        raise ValueError(
            'shares must sum to one! If you have NAs in your shares consider setting "na_action" to "fill".'
        )

    K = len(shares)
    have_mean_only = np.isfinite(shares) & ~np.isfinite(sds)
    have_sd_only = np.isfinite(sds) & ~np.isfinite(shares)
    if np.sum(have_mean_only) > 0:
        warnings.warn(
            "You have standard deviations for shares without a best guess estimate. This is not recommended, please check your inputs. These will be treated as missing values and removed from the calculation."
        )
    have_both = np.isfinite(shares) & np.isfinite(sds)

    if np.all(have_both):
        # use generalized dirichlet
        sample, gamma = generalized_dirichlet(n, shares, sds)
    elif np.all(have_mean_only):
        # maximize entropy for dirichlet
        sample, gamma = dirichlet_max_ent(n, shares, grad_based=grad_based, **kwargs)
    else:
        #
        sample = np.zeros((n, K))
        if np.sum(have_both) > 0:
            sample[:, have_both] = rbeta3(
                n, shares[have_both], sds[have_both], max_iter=max_iter
            )
        if np.sum(have_mean_only) > 0:
            alpha2 = shares[have_mean_only] / np.sum(shares[have_mean_only])
            sample_temp, gamma = dirichlet_max_ent(n, alpha2, **kwargs)
            sample[:, have_mean_only] = sample_temp * (
                1 - sample.sum(axis=1, keepdims=True)
            )

    return sample, gamma


def rbeta3(n, shares, sds, fix=True, max_iter=1e3):
    var = sds**2
    undef_comb = (shares * (1 - shares)) < var
    if not np.all(~undef_comb):
        if fix:
            var[undef_comb] = shares[undef_comb] ** 2
        else:
            raise ValueError(
                "The beta distribution is not defined for the parameter combination you provided! sd must be smaller or equal sqrt(shares*(1-shares))"
            )

    alpha = shares * (((shares * (1 - shares)) / var) - 1)
    beta = (1 - shares) * (((shares * (1 - shares)) / var) - 1)

    k = len(shares)
    x = np.zeros((n, k))
    for i in range(k):
        x[:, i] = np.random.beta(alpha[i], beta[i], size=n)

    larger_one = x.sum(axis=1) > 1
    count = 0
    while np.sum(larger_one) > 0:
        for i in range(k):
            x[larger_one, i] = np.random.beta(
                alpha[i], beta[i], size=np.sum(larger_one)
            )
        larger_one = x.sum(axis=1) > 1
        count += 1
        if count > max_iter:
            raise ValueError(
                "max_iter is reached. the combinations of shares and sds you provided does allow to generate `n` random samples that are not larger than 1. Either increase max_iter, or change parameter combination."
            )
    return x
