import numpy as np
from scipy.stats import dirichlet, gamma
form .maxent_dirichlet import find_gamma_maxent2, dirichlet_entropy

def rdir1(n, length, names=None):
    sample = dirichlet.rvs([1] * length, size=n)
    if names is not None:
        sample = np.array(sample)
        sample = np.core.records.fromarrays(sample.transpose(), names=names)
    return sample

def rdirg(n, shares, sds):
    if not np.array_equal(list(shares.keys()), list(sds.keys())):
        raise ValueError('shares and sds need to have the same column names. can also be both NULL')

    alpha2 = (shares / sds) ** 2
    beta2 = shares / (sds) ** 2
    k = len(alpha2)
    x = np.zeros((n, k))
    for i in range(k):
        x[:, i] = gamma.rvs(alpha2[i], scale=1/beta2[i], size=n)
    sample = x / x.sum(axis=1, keepdims=True)
    return sample

def rdir_maxent(n, shares, find_gamma_maxent2, dirichlet_entropy, **kwargs):
    out = find_gamma_maxent2(shares, eval_f=dirichlet_entropy, **kwargs)
    sample = dirichlet.rvs(shares * out['solution'], size=n)
    return sample

def rdir(n, shares, gamma, threshold=1E-2):
    alpha = gamma * shares
    l = len(alpha)
    rate = np.ones(l)
    rate[alpha < threshold] = 1 / alpha[alpha < threshold]
    alpha[alpha < threshold] = 1
    x = gamma.rvs(alpha, scale=1/rate, size=(n, l))
    sample = x / x.sum(axis=1, keepdims=True)
    return sample

def rdirichlet(n, alpha, threshold=1E-2):
    l = len(alpha)
    rate = np.ones(l)
    rate[alpha < threshold] = 1 / alpha[alpha < threshold]
    alpha[alpha < threshold] = 1
    x = gamma.rvs(alpha, scale=1/rate, size=(n, l))
    sample = x / x.sum(axis=1, keepdims=True)
    return sample

def sample_shares(n, shares, sds=None, na_action='fill', max_iter=1E3, **kwargs):
    if sds is None:
        sds = np.full_like(shares, np.nan)
    
    if na_action == 'remove':
        sds = sds[~np.isnan(shares)]
        shares = shares[~np.isnan(shares)]
    elif na_action == 'fill':
        shares[np.isnan(shares)] = (1 - np.nansum(shares)) / np.sum(np.isnan(shares))
    else:
        raise ValueError('na_action must be either "remove" or "fill"!')

    if not np.isclose(np.sum(shares), 1):
        raise ValueError('shares must sum to one! If you have NAs in your shares consider setting "na_action" to "fill".')

    K = len(shares)
    have_mean_only = np.isfinite(shares) & ~np.isfinite(sds)
    have_sd_only = np.isfinite(sds) & ~np.isfinite(shares)
    have_both = np.isfinite(shares) & np.isfinite(sds)

    if np.all(have_both):
        sample = rdirg(n, shares, sds)
    elif np.all(have_mean_only):
        sample = rdir_maxent(n, shares, **kwargs)
    else:
        sample = np.zeros((n, K))
        if np.sum(have_both) > 0:
            sample[:, have_both] = rbeta3(n, shares[have_both], sds[have_both], max_iter=max_iter)
        if np.sum(have_mean_only) > 0:
            alpha2 = shares[have_mean_only] / np.sum(shares[have_mean_only])
            sample_temp = rdir_maxent(n, alpha2, **kwargs)
            sample[:, have_mean_only] = sample_temp * (1 - sample.sum(axis=1, keepdims=True))
    return sample

def rbeta3(n, shares, sds, fix=True, max_iter=1E3):
    var = sds ** 2
    undef_comb = (shares * (1 - shares)) < var
    if not np.all(~undef_comb):
        if fix:
            var[undef_comb] = shares[undef_comb] ** 2
        else:
            raise ValueError('The beta distribution is not defined for the parameter combination you provided! sd must be smaller or equal sqrt(shares*(1-shares))')

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
            x[larger_one, i] = np.random.beta(alpha[i], beta[i], size=np.sum(larger_one))
        larger_one = x.sum(axis=1) > 1
        count += 1
        if count > max_iter:
            raise ValueError('max_iter is reached. the combinations of shares and sds you provided does allow to generate `n` random samples that are not larger than 1. Either increase max_iter, or change parameter combination.')
    return x