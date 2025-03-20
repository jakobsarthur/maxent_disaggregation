import numpy as np
from scipy.special import gamma, digamma, polygamma
import nlopt

def dirichlet_entropy_grad(x, shares):
    e1 = shares * x
    e2 = np.sum(e1)
    term1 = (1 - np.prod(gamma(e1)) / (beta2(e1) * gamma(e2))) * digamma(e2)
    term2 = (e2 - len(e1)) * polygamma(1, e2)
    term3 = np.sum(shares * ((e1 - 1) * polygamma(1, e1) + digamma(e1)))
    return -((term1 + term2) * np.sum(shares) - term3)

def beta2(alpha):
    return np.prod(gamma(alpha)) / gamma(np.sum(alpha))

def dirichlet_entropy(x, shares):
    alpha = x * shares
    K = len(alpha)
    psi = digamma(alpha)
    alpha0 = np.sum(alpha)
    return -(np.log(beta2(alpha)) + (alpha0 - K) * digamma(alpha0) - np.sum((alpha - 1) * psi))

def find_gamma_maxent2(shares, eval_f=dirichlet_entropy, x0=1, x0_n_tries=100, bounds=(0.001, 172), shares_lb=0, local_opts=None, opts=None):
    if not np.isclose(np.sum(shares), 1):
        raise ValueError(f'Shares must sum to 1. But `sum(shares)` gives {np.sum(shares)}')
    
    shares = shares[shares > shares_lb]
    shares = shares / np.sum(shares)

    lb, ub = bounds

    count = 0
    count2 = 0
    while not np.isfinite(eval_f(x=x0, shares=shares)):
        if count > x0_n_tries:
            if count2 == 1:
                raise ValueError('Error: Could not find an initial value x0 which is defined by eval_f and/or eval_grad_f. \n'
                                 'You have different options:\n'
                                 '1. increase the lower bound for the shares (under which all shares are excluded). E.g. shares_lb = 1E-3 \n'
                                 '2. increase x0_n_tries (default: 100)\n'
                                 '3. increase the parameter space with the bounds argument (e.g. bounds = c(1E-5, 1E4). Note: might take longer.\n'
                                 'If none works, raise an issue on Github.')
                break

            shares = shares[shares > shares_lb]
            shares = shares / np.sum(shares)

            count = 0
            count2 = 1
        else:
            x0 = np.random.uniform(lb, ub)
            count += 1

    def objective(x, grad):
        if grad.size > 0:
            grad[:] = dirichlet_entropy_grad(x, shares)
        return eval_f(x, shares)

    opt = nlopt.opt(nlopt.GN_DIRECT, 1)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(objective)
    opt.set_xtol_rel(1.0e-4)
    opt.set_maxeval(1000)

    if local_opts:
        local_opt = nlopt.opt(nlopt.LD_MMA, 1)
        local_opt.set_xtol_rel(1.0e-4)
        opt.set_local_optimizer(local_opt)

    res = opt.optimize([x0])
    return res