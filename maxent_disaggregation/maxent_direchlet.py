import numpy as np
from scipy.special import gamma, digamma, polygamma, psi
from scipy.stats import dirichlet
import nlopt


def dirichlet_entropy_derivative(gamma_par, shares):
    """
    Computes the derivative of the entropy of the Dirichlet distribution
    with respect to scaling parameter x, assuming alpha = x * shares.

    Parameters:
    - x: scalar multiplier
    - shares: list or numpy array of fixed Dirichlet parameters

    Returns:
    - derivative of entropy with respect to x
    """
    print("Entropy derative funtion being used!")
    alpha = gamma_par * np.array(shares)
    k = len(shares)
    sum_alpha = np.sum(alpha)

    term1 = psi(sum_alpha)
    term2 = polygamma(1, sum_alpha)  # trigamma

    derivative = 0
    for j in range(k):
        a_j = alpha[j]
        a_fix_j = shares[j]
        dH_dx_j = a_fix_j * (
            term1 + (sum_alpha - k) * term2 - psi(a_j) - (a_j - 1) * polygamma(1, a_j)
        )
        derivative += dH_dx_j

    return derivative


def dirichlet_entropy(gamma_par, shares):
    """
    Computes the entropy of a Dirichlet distribution.
    The entropy is calculated based on the given parameters `x` and `shares`,
    which are used to derive the concentration parameters `alpha` of the
    Dirichlet distribution.
    Parameters:
        gamma_par (float): The gamma scaling factor applied to the `shares` to compute the
                   concentration parameters `alpha`.
        shares (array-like): A vector of proportions that, when scaled by `gamma_par`,
                             define the concentration parameters `alpha` of
                             the Dirichlet distribution.
    Returns:
        float: The negative entropy of the Dirichlet distribution.
    """

    alpha = gamma_par * shares
    return -dirichlet.entropy(alpha)


def find_gamma_maxent2(
    shares,
    eval_f=dirichlet_entropy,
    x0=1,
    x0_n_tries=100,
    bounds=(0.001, 172),
    shares_lb=0,
    eval_grad_f=dirichlet_entropy_derivative,
    grad_based=False,
):
    if not np.isclose(np.sum(shares), 1):
        raise ValueError(
            f"Shares must sum to 1. But `sum(shares)` gives {np.sum(shares)}"
        )

    shares = shares[shares > shares_lb]
    shares = shares / np.sum(shares)

    lb, ub = bounds

    count = 0
    count2 = 0
    while not np.isfinite(eval_f(gamma_par=x0, shares=shares)):
        if count > x0_n_tries:
            if count2 == 1:
                raise ValueError(
                    "Error: Could not find an initial value x0 which is defined by eval_f and/or eval_grad_f. \n"
                    "You have different options:\n"
                    "1. increase the lower bound for the shares (under which all shares are excluded). E.g. shares_lb = 1E-3 \n"
                    "2. increase x0_n_tries (default: 100)\n"
                    "3. increase the parameter space with the bounds argument (e.g. bounds = c(1E-5, 1E4). Note: might take longer.\n"
                    "If none works, raise an issue on Github."
                )
                break

            shares = shares[shares > shares_lb]
            shares = shares / np.sum(shares)

            count = 0
            count2 = 1
        else:
            x0 = np.random.uniform(lb, ub)
            count += 1

    # Define the objective function for nlopt
    def objective(x, grad):
        if grad.size > 0:
            grad[:] = eval_grad_f(x, shares)
        return eval_f(x, shares)

    opt = nlopt.opt(nlopt.GN_DIRECT, 1)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_min_objective(objective)
    opt.set_xtol_rel(1.0e-4)
    opt.set_maxeval(1000)

    if grad_based:
        print("Using gradient-based optimization.")
        local_opt = nlopt.opt(nlopt.LD_MMA, 1)
        local_opt.set_xtol_rel(1.0e-4)
        opt.set_local_optimizer(local_opt)

    res = opt.optimize([x0])
    return res[0]


# def dirichlet_entropy_grad(x, shares):
#     """
#     Computes the gradient of the entropy of a Dirichlet distribution.
#     Parameters:
#         x (array-like): A vector of parameters for the Dirichlet distribution.
#         shares (array-like): A vector of shares corresponding to the Dirichlet parameters.
#     Returns:
#         float: The negative gradient of the Dirichlet entropy.
#     Notes:
#         - This function is currently not used. In the R-package it was found to produce wrong results.
#         - The function uses special functions such as the gamma function, digamma function,
#           and polygamma function to compute the gradient.
#         - The computation involves terms related to the Dirichlet distribution's entropy
#           and its derivatives.
#
#     """
#
#     e1 = shares * x
#     e2 = np.sum(e1)
#     term1 = (1 - np.prod(gamma(e1)) / (beta2(e1) * gamma(e2))) * digamma(e2)
#     term2 = (e2 - len(e1)) * polygamma(1, e2)
#     term3 = np.sum(shares * ((e1 - 1) * polygamma(1, e1) + digamma(e1)))
#     return -((term1 + term2) * np.sum(shares) - term3)
#
# def beta2(alpha):
#     """
#     Computes the multivariate beta function for a given array of alpha parameters.
#     The multivariate beta function is defined as:
#         B(alpha) = (Gamma(alpha_1) * Gamma(alpha_2) * ... * Gamma(alpha_n)) / Gamma(sum(alpha))
#     where `Gamma` is the gamma function.
#     Parameters:
#         alpha (array-like): A sequence of positive values representing the parameters of the beta function.
#     Returns:
#         float: The computed value of the multivariate beta function.
#     Raises:
#         ValueError: If any value in `alpha` is non-positive.
#     """
#
#     return np.prod(gamma(alpha)) / gamma(np.sum(alpha))
