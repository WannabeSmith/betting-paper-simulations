from bentkus_conf_seq.conc_ineq.bentkus import bentkus
import numpy as np
from confseq.betting import get_ci_seq
from confseq.predmix import predmix_empbern_cs
from small_sample_mean_bounds.bound import b_alpha_l2norm, b_alpha_linear
from typing import Sequence
from gaffke import gaffke


def hoeffding_ci(x, times, alpha=0.05):
    x = np.array(x)
    S_t = np.cumsum(x)
    t = np.arange(1, len(x) + 1)
    mu_hat_t = S_t / t

    margin = np.sqrt(np.log(2 / alpha) / (2 * t))
    l, u = mu_hat_t - margin, mu_hat_t + margin
    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    return l[times - 1], u[times - 1]


def predmix_empbern_ci(x, alpha, truncation=1 / 2, parallel=True):
    x = np.array(x)

    l, u = predmix_empbern_cs(
        x, alpha=alpha, truncation=truncation, running_intersection=True, fixed_n=len(x)
    )

    return l[-1], u[-1]


def predmix_empbern_ci_seq(x, alpha, times, truncation=1 / 2, parallel=False):
    ci_fn = lambda x: predmix_empbern_ci(x, alpha=alpha, truncation=truncation)
    return get_ci_seq(x, ci_fn, times=times, parallel=parallel)


def bennett_ci(x, times, variance, alpha=0.05):
    x = np.array(x)
    S_t = np.cumsum(x)
    t = np.arange(1, len(x) + 1)
    mu_hat_t = S_t / t

    margin = np.sqrt(2 * variance * np.log(2 / alpha) / t) + np.log(2 / alpha) / (3 * t)

    l, u = mu_hat_t - margin, mu_hat_t + margin
    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    return l[times - 1], u[times - 1]


def bentkus_ci(x, times, variance, alpha=0.05):
    x = np.array(x)
    S_t = np.cumsum(x)
    t = np.arange(1, len(x) + 1)
    mu_hat_t = (S_t / t)[times - 1]

    margin = np.ones(len(times))
    for i in range(len(times)):
        time = times[i]
        margin[i] = (1 / time) * bentkus(
            n=time, delta=alpha / 2, A=np.sqrt(variance), B=1
        )

    l, u = mu_hat_t - margin, mu_hat_t + margin
    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    return l, u


def anderson_ci(x, alpha):
    n = len(x)
    i = np.arange(1, n + 1)
    u_DKW = np.maximum(0, i / n - np.sqrt(np.log(2 / alpha) / (2 * n)))

    zu_i = np.sort(x)
    zu_iplus1 = np.append(zu_i, 1)[1:]
    zl_i = np.flip(1 - zu_i)
    zl_iplus1 = np.append(zl_i, 1)[1:]
    u = 1 - np.sum(u_DKW * (zu_iplus1 - zu_i))
    l = np.sum(u_DKW * (zl_iplus1 - zl_i))
    return l, u


def anderson_ci_seq(x, alpha, times, parallel=False):
    def ci_fn(x):
        return anderson_ci(x, alpha=alpha)

    return get_ci_seq(x, ci_fn, times=times, parallel=parallel)


def ptl_l2_ci(x, alpha):
    upper = b_alpha_l2norm(z=x, alpha=alpha / 2, upper=1)
    lower = b_alpha_l2norm(z=1 - x, alpha=alpha / 2, upper=1)

    return 1 - lower, upper


def ptl_linear_ci(x, alpha):
    upper = b_alpha_linear(z=x, alpha=alpha / 2, upper=1)
    lower = b_alpha_linear(z=1 - x, alpha=alpha / 2, upper=1)

    return 1 - lower, upper


def maurer_pontil_empbern_ci(x, times, alpha=0.05):
    x = np.array(x)
    S_t = np.cumsum(x)
    t = np.arange(1, len(x) + 1)
    mu_hat_t = S_t / t

    V_t = np.cumsum(np.power(x, 2)) / t - np.power(S_t / t, 2)
    margin = np.sqrt(2 * V_t * np.log(4 / alpha) / t) + 7 * np.log(4 / alpha) / (
        3 * (t - 1)
    )

    l, u = mu_hat_t - margin, mu_hat_t + margin

    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    return l[times - 1], u[times - 1]


def audibert_empbern_ci(x, times, alpha=0.05):
    x = np.array(x)
    S_t = np.cumsum(x)
    t = np.arange(1, len(x) + 1)
    mu_hat_t = S_t / t

    V_t = np.cumsum(np.power(x, 2)) / t - np.power(S_t / t, 2)
    margin = np.sqrt(2 * V_t * np.log(3 / alpha) / t) + 3 * np.log(3 / alpha) / t

    l, u = mu_hat_t - margin, mu_hat_t + margin

    l = np.maximum(l, 0)
    u = np.minimum(u, 1)

    return l[times - 1], u[times - 1]


class LBOW_Bettor:
    def __init__(self, x, WoR=False, N=None):
        self.x = x
        self.WoR = WoR
        self.N = N
        t = np.arange(1, len(x) + 1)

        # Get mu_hat estimates
        mu_hat_t = (1 / 2 + np.cumsum(x)) / (t + 1)
        self.mu_hat_tminus1 = np.append(1 / 4, mu_hat_t[0 : (len(mu_hat_t) - 1)])

        sigma2_hat_t = (1 / 4 + np.cumsum(np.power(x, 2))) / (t + 1) - np.power(
            mu_hat_t, 2
        )

        self.sigma2_hat_tminus1 = np.append(
            1 / 4, sigma2_hat_t[0 : (len(sigma2_hat_t) - 1)]
        )

    def get_bet(self, against, idx):
        # Get bet using the LBOW scheme
        m_star = against if self.mu_hat_tminus1[idx] - against >= 0 else 1 - against
        bet = (self.mu_hat_tminus1[idx] - against) / (
            m_star * np.abs(self.mu_hat_tminus1[idx] - against)
            + self.sigma2_hat_tminus1[idx]
            + np.power(self.mu_hat_tminus1[idx] - against, 2)
        )
        return bet

    def get_l_bet(self, l_bdry, idx):
        # Get lower bet pre-truncation
        if self.WoR:
            against = (self.N * l_bdry - np.sum(self.x[0:idx])) / (self.N - idx)
        else:
            against = l_bdry
        return self.get_bet(against=against, idx=idx)

    def get_u_bet(self, u_bdry, idx):
        # Get upper bet pre-truncation
        if self.WoR:
            against = (self.N * u_bdry - np.sum(self.x[0:idx])) / (self.N - idx)
        else:
            against = u_bdry
        return self.get_bet(against=against, idx=idx)


def conbo_cs(
    x,
    Bettor=LBOW_Bettor,
    alpha=0.05,
    WoR=False,
    N=None,
    theta=1 / 2,
    breaks=1000,
    running_intersection=False,
    trunc_scale=1 / 2,
):
    n = len(x)
    t = np.arange(1, n + 1)
    l = np.zeros(n)
    u = np.ones(n)

    l_bdry = 0
    u_bdry = 1

    capital_vector_positive = np.ones(breaks + 1)
    capital_vector_negative = np.ones(breaks + 1)
    possible_m = np.arange(0, 1 + 1 / breaks, step=1 / breaks)

    bettor = Bettor(x)

    if WoR:
        assert N is not None
        S_tminus1 = np.append(0, np.cumsum(x)[0 : (len(x) - 1)])

    # For each time
    for i in range(n):
        if WoR:
            m_t = (N * possible_m - S_tminus1[i]) / (N - t[i] + 1)
        else:
            m_t = possible_m

        l_bet = bettor.get_l_bet(l_bdry, i)
        l_bet = np.maximum(l_bet, 0)
        l_bet = np.minimum(l_bet, trunc_scale / m_t)

        u_bet = bettor.get_u_bet(u_bdry, i)
        u_bet = np.abs(np.minimum(u_bet, 0))
        u_bet = np.minimum(u_bet, trunc_scale / (1 - m_t))

        capital_vector_positive *= 1 + l_bet * (x[i] - m_t)
        capital_vector_negative *= 1 - u_bet * (x[i] - m_t)
        capital_vector = (
            theta * capital_vector_positive + (1 - theta) * capital_vector_negative
        )
        capital_vector[np.logical_or(m_t < 0, m_t > 1)] = math.inf
        assert all(capital_vector >= 0)

        where_not_reject = np.where(capital_vector < 1 / alpha)[0]
        if len(where_not_reject) is 0 and i is not 0:
            l[i] = l[i - 1]
            u[i] = u[i - 1]
        elif len(where_not_reject) is 0 and i is 0:
            l[i] = 0
            u[i] = 1
        else:
            # Need to take superset
            l[i] = possible_m[where_not_reject[0]] - 1 / breaks
            u[i] = possible_m[where_not_reject[-1]] + 1 / breaks
            l_bdry = max(l_bdry, l[i])
            u_bdry = min(u_bdry, u[i])

    if WoR:
        # Intersect with logical cs
        l_logical, u_logical = logical_cs(x, N)
        l = np.maximum(l, l_logical)
        u = np.minimum(u, u_logical)
    else:
        l = np.maximum(0, l)
        u = np.minimum(1, u)

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    return l, u


def lambda_COLT18_ONS(x, m, c=1 / 2):
    lambdas = np.zeros(len(x))
    lambdas[0] = 0
    A = 1
    for i in np.arange(1, len(lambdas)):
        g = x[i - 1] - m
        z = -g / (1 + g * lambdas[i - 1])
        A = A + np.power(z, 2)
        lambdas[i] = lambdas[i - 1] - (2 / (2 - np.log(3))) * z / A
        lambdas[i] = np.minimum(c / m, np.maximum(-c / (1 - m), lambdas[i]))
    return lambdas


def banco(x: Sequence[float], alpha: float, running_intersection=False):
    """Concentration inequality resulting from BANCO (Jun & Orabona (2019))"""

    # Tighest possible upper bound on the SD for [0, 1]-bounded random variables without further information
    sigma_1D = 1 / 2

    t = np.arange(1, len(x) + 1)

    margin = sigma_1D * np.sqrt(
        2
        * np.log(
            np.power(6 * np.pi * np.sqrt(np.e) / alpha, 3 / 2)
            * (np.power(np.log(np.sqrt(t)), 2) + 1)
        )
        / t
    )

    mu_hat_t = np.cumsum(x) / t
    l, u = mu_hat_t - margin, mu_hat_t + margin

    if running_intersection:
        l = np.maximum.accumulate(l)
        u = np.minimum.accumulate(u)

    l = np.maximum(0, l)
    u = np.minimum(1, u)

    return l, u


def lambda_aKelly_oracle(x, m, mu, var, WoR=False, N=None, trunc_scale=1):
    if WoR:
        m_t = mu_t(x, m, N)
    else:
        m_t = np.repeat(m, len(x))
    lambdas = (mu - m_t) / (var + np.power(mu - m_t, 2))

    lambdas = np.maximum(-trunc_scale / (1 - m_t), lambdas)
    lambdas = np.minimum(trunc_scale / m_t, lambdas)

    return lambdas


def gaffke_ci(x, alpha, n_uprime=1000):
    gaffke_ub = gaffke(x=x, delta=alpha / 2, nUPrimes=n_uprime)
    gaffke_lb = 1 - gaffke(x=1 - x, delta=alpha / 2, nUPrimes=n_uprime)

    return gaffke_lb, gaffke_ub


def gaffke_ci_seq(x, alpha, n_uprime, times, parallel=False):
    def ci_fn(x):
        return gaffke_ci(x, alpha=alpha)

    return get_ci_seq(x, ci_fn, times=times, parallel=parallel)
