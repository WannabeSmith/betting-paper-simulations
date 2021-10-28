import sys
import os
# sys.path.append(os.path.relpath("../../"))

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import multinomial
from scipy.optimize import minimize 
import matplotlib.cm as cm

from confseq.predmix import lambda_predmix_eb

N = 100000
times =\
    np.unique(
        np.round(
            np.logspace(1, np.log(N),
                        num=50, base=np.e))).astype(int)

alpha = 0.05

x = np.random.binomial(1, 0.5, N)
t = np.arange(1, len(x) + 1)[times-1]

lambdas = lambda_predmix_eb(x, truncation=math.inf,
                            alpha=alpha)

ms = [0.4, 0.49, 0.51, 0.6]
c = 0.5
theta = 0.5

fig, ax = plt.subplots(1, len(ms), figsize=(13, 3.5))
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 13
for i in range(len(ms)):
    m = ms[i]
    lambdas_positive = np.minimum(c / m, lambdas)
    lambdas_negative = np.minimum(c / (1-m), lambdas)
    positive_capital = np.cumprod(1 + lambdas_positive*(x-m))[times-1]
    negative_capital = np.cumprod(1 - lambdas_negative*(x-m))[times-1]
    
    max_capital = np.maximum(theta*positive_capital,
                             (1-theta)*negative_capital)
    sum_capital = theta*positive_capital +\
        (1-theta)*negative_capital

    ax[i].plot(t, positive_capital, linestyle='-',
               label=r'$\mathcal{K}_t^+(m)$')
    ax[i].plot(t, negative_capital, linestyle='--',
               label=r'$\mathcal{K}_t^-(m)$')
    ax[i].plot(t, max_capital, linestyle='-.',
               label=r'$\mathcal{K}_t^\pm(m)$ (max)')
    ax[i].plot(t, sum_capital, linestyle=':',
               label=r'$\mathcal{M}_t^\pm(m)$ (sum)')
    ax[i].set_yscale('log')
    
    ax[i].set_title(r'$m - \mu = ' + str(round(m-0.5, ndigits=2)) + r'$')
    ax[i].set_xscale('log')
    ax[i].set_xlabel('time $t$')

    locmaj = matplotlib.ticker.LogLocator(base=10,
                                          numticks=7) 
   
    ax[i].xaxis.set_major_locator(locmaj)
    locmin = matplotlib.ticker.LogLocator(base=10.0,
                                            subs=np.arange(0.1, 1, step=0.1),
                                            numticks=12)
    ax[i].xaxis.set_minor_locator(locmin)
    ax[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    ax[i].set_ylim((10e-14, 10e10))


plt.tight_layout()
ax[0].set_ylabel('Wealth (log-scale)')
ax[0].legend(loc='lower left', bbox_to_anchor=(0.95, -0.5), ncol=4)

plt.savefig('./figures/plots.pdf', bbox_inches='tight')
