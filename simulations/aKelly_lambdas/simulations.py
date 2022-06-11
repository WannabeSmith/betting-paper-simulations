import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.relpath("../../"))
from other_bounds import lambda_aKelly_oracle

from confseq.betting_strategies import lambda_aKelly

figures_location = os.path.relpath('figures/')

N = 100000

ms = 0.5 + np.array([-0.2, -0.05, 0, 0.05, 0.2])


fig, ax = plt.subplots(1, 2, figsize=(10, 4))
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 11

x = np.random.binomial(1, 0.5, N)
for m in ms:
    t = np.arange(10, N+1)
    lambdas = lambda_aKelly(x, m)[t-1]
    lambdas_oracle = lambda_aKelly_oracle(x, m, mu = 0.5,
                                        var=1/4)[t-1]
    ax[0].set_title(r'$X_i \sim$ Bernoulli($1/2$)')
    line, = ax[0].plot(t, lambdas, 
                       label=r'$m = $' + str(m),
                       alpha=0.8)
    ax[0].plot(t, lambdas_oracle,
               color=line.get_color(), 
               linestyle=':')

x = np.random.beta(1, 1, N)
for m in ms:
    t = np.arange(10, N+1)
    lambdas = lambda_aKelly(x, m)[t-1]
    lambdas_oracle = lambda_aKelly_oracle(x, m, mu = 0.5,
                                        var=1/12)[t-1]
    ax[1].set_title(r'$X_i \sim$ Beta(1, 1)')
    line, = ax[1].plot(t, lambdas, 
                       label=r'$m = $' + str(m),
                       alpha=0.8)
    ax[1].plot(t, lambdas_oracle,
               color=line.get_color(), 
               linestyle=':')


ax[0].set_xscale('log')
ax[0].set_ylabel(r'$\lambda_t^{\mathregular{aGRAPA}}$')
ax[0].set_xlabel(r'$t$')
ax[0].set_ylim(-2, 2)
ax[1].set_xscale('log')
ax[1].set_xlabel(r'$t$')
ax[1].set_ylim(-2, 2)

#ax[0].legend(loc='upper left')
plt.tight_layout()
ax[0].legend(loc='lower left', bbox_to_anchor=(0.2, -0.4), ncol=5)
plt.savefig(figures_location + '/lambdas.pdf', bbox_inches='tight')
