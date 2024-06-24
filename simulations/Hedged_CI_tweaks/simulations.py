import sys
import os
import random
import matplotlib.cm as cm
from scipy.stats import multinomial
from scipy.stats import beta
from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs
from confseq.predmix import predmix_empbern_ci_seq
from confseq.betting import betting_ci_seq

sys.path.append(os.path.relpath("../../"))
figures_location = os.path.relpath('figures/')

N = 10000
alpha = 0.05
times =\
    np.unique(
        np.round(
            np.logspace(1, np.log(N),
                        num=50, base=np.e))).astype(int)

hoeffding_lambda =\
    lambda x, m: np.sqrt(8*np.log(2/alpha) / len(x))

cs_list = [
    ConfseqToPlot(
        lambda x:
        betting_ci_seq(x, times=times,
                       alpha=alpha,
                       lambdas_fns_positive=hoeffding_lambda,
                       parallel=True,
                       running_intersection=False,
                       m_trunc=False,
                       trunc_scale=3/4),
        r'No tweaks',
        'tab:red',
        '-'
    ),

    ConfseqToPlot(
        lambda x:
            betting_ci_seq(x, times=times,
                           alpha=alpha,
                           parallel=True,
                           running_intersection=False,
                           m_trunc=False,
                           trunc_scale=3/4),
        r'PrPl',
        'tab:orange',
        '--'
    ),
    ConfseqToPlot(
        lambda x: betting_ci_seq(x, times=times,
                                 alpha=alpha,
                                 parallel=True,
                                 running_intersection=False,
                                 m_trunc=True,
                                 trunc_scale=3/4),
        r'PrPl, $m$',
        'tab:blue',
        '-.'
    ),
    ConfseqToPlot(
        lambda x: betting_ci_seq(x, times=times,
                                 alpha=alpha,
                                 parallel=True,
                                 running_intersection=True,
                                 m_trunc=True,
                                 trunc_scale=3/4),
        r'PrPl, $m$, $\cap_{i=1}^n$',
        'tab:green',
        ':'
    ),
]

dgp_list = [
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.binomial(1, 0.5, N),
        dist_fn=lambda x: binom.pmf(x, 1, 0.5),
        mean=0.5,
        name='Bernoulli_0.5_',
        discrete=True,
        title='$X_i \sim$ Bernoulli(1/2)'
    ),
    # DataGeneratingProcess(
    #     data_generator_fn = lambda: np.random.binomial(1, 0.1, N),
    #     dist_fn = lambda x: binom.pmf(x, 1, 0.1),
    #     mean = 0.1,
    #     name = 'Bernoulli_0.1_',
    #     discrete = True,
    #     title = '$X_i \sim$ Bernoulli(1/10)'
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn = lambda: np.random.beta(1, 1, N),
    #     dist_fn = lambda x: beta.pdf(x, 1, 1),
    #     mean = 0.5,
    #     name = 'Beta_1,_1_',
    #     discrete = False,
    #     title = '$X_i \sim$ Beta(1, 1)'
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn=lambda: np.random.beta(10, 30, N),
    #     dist_fn=lambda x: beta.pdf(x, 10, 30),
    #     mean=1/4,
    #     name='Beta_10,_30_',
    #     discrete=False,
    #     title='$X_i \sim$ Beta(10, 30)'
    # )
]

plot_CSs(dgp_list, cs_list, times=times,
         time_uniform=False, display_start=10,
         nsim=5, log_scale=True, folder=figures_location)

print('Done!')

