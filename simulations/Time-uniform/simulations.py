import sys
import os
sys.path.append(os.path.relpath("../../"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import multinomial
import matplotlib.cm as cm

from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs
from confseq.predmix import predmix_empbern_cs, predmix_hoeffding_cs
from confseq.betting import betting_cs

figures_location = os.path.relpath('figures/')

N = 10000
alpha = 0.05

cs_list = [
    ConfseqToPlot(
        lambda x: predmix_hoeffding_cs(x, alpha=alpha,
                                       running_intersection=True),
        'PrPl-H [Prop 1]',
        'tab:orange',
        '-.'
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_cs(x, truncation=0.5, alpha=alpha,
                                     running_intersection=True),
        'PrPl-EB [Thm 2]',
        'tab:blue',
        '--'
    ),
    ConfseqToPlot(
        lambda x: betting_cs(x, breaks=1000, trunc_scale=1/2,
                             alpha=alpha, parallel=True,
                             running_intersection=True),
        r'Hedged [Thm 3]',
        'tab:green',
        '-'
    ),

]

dgp_list = [
    DataGeneratingProcess(
        data_generator_fn = lambda: np.random.binomial(1, 0.5, N),
        dist_fn = lambda x: binom.pmf(x, 1, 0.5),
        mean = 0.5,
        name = 'Bernoulli_0.5_',
        discrete = True,
        title = '$X_i \sim$ Bernoulli(1/2)'
    ),
    DataGeneratingProcess(
        data_generator_fn = lambda: np.random.binomial(1, 0.1, N),
        dist_fn = lambda x: binom.pmf(x, 1, 0.1),
        mean = 0.1,
        name = 'Bernoulli_0.1_',
        discrete = True,
        title = '$X_i \sim$ Bernoulli(1/10)'
    ),
    DataGeneratingProcess(
        data_generator_fn = lambda: np.random.beta(1, 1, N),
        dist_fn = lambda x: beta.pdf(x, 1, 1),
        mean = 0.5,
        name = 'Beta_1,_1_',
        discrete = False,
        title = '$X_i \sim$ Beta(1, 1)'
    ),
    DataGeneratingProcess(
        data_generator_fn = lambda: np.random.beta(10, 30, N),
        dist_fn = lambda x: beta.pdf(x, 10, 30),
        mean = 1/4,
        name = 'Beta_10,_30_',
        discrete = False,
        title = '$X_i \sim$ Beta(10, 30)'
    ),
]

plot_CSs(dgp_list, cs_list, time_uniform=True,
         display_start=10,
         nsim=nsim, log_scale=True, folder=figures_location)

print('Done!')
