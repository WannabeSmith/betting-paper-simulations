import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import multinomial

import sys

figures_location = os.path.relpath('figures/')

from confseq.predmix import predmix_empbern_cs
from confseq.conjmix_bounded import conjmix_empbern_cs
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs

N = 10000
alpha = 0.05

cs_list = [
    ConfseqToPlot(
        lambda x: conjmix_empbern_cs(x, v_opt=60/4, alpha=alpha,
                                     running_intersection=True),
        'CM-EB [HRMS20]',
        'tab:red',
        '-'
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_cs(x, alpha=alpha,
                                     running_intersection=True),
        'PrPl-EB [Thm 2]',
        'tab:blue',
        '--'
    ),
]

dgp_list = [
    # DataGeneratingProcess(
    #     data_generator_fn = lambda: np.random.binomial(1, 0.5, N),
    #     dist_fn = lambda x: binom.pmf(x, 1, 0.5),
    #     mean = 0.5,
    #     name = 'Bernoulli(0.5)',
    #     discrete = True,
    #     title = '$X_i \sim$ Bernoulli(1/2)'
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn = lambda: np.random.binomial(1, 0.1, N),
    #     dist_fn = lambda x: binom.pmf(x, 1, 0.1),
    #     mean = 0.1,
    #     name = 'Bernoulli(0.1)',
    #     discrete = True,
    #     title = '$X_i \sim$ Bernoulli(1/10)'
    # ),
    DataGeneratingProcess(
        data_generator_fn = lambda: np.random.beta(1, 1, N),
        dist_fn = lambda x: beta.pdf(x, 1, 1),
        mean = 0.5,
        name = 'Beta_1,_1_',
        discrete = False,
        title = '$X_i \sim$ Beta(1, 1)'
    ),
    # DataGeneratingProcess(
    #     data_generator_fn = lambda: np.random.beta(10, 30, N),
    #     dist_fn = lambda x: beta.pdf(x, 10, 30),
    #     mean = 1/4,
    #     name = 'Beta(10, 30)',
    #     discrete = False,
    #     title = '$X_i \sim$ Beta(10, 30)'
    # ),
]

plot_CSs(dgp_list, cs_list, 
         time_uniform=True,
         display_start=10,
         nsim=5,
         log_scale=True,
         folder=figures_location)

print("Done!")
