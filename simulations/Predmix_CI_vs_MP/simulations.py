import sys
import os

sys.path.append(os.path.relpath("../../"))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import multinomial
import matplotlib.cm as cm

import sys

from other_bounds import maurer_pontil_empbern_ci, audibert_empbern_ci
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs
from confseq.predmix import predmix_empbern_ci_seq

figures_location = os.path.relpath("figures/")

import random

N = 10000
alpha = 0.05
times = np.arange(10, N, step=10)

cs_list = [
    ConfseqToPlot(
        lambda x: audibert_empbern_ci(x, times=times, alpha=alpha),
        "EB-CI [AMS07]",
        "tab:red",
        "--",
    ),
    ConfseqToPlot(
        lambda x: maurer_pontil_empbern_ci(x, times=times, alpha=alpha),
        "EB-CI [MP09]",
        "tab:purple",
        "-.",
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_ci_seq(
            x, times=times, truncation=1 / 2, alpha=alpha, parallel=True
        ),
        "PrPl-EB-CI [Rmk 1]",
        "tab:blue",
        ":",
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
        data_generator_fn=lambda: np.random.beta(1, 1, N),
        dist_fn=lambda x: beta.pdf(x, 1, 1),
        mean=0.5,
        name="Beta_1,_1_",
        discrete=False,
        title="$X_i \sim$ Beta(1, 1)",
    ),
    # DataGeneratingProcess(
    #     data_generator_fn = lambda: np.random.beta(10, 30, N),
    #     dist_fn = lambda x: beta.pdf(x, 10, 30),
    #     mean = 1/4,
    #     name = 'Beta(10, 30)',
    #     discrete = False,
    #     title = '$X_i \sim$ Beta(10, 30)'
    # )
]

plot_CSs(
    dgp_list,
    cs_list,
    times=times,
    time_uniform=False,
    nsim=5,
    log_scale=True,
    folder=figures_location,
)

print("Done!")
