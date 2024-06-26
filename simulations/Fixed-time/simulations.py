import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import multinomial
import matplotlib.cm as cm

sys.path.append(os.path.relpath("../.."))
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs
from other_bounds import hoeffding_ci, maurer_pontil_empbern_ci
from confseq.predmix import predmix_empbern_ci_seq
from confseq.betting import betting_ci_seq

figures_location = os.path.relpath("figures/")

import random

N = 10000
alpha = 0.05
times = np.unique(np.round(np.logspace(1, np.log(N), num=50, base=np.e))).astype(int)

cs_list = [
    ConfseqToPlot(
        lambda x: hoeffding_ci(x, times=times, alpha=alpha),
        "H-CI [H63]",
        "tab:orange",
        "-.",
    ),
    ConfseqToPlot(
        lambda x: maurer_pontil_empbern_ci(x, times=times, alpha=alpha),
        "EB-CI [MP09]",
        "tab:purple",
        "-",
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_ci_seq(
            x, times=times, truncation=1 / 2, alpha=alpha, parallel=True
        ),
        "PrPl-EB-CI [Rmk 1]",
        "tab:blue",
        "--",
    ),
    ConfseqToPlot(
        lambda x: betting_ci_seq(
            x, times=times, alpha=alpha, parallel=True, m_trunc=True, trunc_scale=3 / 4
        ),
        "Hedged-CI [Rmk 3]",
        "tab:green",
        ":",
    ),
]

dgp_list = [
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.binomial(1, 0.5, N),
        dist_fn=lambda x: binom.pmf(x, 1, 0.5),
        mean=0.5,
        name="Bernoulli_0.5_",
        discrete=True,
        title="$X_i \sim$ Bernoulli(1/2)",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.binomial(1, 0.1, N),
        dist_fn=lambda x: binom.pmf(x, 1, 0.1),
        mean=0.1,
        name="Bernoulli_0.1_",
        discrete=True,
        title="$X_i \sim$ Bernoulli(1/10)",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.beta(1, 1, N),
        dist_fn=lambda x: beta.pdf(x, 1, 1),
        mean=0.5,
        name="Beta_1,_1_",
        discrete=False,
        title="$X_i \sim$ Beta(1, 1)",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.random.beta(10, 30, N),
        dist_fn=lambda x: beta.pdf(x, 10, 30),
        mean=1 / 4,
        name="Beta_10,_30_",
        discrete=False,
        title="$X_i \sim$ Beta(10, 30)",
    ),
    # DataGeneratingProcess(
    #     data_generator_fn =\
    #         lambda: np.append(np.random.binomial(1, 1/2, N-int(N/2)),
    #                           np.random.beta(1, 1, int(N/2))),
    #     dist_fn = lambda x: beta.pdf(x, 1, 1),
    #     mean = 1/2,
    #     name = 'Non_iid',
    #     discrete = False,
    #     title = 'Beta + Bernoulli'
    # )
]

plot_CSs(
    dgp_list,
    cs_list,
    times=times,
    time_uniform=False,
    display_start=10,
    nsim=5,
    log_scale=True,
    folder=figures_location,
)

print("Done!")
