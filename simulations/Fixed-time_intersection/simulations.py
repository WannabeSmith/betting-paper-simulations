import sys
import os
from confseq.betting import betting_ci, betting_cs

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom
from scipy.stats import beta
from scipy.stats import multinomial
import matplotlib.cm as cm

sys.path.append(os.path.relpath("../.."))
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs
from confseq.predmix import lambda_predmix_eb

figures_location = os.path.relpath("figures/")

import random

N = 1000
alpha = 0.05

cs_list = [
    ConfseqToPlot(
        lambda x: betting_cs(
            x,
            alpha=alpha,
            running_intersection=True,
            breaks=1000,
            convex_comb=False,
            lambdas_fns_positive=[
                lambda x, m: lambda_predmix_eb(x, alpha=alpha, fixed_n=100)
            ],
            m_trunc=True,
            trunc_scale=3 / 4,
        ),
        "Intersection",
        "tab:red",
        "-",
    ),
    ConfseqToPlot(
        lambda x: betting_cs(
            x,
            alpha=alpha,
            running_intersection=False,
            breaks=1000,
            convex_comb=False,
            lambdas_fns_positive=[
                lambda x, m: lambda_predmix_eb(x, alpha=alpha, fixed_n=100)
            ],
            m_trunc=True,
            trunc_scale=3 / 4,
        ),
        "No intersection",
        "tab:green",
        "--",
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
    # DataGeneratingProcess(
    #     data_generator_fn=lambda: np.random.beta(1, 1, N),
    #     dist_fn=lambda x: beta.pdf(x, 1, 1),
    #     mean=0.5,
    #     name="Beta_1,_1_",
    #     discrete=False,
    #     title="$X_i \sim$ Beta(1, 1)",
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn=lambda: np.random.beta(10, 30, N),
    #     dist_fn=lambda x: beta.pdf(x, 10, 30),
    #     mean=1 / 4,
    #     name="Beta_10,_30_",
    #     discrete=False,
    #     title="$X_i \sim$ Beta(10, 30)",
    # ),
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
    time_uniform=True,
    display_start=10,
    nsim=1,
    log_scale=True,
    folder=figures_location,
)

plt.show()

print("Done!")
