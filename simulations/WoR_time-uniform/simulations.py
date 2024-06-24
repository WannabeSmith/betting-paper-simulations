import sys
import os
import numpy as np
from scipy.stats import binom
import sys
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs
from confseq.betting import betting_cs, mu_t, dKelly_cs
from confseq.predmix import (
    predmix_empbern_cs,
    predmix_hoeffding_cs
)
import matplotlib.pyplot as plt

figures_location = os.path.relpath("figures/")


N = 10000
alpha = 0.05

x = np.append(np.zeros(int(N / 2)), np.ones(int(N / 2)))
pmf = lambda x: binom.pmf(x, 1, 0.5)


def data_generator():
    np.random.shuffle(x)
    return x


t = np.arange(1, N + 1)

alpha = 0.05

cs_list = [
    ConfseqToPlot(
        lambda x: predmix_hoeffding_cs(x, N=N, alpha=alpha, running_intersection=True),
        "H-WoR [WR20]",
        "tab:orange",
        "-.",
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_cs(x, N=N, alpha=alpha, running_intersection=True),
        "EB-WoR [WR20]",
        "tab:blue",
        "--",
    ),
    ConfseqToPlot(
        lambda x: betting_cs(
            x,
            alpha=alpha,
            N=N,
            parallel=True,
            running_intersection=True,
            trunc_scale=1 / 2,
        ),
        r"Hedged-WoR [Thm 4]",
        "tab:green",
        "-",
    ),
    ConfseqToPlot(
        lambda x: dKelly_cs(
            x,
            alpha=alpha,
            N=N,
            parallel=True,
            running_intersection=True
        ),
        r"hgKelly",
        "tab:purple",
        ":",
    )
]


def generate_data_generator(x):
    def d_g():
        np.random.shuffle(x)
        return x

    return d_g


x_binom05 = np.random.binomial(1, 0.51, N)
x_binom01 = np.random.binomial(1, 1 / 20, N)
x_beta11 = np.random.beta(1, 1, N)
x_beta1030 = np.random.beta(10, 30, N)

dgp_list = [
    DataGeneratingProcess(
        data_generator_fn=generate_data_generator(x_binom05),
        name="Bernoulli_0.5__WoR",
        title="Discrete 0/1 high variance",
        WoR=True
    ),
    # DataGeneratingProcess(
    #     data_generator_fn=generate_data_generator(x_binom01),
    #     name="Bernoulli_0.1__WoR",
    #     title="Discrete 0/1 low variance",
    #     WoR=True
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn=generate_data_generator(x_beta11),
    #     name="Beta_1,_1__WoR",
    #     title="Real-valued evenly spread",
    #     WoR=True
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn=generate_data_generator(x_beta1030),
    #     name="Beta_10,_30__WoR",
    #     title="Real-valued concentrated",
    #     WoR=True
    # ),
]

plot_CSs(
    dgp_list,
    cs_list,
    time_uniform=True,
    nsim=5,
    log_scale=True,
    display_start=10,
    folder=figures_location,
)

print("Done!")
