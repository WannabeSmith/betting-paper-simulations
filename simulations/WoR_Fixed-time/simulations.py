import os
import numpy as np
import matplotlib.pyplot as plt
from confseq.predmix import predmix_hoeffding_ci, predmix_empbern_ci
from confseq.betting import get_ci_seq, betting_ci_seq
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs

figures_location = os.path.relpath("figures/")
import random

N = 10000
alpha = 0.05
times = np.unique(np.round(np.logspace(1, np.log(N), num=50, base=np.e))).astype(int)

ci_list = [
    ConfseqToPlot(
        lambda x: get_ci_seq(
            x,
            ci_fn=lambda y: predmix_hoeffding_ci(
                y, alpha=alpha, N=N, running_intersection=True
            ),
            times=times,
            parallel=True,
        ),
        "H-WoR-CI [WR20]",
        "tab:orange",
        "-.",
    ),
    ConfseqToPlot(
        lambda x: get_ci_seq(
            x,
            ci_fn=lambda y: predmix_empbern_ci(
                y,
                alpha=alpha,
                N=N,
                running_intersection=True,
                truncation=1/2,
            ),
            times=times,
            parallel=True,
        ),
        "EB-WoR-CI [WR20]",
        "tab:blue",
        "--",
    ),
    ConfseqToPlot(
        lambda x: betting_ci_seq(
            x,
            N=N,
            times=times,
            alpha=alpha,
            parallel=True,
            breaks=1000,
            m_trunc=True,
            trunc_scale=3 / 4,
        ),
        "Hedged-WoR-CI [Rmk 4]",
        "tab:green",
        ":",
    ),
]


def generate_data_generator(x):
    def d_g():
        np.random.shuffle(x)
        return x

    return d_g


x_binom05 = np.random.binomial(1, 1 / 2, N)
x_binom01 = np.random.binomial(1, 1 / 10, N)
x_beta11 = np.random.beta(1, 1, N)
x_beta1030 = np.random.beta(10, 30, N)

dgp_list = [
    DataGeneratingProcess(
        data_generator_fn=generate_data_generator(x_binom05),
        WoR=True,
        name="Bernoulli_0.5__WoR",
        title="Discrete 0/1 high variance",
    ),
    DataGeneratingProcess(
        data_generator_fn=generate_data_generator(x_binom01),
        WoR=True,
        name="Bernoulli_0.1__WoR",
        title="Discrete 0/1 low variance",
    ),
    DataGeneratingProcess(
        data_generator_fn=generate_data_generator(x_beta11),
        WoR=True,
        name="Beta_1,_1__WoR",
        title="Real-valued evenly spread",
    ),
    DataGeneratingProcess(
        data_generator_fn=generate_data_generator(x_beta1030),
        WoR=True,
        name="Beta_10,_30__WoR",
        title="Real-valued concentrated",
    ),
]

plot_CSs(
    dgp_list,
    ci_list,
    time_uniform=False,
    times=times,
    nsim=5,
    log_scale=True,
    display_start=10,
    folder=figures_location,
)

print("Done!")
