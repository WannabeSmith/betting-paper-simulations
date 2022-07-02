import os
from typing import Sequence
import numpy as np
from scipy.stats import binom
from scipy.stats import beta

from confseq.betting import betting_cs, dKelly_cs
from confseq.predmix import predmix_hoeffding_cs, predmix_empbern_cs
from confseq.conjmix_bounded import conjmix_bernoulli_cs
from confseq.other_bounded import banco
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs

from bentkus_conf_seq.conc_ineq.bentkus import adaptive_bentkus_seq

figures_location = os.path.relpath('figures/')

N = 10000
alpha = 0.05


def a_bentkus(x: Sequence[float], alpha: float, eta: float, power: float):
    l, u, _, _ = adaptive_bentkus_seq(Y=x, delta=alpha, L=0, U=1, eta=eta, power=power)

    return l, u


cs_list = [
    ConfseqToPlot(
        lambda x: predmix_hoeffding_cs(x, alpha=alpha, running_intersection=True),
        "PrPl-H [Prop 1]",
        "tab:orange",
        "-.",
    ),
    ConfseqToPlot(
        lambda x: predmix_empbern_cs(
            x, truncation=0.5, alpha=alpha, running_intersection=True
        ),
        "PrPl-EB [Thm 2]",
        "tab:blue",
        "--",
    ),
    ConfseqToPlot(
        lambda x: betting_cs(
            x,
            breaks=1000,
            trunc_scale=1 / 2,
            alpha=alpha,
            parallel=True,
            running_intersection=True,
        ),
        "Hedged [Thm 3]",
        "tab:green",
        "-",
    ),
    ConfseqToPlot(
        lambda x: dKelly_cs(
            x,
            D=20,
            alpha=alpha,
            breaks=1000,
            running_intersection=True,
            parallel=True,
            theta=1 / 2,
        ),
        "hgKelly",
        "tab:purple",
        ":",
    ),
    ConfseqToPlot(
        lambda x: banco(x, alpha=alpha, running_intersection=True),
        "BANCO [JO19]",
        "grey",
        dashes=[6, 6],
    ),
    ConfseqToPlot(
        lambda x: a_bentkus(x, alpha=alpha, eta=1.1, power=1.1),
        "A-Bentkus [KZ21]",
        "black",
        dashes=[1, 2, 1, 6],
    ),
    ConfseqToPlot(
        lambda x: conjmix_bernoulli_cs(
            x,
            t_opt=20,
            alpha=alpha,
            breaks=1000,
            running_intersection=True,
            parallel=True,
        ),
        r"Bernoulli [HRMS20]",
        "tab:red",
        dashes=[6, 2, 1, 2, 1, 2, 1, 2],
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
]

plot_CSs(
    dgp_list,
    cs_list,
    time_uniform=True,
    display_start=10,
    nsim=5,
    log_scale=True,
    folder=figures_location,
    legend_on_last_only=True,
    legend_outside_plot=True,
    legend_columns=4,
    bbox_to_anchor=(-1.15, -0.7),
)

print("Done!")
