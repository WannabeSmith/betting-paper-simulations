import numpy as np
from scipy.stats import binom
from scipy.stats import beta

from confseq.betting import betting_ci_seq, dKelly_cs
from confseq.misc import get_ci_seq
from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs

import sys
import os

sys.path.append(os.path.relpath("../.."))
from other_bounds import (
    hoeffding_ci,
    maurer_pontil_empbern_ci,
    ptl_l2_ci,
    predmix_empbern_ci_seq,
)

figures_location = os.path.relpath("figures/")


N = 200
alpha = 0.05
times = np.unique(np.round(np.logspace(1, np.log(N), num=10, base=np.e))).astype(int)


def gridKelly_times(x):
    l, u = dKelly_cs(x, D=10, alpha=alpha, running_intersection=True, parallel=True)
    return l[times - 1], u[times - 1]


cs_list = [
    ConfseqToPlot(
        lambda x: hoeffding_ci(x, times=times, alpha=alpha),
        "H-CI [H63]",
        "tab:orange",
        "-",
    ),
    ConfseqToPlot(
        lambda x: get_ci_seq(
            x, ci_fn=lambda y: ptl_l2_ci(y, alpha=alpha), times=times, parallel=True
        ),
        r"PTL-$\ell_2$-CI [PTL21]",
        "black",
        dashes=[6, 2, 1, 2, 1, 2],
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
    # DataGeneratingProcess(
    #     data_generator_fn=lambda: np.random.binomial(1, 0.5, N),
    #     dist_fn=lambda x: binom.pmf(x, 1, 0.5),
    #     mean=0.5,
    #     name="Bernoulli_0.5_",
    #     discrete=True,
    #     title="$X_i \sim$ Bernoulli(1/2)",
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn=lambda: np.random.binomial(1, 0.1, N),
    #     dist_fn=lambda x: binom.pmf(x, 1, 0.1),
    #     mean=0.1,
    #     name="Bernoulli_0.1_",
    #     discrete=True,
    #     title="$X_i \sim$ Bernoulli(1/10)",
    # ),
    # DataGeneratingProcess(
    #     data_generator_fn=lambda: np.random.beta(1, 1, N),
    #     dist_fn=lambda x: beta.pdf(x, 1, 1),
    #     mean=0.5,
    #     name="Beta_1,_1_",
    #     discrete=False,
    #     title="$X_i \sim$ Beta(1, 1)",
    # ),
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
    display_start=5,
    nsim=5,
    log_scale=True,
    folder=figures_location,
    legend_outside_plot=True,
    legend_on_last_only=True,
    legend_columns=3,
    # bbox_to_anchor=(-1.15, -0.7)
)

print("Done!")
