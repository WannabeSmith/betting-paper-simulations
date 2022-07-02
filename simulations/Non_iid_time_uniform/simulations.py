import matplotlib.cm as cm
from scipy.stats import multinomial
from scipy.stats import beta
from scipy.stats import binom
import matplotlib.pyplot as plt

# sys.path.append(os.path.relpath("../../"))
import numpy as np
import sys
import os

from confseq.cs_plots import ConfseqToPlot, DataGeneratingProcess, plot_CSs
from confseq.predmix import predmix_hoeffding_cs, predmix_empbern_cs
from confseq.betting import betting_cs, dKelly_cs


figures_location = os.path.relpath("figures/")

N = 10000
alpha = 0.05

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
            x, breaks=1000, alpha=alpha, parallel=True, running_intersection=True
        ),
        r"Hedged [Thm 4]",
        "tab:green",
        "-",
    ),
    ConfseqToPlot(
        lambda x: dKelly_cs(
            x, D=20, alpha=alpha, parallel=True, running_intersection=True
        ),
        r"hgKelly",
        "tab:purple",
        ":",
    ),
]

dgp_list = [
    DataGeneratingProcess(
        data_generator_fn=lambda: np.hstack(
            (
                np.random.beta(10, 10, int(N / 4000)),
                np.random.binomial(1, 0.5, N - int(N / 4000)),
            )
        ),
        dist_fn=lambda x: binom.pmf(x, 1, 0.5),
        mean=0.5,
        name="250_beta_rest_bernoulli",
        discrete=True,
        title="250 Beta(10, 10),\nrest Bernoulli(0.5)",
    ),
    DataGeneratingProcess(
        data_generator_fn=lambda: np.hstack(
            (
                np.random.beta(10, 10, int(N / 400)),
                np.random.binomial(1, 0.5, N - int(N / 400)),
            )
        ),
        dist_fn=lambda x: binom.pmf(x, 1, 0.5),
        mean=0.5,
        name="2500_beta_rest_bernoulli",
        discrete=True,
        title="2500 Beta(10, 10),\nrest Bernoulli(0.5)",
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
    include_density=False,
)

print("Done!")
