import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
import os
from time import time

# sys.path.append(os.path.relpath("../../"))

from confseq.betting_strategies import lambda_Kelly, lambda_aKelly, lambda_LBOW
from confseq.betting import betting_mart

truncation = 0.5

wealth_processes = {
    "GRAPA": lambda x, m: betting_mart(
        x, m, lambdas_fn_positive=lambda_Kelly, theta=1, trunc_scale=truncation
    ),
    "aGRAPA": lambda x, m: betting_mart(
        x, m, lambdas_fn_positive=lambda_aKelly, theta=1, trunc_scale=truncation
    ),
    "LBOW": lambda x, m: betting_mart(
        x, m, lambdas_fn_positive=lambda_LBOW, theta=1, trunc_scale=truncation
    ),
    "gKelly": lambda x, m: np.mean(
        np.vstack(
            [np.cumprod(1 + l * (x / m - 1)) for l in np.arange(-0.9, 1, step=0.1)]
        ),
        axis=0,
    ),
}
linestyles = {"GRAPA": "-", "aGRAPA": "--", "LBOW": "-.", "gKelly": ":"}
computation_times = {}

N = 500
start_time = 10
t = np.arange(start_time, N + 1)
data_generator_fn = lambda: np.random.beta(10, 10, N - start_time + 1)

num_repeats = 200

m_list = [0.5, 0.51, 0.55]

fig, axs = plt.subplots(1, len(m_list), figsize=(12, 3.5))
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 13
for i in range(len(m_list)):
    m = m_list[i]
    for wealth_name in wealth_processes:
        wealth_proc = wealth_processes[wealth_name]
        avg_capital = np.zeros(N - start_time + 1)
        avg_comp_time = 0
        for j in range(num_repeats):
            x = data_generator_fn()
            init_timer = time()
            end_timer = time()
            avg_comp_time += (end_timer - init_timer) / num_repeats
            avg_capital = avg_capital + wealth_proc(x, m) / num_repeats

        computation_times[wealth_name] = avg_comp_time
        axs[i].plot(
            t, avg_capital, label=wealth_name, linestyle=linestyles[wealth_name]
        )
        axs[i].set_xscale("log")
        axs[i].set_yscale("log")
        axs[i].set_xlabel(r"time $t$")
        axs[i].set_ylim(1e-1, 1e4)
        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=6)
        axs[i].xaxis.set_major_locator(locmaj)
        locmin = matplotlib.ticker.LogLocator(
            base=10.0, subs=np.arange(0.1, 1, step=0.1), numticks=12
        )
        axs[i].xaxis.set_minor_locator(locmin)
        axs[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[i].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    print(computation_times)

axs[0].set_ylabel("Wealth (log-scale)")
axs[0].set_title(r"$m = \mu$")
axs[1].set_title(r"$m > \mu$")
axs[2].set_title(r"$m \gg \mu$")
axs[2].legend(loc="best")

plt.tight_layout()
plt.savefig("figures/game_theoretic_wealths.pdf")
