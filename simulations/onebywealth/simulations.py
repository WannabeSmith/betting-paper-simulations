from confseq.betting import betting_mart
from confseq.betting_strategies import (
    lambda_aKelly,
    lambda_LBOW,
    lambda_Kelly,
)
import numpy as np
import matplotlib.pyplot as plt

times = [25, 100, 250]
breaks = 1000
x = np.random.binomial(1, 0.5, times[-1])

m_list = np.arange(1 / breaks, 1, step=1 / breaks)

betting_strategies = {
    "GRAPA": lambda_Kelly,
    "aGRAPA": lambda_aKelly,
    "LBOW": lambda_LBOW,
}

linestyles = {
    "GRAPA": "-",
    "aGRAPA": "-.",
    "LBOW": ":",
}



strategy_wealth_dict = {}

for strat_name, strat_fn in betting_strategies.items():

    wealth_time_dict = {time: np.array([None] * len(m_list)) for time in times}
    for idx, m in enumerate(m_list):
        mart = betting_mart(x, m, lambdas_fn_positive=strat_fn, theta=1)
        for time in times:
            wealth_time_dict[time][idx] = mart[time - 1]

    strategy_wealth_dict[strat_name] = wealth_time_dict

plt.figure(figsize=(8, 4), dpi=80)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 11
fig, axs = plt.subplots(1, len(times), figsize=(12, 3))

for strat_name in betting_strategies.keys():
    wealth_time_dict = strategy_wealth_dict[strat_name]

    for idx, time in enumerate(times):
        axs[idx].plot(
            m_list,
            1 / wealth_time_dict[time],
            label=strat_name,
            linestyle=linestyles[strat_name],
        )
        axs[idx].set_xlabel("m")
        axs[idx].set_title(r"Time " + "$t = " + str(time) + "$.")

axs[0].set_ylabel(r"1/$\mathcal{K}_t(m)$")
axs[-1].legend(loc="upper right")
plt.savefig("figures/onebywealth.pdf", bbox_inches="tight")
