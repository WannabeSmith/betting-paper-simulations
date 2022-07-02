from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib as mpl
from matplotlib.collections import LineCollection

from confseq.betting import betting_mart
from confseq.betting_strategies import lambda_aKelly
from confseq.types import RealArray


def conf_dist(
    x: RealArray, lambdas_fn, alpha_list, m_breaks=1000
) -> Dict[float, Tuple[RealArray, RealArray]]:
    m_list = np.arange(1 / m_breaks, 1, step=1 / m_breaks)
    lambdas_list = [lambdas_fn(x, m) for m in m_list]

    martingale_list = [
        np.cumprod(1 + lambdas_list[i] * (x - m_list[i])) for i in range(len(m_list))
    ]

    martingale_matrix = np.vstack(martingale_list)

    cs_dict = {}

    for alpha in alpha_list:
        accept_mtx = martingale_matrix < 1 / alpha

        l = np.zeros(len(x))
        u = np.ones(len(x))
        for j in np.arange(0, len(x)):
            where_in_cs = np.where(accept_mtx[:, j])
            if len(where_in_cs[0]) == 0:
                l[j] = 0
                u[j] = 1
            else:
                l[j] = m_list[where_in_cs[0][0]]
                u[j] = m_list[where_in_cs[0][-1]]

        cs_dict[alpha] = (l, u)

    return cs_dict


N = 1000
alpha_breaks = 5000
x = np.random.beta(1, 1, N)
alpha_list = np.arange(5 / alpha_breaks, 0.5, step=1 / alpha_breaks)
cs_dict = conf_dist(x, lambda_aKelly, alpha_list=alpha_list)
t = np.arange(1, N + 1)

cmap = pl.cm.get_cmap("Spectral")
plt.figure(figsize=(8, 4), dpi=80)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 13

for alpha in alpha_list:
    lower_cs = cs_dict[alpha][0]
    upper_cs = cs_dict[alpha][1]
    plt.plot(t, lower_cs, color=cmap(2 * alpha), linewidth=2.5, alpha=0.7)
    plt.plot(t, upper_cs, color=cmap(2 * alpha), linewidth=2.5, alpha=0.7)

norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(
    sm,
    ticks=np.linspace(0, 0.5, 10),
    boundaries=np.arange(-0.05, 0.55, 0.1),
    label=r"$\alpha \in (0, 1/2)$",
)
plt.xlabel("Time t")
plt.ylabel("Confidence sequence")
plt.xscale("log")

plt.savefig("figures/conf_dist.pdf", bbox_inches="tight")
plt.show()
