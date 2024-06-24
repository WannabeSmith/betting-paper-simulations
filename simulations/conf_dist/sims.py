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
        np.maximum.accumulate(np.cumprod(1 + lambdas_list[i] * (x - m_list[i]))) for i in range(len(m_list))
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

# cmap = pl.cm.get_cmap("Spectral")
plt.figure(figsize=(8, 4), dpi=80)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 13

# ax = plt.axes()

# N = 50
# x = np.arange(N)
# Here are many sets of y to plot vs x
# ys = [x + i for i in x]

# We need to set the plot limits, they will not autoscale
ax = plt.axes()
ax.set_xlim(1, N)
ax.set_ylim(0, 1)

# colors is sequence of rgba tuples
# linestyle is a string or dash tuple. Legal string values are
#          solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq)
#          where onoffseq is an even length tuple of on and off ink in points.
#          If linestyle is omitted, 'solid' is used
# See matplotlib.collections.LineCollection for more information

# Make a sequence of x,y pairs
# line_segments = LineCollection([list(zip(x, y)) for y in ys],
#                                linewidths=(0.5, 1, 1.5, 2),
#                                linestyles='solid')


colormap = plt.cm.PuBuGn

lower_segments = LineCollection(
    [list(zip(t, cs_dict[alpha][0])) for alpha in alpha_list],
    array=alpha_list,
    cmap=colormap,
    linewidths=(1),
)
upper_segments = LineCollection(
    [list(zip(t, cs_dict[alpha][1])) for alpha in alpha_list],
    array=alpha_list,
    cmap=colormap,
    linewidths=(1),
)
# linestyles='solid')
# line_segments.set_array(t)
ax.add_collection(lower_segments)
ax.add_collection(upper_segments)

ax.set_xscale("log")
ax.set_xlabel("Time t")
ax.set_ylabel("Confidence sequence")
# line_segments.set_array(x)
# ax.add_collection(line_segments)
fig = plt.gcf()
axcb = fig.colorbar(lower_segments)
axcb.set_label(r"$\alpha \in (0, 1/2)$")
plt.sci(lower_segments)  # This allows interactive changing of the colormap.
plt.savefig("figures/conf_dist.pdf", bbox_inches="tight")
plt.show()


# for alpha in alpha_list:
#     lower_cs = cs_dict[alpha][0]
#     upper_cs = cs_dict[alpha][1]
#     plt.plot(t, lower_cs, color=cmap(2 * alpha), linewidth=2.5, alpha=0.7)
#     plt.plot(t, upper_cs, color=cmap(2 * alpha), linewidth=2.5, alpha=0.7)

# norm = mpl.colors.Normalize(vmin=0, vmax=0.5)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# plt.colorbar(
#     sm,
#     ticks=np.linspace(0, 0.5, 10),
#     boundaries=np.arange(-0.05, 0.55, 0.1),
#     label=r"$\alpha \in (0, 1/2)$",
# )

