from confseq.betting import betting_ci
import numpy as np
import time
import os, sys
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, "../.."))
from other_bounds import (
    ptl_l2_ci,
)


sample_sizes = np.arange(5, 250, step=10)
ptl_compute_times = [None] * len(sample_sizes)
betting_compute_times = [None] * len(sample_sizes)

for i in range(len(sample_sizes)):
    sample_size = sample_sizes[i]
    x = np.random.beta(1, 1, sample_size)
    start = time.time()
    ptl_l2_ci(x, alpha=0.05)
    end = time.time()
    compute_time = end - start
    print(
        "PTL: sample size of "
        + str(sample_size)
        + " took "
        + str(compute_time)
        + " seconds"
    )
    ptl_compute_times[i] = compute_time

    start = time.time()
    betting_ci(x, alpha=0.05)
    end = time.time()
    compute_time = end - start
    print(
        "Betting: sample size of "
        + str(sample_size)
        + " took "
        + str(compute_time)
        + " seconds"
    )
    betting_compute_times[i] = compute_time

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["font.size"] = 13
plt.plot(sample_sizes, ptl_compute_times, label=r"PTL-$\ell_2$", color="royalblue", linestyle="-")
plt.plot(sample_sizes, betting_compute_times, label=r"Betting", color="tomato", linestyle="-.")
plt.xlabel(r"Sample size $n$")
plt.ylabel(r"Computation time (seconds)")
plt.legend(loc="best")
plt.savefig('figures/compute_times.pdf')