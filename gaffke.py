"""
A Python translation of Philip Thomas' C++ implementation of Gaffke's bound
"""
import numpy as np
from numpy.typing import NDArray
import math

RealArray = NDArray[np.float_]

def m(z : RealArray, u: RealArray):
    """
    z are order statistics (assumed sorted)
    """
    n = len(z)

    result = 1.0

    for i in range(n-1):
        result -= (z[i+1] - z[i]) * u[i]

    result -= (1.0 - z[n-1]) * u[n-1]

    return result


def gaffke(x: RealArray, delta: float, nUPrimes : 1000):

    n = len(x)

    UPrimes = [sorted(np.random.beta(1, 1, n)) for _ in range(nUPrimes)]
    x_sorted = sorted(x)

    ms = [0.0] * len(UPrimes)

    for i in range(len(UPrimes)):
        ms[i] = m(x_sorted, UPrimes[i])

    ms_sorted = sorted(ms)

    return ms_sorted[min(math.ceil((1-delta) * len(ms)), len(ms)-1)]


