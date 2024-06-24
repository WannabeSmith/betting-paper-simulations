from gaffke import gaffke
import numpy as np

n = 1000

numTrials = 100


population = np.random.beta(1000, 0.1, n)
mu = np.mean(population)

generate_random_nums = lambda x: np.random.choice(population, n)

delta = 0.01

# UPrimes = [sorted(np.random.beta(1, 1, n)) for _ in range(nUPrimes)]

numFailures = 0

for trial_idx in range(numTrials):
    print("Running trial:", trial_idx)
    sample = generate_random_nums(n)

    gaffke_bd = gaffke(x=sample, delta=delta, nUPrimes=1000)

    if gaffke_bd < mu:
        numFailures += 1

print("Fraction of failures:", numFailures / numTrials)
