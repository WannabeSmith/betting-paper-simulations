#include <iostream>
#include <vector>
#include <random>

using namespace std;

// The function m from Learned-Miller and Thomas' formulation of Gaffke's inequality. Assumes z are the order statistics (already sorted)
double m(const vector<double>& z, const vector<double>& u)
{
	int n = (int)z.size();
	double result = 1.0;
	for (int i = 0; i < n - 1; i++)
		result -= (z[i + 1] - z[i]) * u[i];
	result -= (1.0 - z[n - 1]) * u[n - 1];
	return result;
}

// Gaffke's bound (also LMT from our Arxiv paper)
double Gaffke(vector<double> x, const double& delta, const vector<vector<double>>& UPrimes)
{
	// Sort the samples to get the order statistics
	sort(x.begin(), x.end());

	// Create array that will be used to store m(z,U) for each of the samples of U.
	vector<double> ms(UPrimes.size());

	// Compute m(z,u) for each u in UPrimes
	for (int i = 0; i < (int)UPrimes.size(); i++)
		ms[i] = m(x, UPrimes[i]);

	// Sort the m(z,u) values
	sort(ms.begin(), ms.end(), [](const double& a, const double& b) {return a > b; });

	// Return the value of m(z,u) such that roughly 100*delta percent are smaller. With large number of UPrimes we don't worry about floor/ceiling when casting to an integer
	return ms[(int)(delta * ms.size())];
}

// Gaffke implemntation requires UPrimes. This function loads these values
void loadUPrimes(vector<vector<double>>& UPrimes, int nUPrimes, int n, mt19937_64& generator)
{
	// Resize to the number of samples of U
	UPrimes.resize(nUPrimes);

	// Create a uniform distribution over [0,1]
	uniform_real_distribution<double> uniformDistribution(0, 1);

	// Loop over entires of UPrimes, loading each one
	for (auto& u : UPrimes)
	{
		// First resize this sample to be of length n
		u.resize(n);

		// Load each value with a sample from U(0,1)
		for (double & value : u)
			value = uniformDistribution(generator);

		// Sort the samples
		sort(u.begin(), u.end());
	}
}

int main(int argc, char* argv[])
{
	// Hyperparameters
	int nUPrimes = 10000;									// Number of samples of UPrime to use. Typically 10,000 for quick runs, 100,000 or more for more precise estiamtes.
	int n = 3;												// Number of samples of the random variable
	int numTrials = 10000;									// Number of times we will run Gaffke's bound
	double alpha = 5.0, beta = 1.0;							// Parameters of a beta distribution
	double delta = 0.05;									// We ask for a 1-delta confidence upper bound

	// Create the actual distribution and compute its mean
	_Beta_distribution<double> distribution(alpha, beta);
	double mean = alpha / (alpha + beta);					// Mean of the beta distribution

	// Create the random number generator
	mt19937_64 generator;									// Just use the default seed, though this makes the run deterministic

	// Load UPrimes once
	vector<vector<double>> UPrimes;							// Create the vector of vectors. UPrimes[i] is a vector corresponding to one sample of the random variable U.
	loadUPrimes(UPrimes, nUPrimes, n, generator);			// Load UPrimes

	// Track the average high-confidence upper bound output by Gaffke and the numeber of times it failed
	int numFailures = 0;									// How many times was the high-confidence upper bound larger than the mean?
	double averageGaffke = 0;								// What is the average value of the output of Gaffke? (At first this is the sum, then later we divide by the number of trials to get the average)

	// Loop over trials
	vector<double> sample(n);								// Will hold the samples we generate.
	for (int trialCount = 0; trialCount < numTrials; trialCount++)
	{
		// Sample each of the n values
		for (double& value : sample)						// For each value in sample
			value = distribution(generator);				// Sample the value from the generator

		// Run Gaffke
		double GaffkeOutput = Gaffke(sample, delta, UPrimes);

		// Update numFailures and average
		averageGaffke += GaffkeOutput;
		if (GaffkeOutput < mean)
			numFailures++;
	}

	// Right now average is actually the sum. Convert to average
	averageGaffke /= (double)numTrials;
	double failureProbability = (double)numFailures / (double)numTrials;

	// Print results
	cout << "We ran Gaffke " << numTrials << " times using " << n << " samples from Beta(" << alpha << ", " << beta << ") and delta = " << delta << "." << endl;
	cout << "\tActual mean = " << mean << endl;
	cout << "\tAverage " << 1.0 - delta << "-confidence upper bound on the mean using Gaffke: " << averageGaffke << endl;
	cout << "\tGaffke output something larger than the mean in " << numFailures << " of " << numTrials << " trials. That's an empirical probability of " << failureProbability << "." << endl;

	cout << endl << "Done. Terminating." << endl;
	exit(0);
}
