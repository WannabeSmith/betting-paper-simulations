# Generating plots for 'Estimating means of bounded random variables by betting'

This repository contains the code for reproducing the plots in 'Estimating means of bounded random variables by betting'.

### Install dependencies
First, install the required dependencies.

```zsh
# Install the required python packages
pip install -r requirements.txt
```

### Generate plots 

Finally, run the shell script `generate_plots.sh`. **Note: this could take quite a while**.
```zsh
# Replace zsh with your preferred shell (e.g. zsh, bash, etc.)
zsh generate_plots.sh
```

Alternatively, you can enter each directory individually and run the relevant simulations:

```zsh
# Enter the desired simulations directory
cd simulations/CM-EB_vs_PM-EB
# Run the python script. This will create figures in `simulations/CM-EB_vs_PM-EB/figures/`.
python simulations.py
```

