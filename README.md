# Generating plots for 'Estimating means of bounded random variables by betting'

This repository contains the code for reproducing the plots in 'Estimating means of bounded random variables by betting'.

### Install dependencies
To generate all of the plots used in this paper, first install the required dependencies.

```zsh
# Install the required python packages
pip install requirements.txt
```

### Generate plots 

We now have all the dependencies required to generate the plots used throughout the paper.

Edit the file `generate_plots.sh` so that your python environment is loaded correctly. For example, my `python` virtual environment is called `venv_betting` but yours can be anywhere:
```zsh
# generate_plots.sh
...
...
...
# Set up pyenv/venv/conda/etc. environment so that you can load the necessary packages
source venv_betting/bin/activate
...
...
...
```

Finally, run the shell script `generate_plots.sh`. **Note: this could take quite a while**.
```zsh
# Replace zsh with your preferred shell (e.g. zsh, bash, etc.)
zsh generate_plots.sh
```
