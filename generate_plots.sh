#!/bin/sh

echo Generating plots...
echo ${bold}Note: This could take a while.${normal}

cd simulations

declare -a sim_dirs=(
    'aKelly_lambdas'
    'CM-EB_vs_PM-EB'
    'Fixed-time'
    'Fixed-time_everything'
    'Fixed-time_Phan'
    'game_theoretic_wealths'
    'Hedged_CI_tweaks'
    'Max_vs_sum'
    'Non_iid_time_uniform'
    'Predmix_CI_vs_MP'
    'Time-uniform'
    'Time-uniform_everything'
    'WoR_Fixed-time'
    'WoR_time-uniform'
)

for sim_dir in ${sim_dirs[@]}
do
    cd $sim_dir
    python_scripts=./*.py

    for python_script in *.py
    do
        echo "Running " $sim_dir"/"$python_script
        python -W ignore $python_script
    done

    cd ..
done
