#!/bin/bash

set -e

if [ -z "$1" ]
then
    echo "Error: no dataset given"
    exit 1
fi

dataset=$1
script="$HOME/Mnist_v6_sbatch/Mnist_TI.py"
origin_folder="/mnt/smb/locker/abbott-locker/Luke/Mnist/Mnist_v6_sbatch"

# Create a fresh file
> ${dataset}_jobs.txt

for exception in true false
do
    for dropout in true false
    do
        for intermediate_layer in true false
        do
            #Baseline
            output_folder=${origin_folder}/exception_${exception}_dropout_${dropout}_intermediate_layer_${intermediate_layer}
            echo  "${script} --output-folder ${output_folder} --exception ${exception} --dropout ${dropout} --intermediate-layer ${intermediate_layer} --epochs 100" >> ${dataset}_jobs.txt
        done
    done
done