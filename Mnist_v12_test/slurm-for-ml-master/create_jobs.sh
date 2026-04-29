#!/bin/bash

set -e

if [ -z "$1" ]
then
    echo "Error: no dataset given"
    exit 1
fi

dataset=$1
script="$HOME/Projects/Mnist_v12_test/Mnist_TI.py"
origin_folder="/mnt/smb/locker/abbott-locker/Luke/Mnist/Mnist_v12_test"
n_items=9
exception_pair="5 3"

# Create a fresh file
> ${dataset}_jobs.txt

for ordering_seed in $(seq 1 5)
do
    for exception in true false
    do
        for dropout in true false
        do
            for intermediate_layer in true false
            do
                if [ "$exception" = "true" ]; then
                    exception_pair_arg="--exception-pair ${exception_pair}"
                    exception_pair_name="${exception_pair// /_}"
                else
                    exception_pair_arg=""
                    exception_pair_name="none"
                fi
                output_folder=${origin_folder}/ordering_${ordering_seed}_exception_${exception}_exceptionpair_${exception_pair_name}_nitems_${n_items}_dropout_${dropout}_intermediate_layer_${intermediate_layer}
                echo "${script} --output-folder ${output_folder} --exception ${exception} --dropout ${dropout} --intermediate-layer ${intermediate_layer} --ordering-seed ${ordering_seed} --n-items ${n_items} --epochs 100 --save-model ${exception_pair_arg}" >> ${dataset}_jobs.txt
            done
        done
    done
done