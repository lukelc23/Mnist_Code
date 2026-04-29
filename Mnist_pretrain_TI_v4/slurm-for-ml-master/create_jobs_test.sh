#!/bin/bash

set -e

if [ -z "$1" ]
then
    echo "Error: no dataset given"
    exit 1
fi

dataset=$1
script="$HOME/Projects/Mnist_TI/Mnist_pretrain_TI_v3/Mnist_TI.py"
origin_folder="/mnt/smb/locker/abbott-locker/Luke/Mnist/Mnist_pretrain_TI_v3"
pretrained_model="$HOME/Projects/Mnist_TI/Mnist_pretrain_TI_v3/mnist_cnn.pt"
n_items=9
ordering_seed=1
exception_pair="5 3"

# Create a fresh file
> ${dataset}_jobs.txt

# Configuration 1: no exception, no pretrained weights
output_folder=${origin_folder}/test_ordering_${ordering_seed}_exception_false_nitems_${n_items}_baseline
echo "${script} --output-folder ${output_folder} --exception false --dropout false --intermediate-layer false --ordering-seed ${ordering_seed} --n-items ${n_items} --epochs ${epochs} --save-model --freeze-conv false" >> ${dataset}_jobs.txt

# Configuration 2: exception + pretrained (exercises rank-based exception_pair path)
exception_pair_name="${exception_pair// /_}"
output_folder=${origin_folder}/test_ordering_${ordering_seed}_exception_true_exceptionpair_${exception_pair_name}_nitems_${n_items}_pretrained
echo "${script} --output-folder ${output_folder} --exception true --exception-pair ${exception_pair} --dropout false --intermediate-layer false --ordering-seed ${ordering_seed} --n-items ${n_items} --epochs ${epochs} --save-model --freeze-conv false --pretrained-weights ${pretrained_model}" >> ${dataset}_jobs.txt