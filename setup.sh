#!/bin/bash

cd /home/mila/c/chris.emezue/representation-learning-assignment

# for opt in 'adam' 'adamw'
# do
#     for lr in '0.01' '0.001' '0.1'
#     do
#         sbatch run_mlp.sh $opt $lr
#         sbatch run_cnn.sh $opt $lr

#     done
# done
#python chris-assignment-mlp.py 

for opt in 'random' 'weighted'
do
    sbatch run_mlp.sh $opt
    sbatch run_cnn.sh $opt
done