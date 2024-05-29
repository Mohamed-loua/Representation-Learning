#!/bin/bash

cd /home/mila/c/chris.emezue/representation-learning-assignment/assignment2


#for exp in 1 2 3 4 5 6 7 8
for exp in 8

do
    sbatch run.sh $exp
done