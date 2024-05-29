#!/bin/bash

cd /home/mila/c/chris.emezue/representation-learning-assignment/assignment2


#for exp in 1 2 3 4 5 6 7 8
for exp in 2 #4 5 6

do
    #for type in 0 A B C D E F G H I J K
    for type in 0 #F B E

    do
        sbatch run.sh $exp $type
    done
done