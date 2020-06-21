#!/bin/bash
for optimizer in "sgdm" "adam"; do
    for lr in 1e-2 1e-3 1e-4; do
        for wd in 1e-3 1e-4 0; do
            python3 main.py --save=False --optimizer=$optimizer --epochs=5 --pretrained=False --lr=$lr --weight_decay=$wd
        done
    done
done 
