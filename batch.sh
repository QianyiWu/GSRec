#!/bin/bash
for i in {1..8}
do
    python train.py\
        --gpu 0\
        -s ./data/replica/scan$i/ --voxel_size 0.001\
        --update_init_factor 16\
        --iterations 30000\
        --mlp_sdf_lr_init 0.0005\
        --implicit_sdf_divide_factor 1.0\
        --fmls_sdf_offset 0.01\
        --fmls_use_normal\
        --use_wandb\
        --knear 50\
        --ps_depth 8\
        --sdf_inside_out\
        --eval\
        --sdf_start_iter 15000\
        --use_implicit\
        -m outputs/scan$i
done
