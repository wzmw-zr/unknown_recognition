#!/bin/bash
if [[ ! -d results ]]
then
  mkdir results
fi

learning_rate=$1
alpha=$2
beta=$3

logit_norm_types=(global local)
norm_types=(LN BN None)
# lrs=(0.00001 0.0001 0.001 0.01)
# lrs=(0.1 0.07 0.05 0.02 0.01 0.007 0.005 0.002 0.001 0.0007 0.0005 0.0002 0.0001)
lrs=($learning_rate)
# epochs=(20 50 100)
epochs=(100)

# for lr_update in CosineAnnealing poly
for lr_update in fixed
do
  for lr in ${lrs[*]}
  do
    for norm_type in ${norm_types[*]}
    do
      for epoch in ${epochs[*]}
      do
        if [[ ! -d results/mlp_softmax ]]
        then
          mkdir results/mlp_softmax
        fi
        python train_single_gpu/mlp_softmax_tversky_loss.py\
        ${lr} ${epoch} ${lr_update} ${norm_type} ${alpha} ${beta}\
        > results/mlp_softmax/mlp_softmax_lr_${lr}_epoch_${epoch}_${lr_update}_${norm_type}_${alpha}_${beta} 2>&1

        # if [[ ! -d results/mlp_softmax_distance ]]
        # then
        #   mkdir results/mlp_softmax_distance
        # fi
        # python train_single_gpu/mlp_softmax_distance_tversky_loss.py\
        # ${lr} ${epoch} ${lr_update} ${norm_type} ${alpha} ${beta}\
        # > results/mlp_softmax_distance/mlp_softmax_distance_lr_${lr}_epoch_${epoch}_${lr_update}_${norm_type}_${alpha}_${beta} 2>&1

        for logit_norm_type in ${logit_norm_types[*]}
        do
          if [[ ! -d results/mlp_logit ]]
          then
            mkdir results/mlp_logit
          fi
          python train_single_gpu/mlp_logit_tversky_loss.py\
          ${lr} ${epoch} ${lr_update} ${logit_norm_type} ${norm_type} ${alpha} ${beta}\
          > results/mlp_logit/mlp_logit_lr_${lr}_epoch_${epoch}_${lr_update}_${logit_norm_type}_${norm_type}_${alpha}_${beta} 2>&1

          if [[ ! -d results/mlp_logit_softmax ]]
          then
            mkdir results/mlp_logit_softmax 
          fi
          python train_single_gpu/mlp_logit_with_softmax_tversky_loss.py\
          ${lr} ${epoch} ${lr_update} ${logit_norm_type} ${norm_type} ${alpha} ${beta}\
          > results/mlp_logit_softmax/mlp_logit_softmax_lr_${lr}_epoch_${epoch}_${lr_update}_${logit_norm_type}_${norm_type}_${alpha}_${beta} 2>&1

          # if [[ ! -d results/mlp_max_logit ]]
          # then
          #   mkdir results/mlp_max_logit
          # fi
          # python train_single_gpu/mlp_max_logit_tversky_loss.py\
          # ${lr} ${epoch} ${lr_update} ${logit_norm_type} ${norm_type} ${alpha} ${beta}\
          # > results/mlp_max_logit/mlp_max_logit_lr_${lr}_epoch_${epoch}_${lr_update}_${logit_norm_type}_${norm_type}_${alpha}_${beta} 2>&1

          # if [[ ! -d results/mlp_max_logit_softmax_distance ]]
          # then
          #   mkdir results/mlp_max_logit_softmax_distance
          # fi
          # python train_single_gpu/mlp_max_logit_with_softmax_distance_tversky_loss.py\
          # ${lr} ${epoch} ${lr_update} ${logit_norm_type} ${norm_type} ${alpha} ${beta}\
          # > results/mlp_max_logit_softmax_distance/mlp_max_logit_softmax_distance_lr_${lr}_epoch_${epoch}_${lr_update}_${logit_norm_type}_${norm_type}_${alpha}_${beta} 2>&1

        done
      done
    done
  done
done

