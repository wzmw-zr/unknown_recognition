#!/bin/bash
if [[ ! -d results ]]
then
  mkdir results
fi

for lr_update in CosineAnnealing
do
  logit_norm_types=(local global)
  lrs=(0.1)
  epochs=(300)

  unknown_ratio=0.1
  known_ratio=0.9
  for lr in ${lrs[*]}
  do
    for epoch in ${epochs[*]}
    do
      if [[ ! -d results/mlp_softmax ]]
      then
        mkdir results/mlp_softmax
      fi

      python train_single_gpu/mlp_softmax_300_epoch_logit_adjust_loss.py ${lr} ${epoch} ${lr_update} ${unknown_ratio} ${known_ratio} > results/mlp_softmax/mlp_softmax_with_logit_adjust_loss_lr_${lr}_epoch_${epoch}_${lr_update}_${unknown_ratio}_${known_ratio} 2>&1

      for logit_norm_type in ${logit_norm_types[*]}
      do
        if [[ ! -d results/mlp_logit_${logit_norm_type} ]]
        then
          mkdir results/mlp_logit_${logit_norm_type}
        fi

        python train_single_gpu/mlp_logit_300_epoch_logit_adjust_loss.py ${lr} ${epoch} ${lr_update} ${unknown_ratio} ${known_ratio} ${logit_norm_type} > results/mlp_logit_${logit_norm_type}/mlp_logit_with_logit_adjust_loss_lr_${lr}_epoch_${epoch}_${lr_update}_${unknown_ratio}_${known_ratio}_${logit_norm_type} 2>&1

        if [[ ! -d results/mlp_logit_with_softmax_${logit_norm_type} ]]
        then
          mkdir results/mlp_logit_with_softmax_${logit_norm_type}
        fi

        python train_single_gpu/mlp_logit_adjust_loss_300_epoch.py ${lr} ${epoch} ${lr_update} ${unknown_ratio} ${known_ratio} ${logit_norm_type} > results/mlp_logit_with_softmax_${logit_norm_type}/mlp_logit_adjust_loss_lr_${lr}_epoch_${epoch}_${lr_update}_${unknown_ratio}_${known_ratio}_${logit_norm_type} 2>&1
      done
    done
  done

done

