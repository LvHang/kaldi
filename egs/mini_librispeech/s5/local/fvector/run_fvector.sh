#!/bin/bash

. ./cmd.sh
set -e

train_stage=-10
data=data/train_clean_5
noise_data=data/noise
egs_dir=exp/fvector/egs
fvector_dir=exp/fvector

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ $train_stage -le 2 ]; then
  #dump egs                                         
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then  
    utils/create_split_dir.pl \
    /export/b{11,12,13,14}/$USER/kaldi-data/egs/minilibrispeech-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
  fi

  steps/nnet3/fvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --frames-per-iter 2000000 \
    --frames-per-iter-diagnostic 200000 \
    --num-diagnostic-percent 5 \
    "$data" "$noise_data" "$egs_dir"
fi

#if [ $stage -le 4 ]; then
#  #prepare configs
#fi

if [ $stage -le 5 ]; then
  #training
  steps/nnet3/xvector/train.sh --cmd "$train_cmd" \                             
    --initial-effective-lrate 0.002 \                                         
    --final-effective-lrate 0.0002 \                                          
    --max-param-change 0.2 \                                                  
    --minibatch-size 16 \                                                     
    --num-epochs 4 --use-gpu $use_gpu --stage $train_stage \            
    --num-jobs-initial 1 --num-jobs-final 8 \                                 
    --egs-dir $egs_dir \                                                      
    $fvector_dir
fi
