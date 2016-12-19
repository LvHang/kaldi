#!/bin/bash

# Begin Configuration section
stage=0
min_additive_noise_len=2 # the minimum duration of each noise file
num_kind_range=4         # the number of kinds of noise ranges
min_snr=0                # the minimum snr value
max_snr=0                # the maximum snr value
seed=-1                  # set the random seed

# End Configuration section

data=$1  # contain wav.scp
noise=$2 # contain noise.scp 
dir=$3   # eg: ranges/


if [ ! -f $data/utt2dur ]; then
  # get original clean wav's duration
  utils/data/get_utt2dur.sh $data 
fi

if [ ! -f $noise/utt2dur ]; then
  # get the duration of each noise file
  utils/data/get_utt2dur.sh $noise
fi

mkdir -p $dir/log
if [ $stage -le 0 ]; then
  echo "$0: generate $num_kind_rage kinds of noise range for each original wav"
  $cmd $dir/log/generate_noise_range.log \
    steps/nnet3/fvector/generate_noise_range.py \
      --num-kind-range=$num_kind_range \
      --min-additive-noise-len=$min_additive_noise_len \
      --min-snr=$min_snr \
      --max-snr=$max_snr \
      --seed=$seed \
      $data/utt2dur $noise/utt2dur $dir
fi

exit 0
