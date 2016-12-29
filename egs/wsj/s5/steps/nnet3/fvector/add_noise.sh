#!/bin/bash
# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin Configuration section.
stage=0
cmd=run.pl
nj=4
# Begain Configuration.
min_additive_noise_len=2.0       # the minimum duration of each noise file
num_kind_range=4                 # the number of kinds of noise ranges
min_snr=-5                       # the minimum snr value
max_snr=-15                      # the maximum snr value
seed=-1                          # set the random seed
variable_len_additive_noise=true #If true, generate the variable-length range files
                                 #If false, generate the fixed-length range files
# End Configuration options.

echo "$0 $@" # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "usage: steps/nnet3/fvector/add_noise.sh <data-dir> <noise-dir> <range-dir>"
  echo "e.g.:  steps/nnet3/fvector/add_noise.sh data/train data/noise ranges"
  echo "main options (for others, see top of script file)"
  echo "  --min-additive-noise-len <second>                # limit the minimum length of noise" 
  echo "  --num-kind-range <n>                             # number of noise range kinds"
  echo "  --variable-len-additive-noise (true|false)       # decide fixed/variable version"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs"
fi

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
      --variable-len-additive-noise $variable_len_additive_noise \
      --seed=$seed \
      $data/utt2dur $noise/utt2dur $dir
fi

exit 0
