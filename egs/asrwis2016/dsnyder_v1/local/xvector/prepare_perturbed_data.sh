#!/bin/bash

# this script prepares the speed-perturbed mfcc training data if it doesn't
# already exist.

. ./cmd.sh
set -e

stage=1
generate_alignments=true # false if doing ctc training
speed_perturb=true
mfccdir=mfcc
mfcc_config=conf/mfcc_hires.conf

. ./path.sh
. ./utils/parse_options.sh

# perturbed data preparation
if [ $stage -le 1 ]; then
  for dataset in train_nodup; do
    if [ -d data/${dataset}_sp ]; then
      echo "$0: directory data/${dataset}_sp already exists, skipping creating it."
    else
      utils/data/perturb_data_dir_speed_3way.sh data/${dataset} data/${dataset}_sp
      utils/data/perturb_data_dir_volume.sh data/${dataset}_sp
    fi
    if [ -f data/${dataset}_sp_hires/feats.scp ]; then
      echo "$0: directory data/${dataset}_sp_hires/feats.scp already exists, skipping creating it."
    else
      utils/copy_data_dir.sh data/${dataset}_sp data/${dataset}_sp_hires
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 \
        --mfcc-config $mfcc_config \
        data/${dataset}_sp_hires exp/make_mfcc/${dataset}_sp_hires $mfccdir || exit 1;
      # we typically won't need the cmvn stats when using hires features-- it's
      # mostly for neural nets.
      utils/fix_data_dir.sh data/${dataset}_sp_hires  # remove segments with problems
    fi
  done
fi

if [ $stage -le 2 ]; then
# Make sure the _hires versions of the test sets exist.
  for dataset in eval2000 train_dev; do
  # Create MFCCs for the eval set
    if [ ! -d data/$dataset ]; then
      echo "$0: Expected directory data/$dataset to exist"
      exit 1
    fi
    if [ -f data/${dataset}_hires/feats.scp ]; then
      echo "$0: data/${dataset}_hires/feats.scp already exists, skipping mfcc generation"
    else
      utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
      steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config $mfcc_config \
        data/${dataset}_hires exp/make_hires/$dataset $mfccdir
      steps/compute_cmvn_stats.sh data/${dataset}_hires exp/make_hires/$dataset $mfccdir
      utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
    fi
  done
fi


exit 0;
