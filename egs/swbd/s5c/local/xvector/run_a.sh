#!/bin/bash

# this script prepares the speed-perturbed mfcc training data if it doesn't
# already exist.

. ./cmd.sh
set -e

stage=3
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
use_gpu=true
feat_dim=40 # this is the MFCC dim we use in the hires features.  you can't change it
            # unless you change local/xvector/prepare_perturbed_data.sh to use a different
            # MFCC config with a different dimension.
data=data/train_nodup_sp_hires  # you can't change this without changing
                                # local/xvector/prepare_perturbed_data.sh
xvector_dim=200 # dimension of the xVector.  configurable.
xvector_dir=exp/xvector_a
egs_dir=exp/xvector_a/egs


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

local/xvector/prepare_perturbed_data.sh --stage $stage


if [ $stage -le 3 ]; then
  # Prepare configs
  mkdir -p $xvector_dir/log

  $train_cmd $xvector_dir/log/make_configs.log \
    steps/nnet3/xvector/make_jesus_configs.py \
      --splice-indexes="-1,0,1 -2,-1,0,1 -3,0,3 -3,0,3 mean+stddev+count(0:3:9:396)" \
      --feat-dim $feat_dim --output-dim $xvector_dim \
      --num-jesus-blocks 100 \
      --jesus-input-dim 300 --jesus-output-dim 1000 --jesus-hidden-dim 2000 \
      $xvector_dir/nnet.config
fi
if [ $stage -le 4 ] && false; then
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,11,12,13}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
  fi
  steps/nnet3/xvector/get_egs.sh --cmd "$train_cmd" \
    "$data" $egs_dir
fi

if [ $stage -le 5 ]; then
  # training for 4 epochs * 3 shifts means we see each eg 12
  # times (3 different frame-shifts of the same eg are counted as different).
  steps/nnet3/xvector/train.sh --cmd "$train_cmd" \
      --num-epochs 4 --num-shifts 3 --use-gpu $use_gpu --stage $train_stage \
      --num-jobs-initial 1 --num-jobs-final 8 \
      --egs-dir $egs_dir \
      $xvector_dir
fi

if [ $stage -le 6 ]; then
  # uncomment the following line to have it remove the egs when you are done.
  # steps/nnet2/remove_egs.sh $xvector_dir/egs
fi


exit 0;
