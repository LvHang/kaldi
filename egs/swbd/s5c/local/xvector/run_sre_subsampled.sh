#!/bin/bash

#TODO

. ./cmd.sh
set -e

stage=1
train_stage=-10
generate_alignments=true # false if doing ctc training
speed_perturb=true
use_gpu=true
feat_dim=60 # this is the MFCC dim we use in the hires features.  you can't change it
            # unless you change local/xvector/prepare_perturbed_data.sh to use a different
            # MFCC config with a different dimension.
                                # local/xvector/prepare_perturbed_data.sh
xvector_dim=300 # dimension of the xVector.  configurable.
xvector_dir=exp/xvectors_sre_e
egs_dir=exp/xvectors_sre_e/egs
subsample=5


. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

data=data/train_user_no_sil
nj=8
sdata=$data/split$nj;
utils/split_data.sh $data $nj || exit 1;

feats_dir=/mnt/NFS.chanas01vg01/speechrec/davids/xvector_sid_v1/feats
mkdir -p ${feats_dir}/log/
run.pl JOB=1:$nj ${feats_dir}/log/train_user.JOB.log \
  add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- \| \
  apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- \| \
  select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp \
  ark,scp:${feats_dir}/train_user_feats.JOB.ark,${feats_dir}/train_user_feats.JOB.scp || exit 1;

if [ $stage -le 3 ]; then
  # Prepare configs
  mkdir -p $xvector_dir/log

  $train_cmd $xvector_dir/log/make_configs.log \
    steps/nnet3/xvector/make_jesus_configs.py \
      --output-dim $xvector_dim \
      --splice-indexes="0 0 0 0 mean+stddev(0:3:9:90000)" \
      --feat-dim $feat_dim --output-dim $xvector_dim \
      --num-jesus-blocks 100 \
      --jesus-input-dim 300 --jesus-output-dim 1000 --jesus-hidden-dim 2000 \
      $xvector_dir/nnet.config
fi


if [ $stage -le 4 ]; then
  # dump egs.
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,11,12,13}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
  fi
  steps/nnet3/xvector/get_egs_sre_subsample.sh --cmd "$train_cmd" \
    --subsample $subsample \
    --nj 6 \
    --stage 0 \
    --frames-per-iter 60000000 \
    --frames-per-iter-diagnostic 60000000 \
    --min-frames-per-chunk 1000 \
    --max-frames-per-chunk 10000 \
    --num-diagnostic-archives 4 \
    --num-repeats 7 \
    "$data" $egs_dir
fi

if [ $stage -le 5 ]; then
  # training for 4 epochs * 3 shifts means we see each eg 12
  # times (3 different frame-shifts of the same eg are counted as different).
  steps/nnet3/xvector/train.sh --cmd "$train_cmd" \
      --initial-effective-lrate 0.002 \
      --final-effective-lrate 0.0002 \
      --max-param-change 0.4 \
      --minibatch-size 8 \
      --num-epochs 8 --num-shifts 3 --use-gpu $use_gpu --stage $train_stage \
      --num-jobs-initial 1 --num-jobs-final 4 \
      --egs-dir $egs_dir \
      $xvector_dir
fi

if [ $stage -le 6 ]; then
  # uncomment the following line to have it remove the egs when you are done.
  # steps/nnet2/remove_egs.sh $xvector_dir/egs
fi


exit 0;
