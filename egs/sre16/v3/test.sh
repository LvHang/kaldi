#!/bin/bash

stage=0
cmd=run.pl
nj=10
data=data/test
data_combine=data/test/temp/combine
dir=exp/test_seg

. ./cmd.sh
. ./path.sh
set -e

mkdir -p $dir/log

if [ $stage -le 0 ]; then
  echo "$0: Allocating egs"
  $cmd $dir/log/allocate_examples_test.log \
    allocate_egs_seg.py \
      --num-repeats=3 \
      --min-frames-per-chunk=500 \
      --max-frames-per-chunk=1000 \
      --kinds-of-length=4 \
      --data-dir=$data \
      --output-dir=${data}/temp || exit 1
fi

if [ $stage -le 1 ]; then
  echo "$0: Combine, shuffle and split list"
  mkdir -p ${data}/temp/combine
  cat ${data}/temp/data_*/feats.scp > ${data_combine}/feats.scp.bak
  utils/shuffle_list.pl ${data_combine}/feats.scp.bak > ${data_combine}/feats.scp
  # split feats.scp
  directories=$(for n in `seq $nj`; do echo ${data_combine}/split${n}; done)
  if ! mkdir -p $directories >&/dev/null; then
    for n in `seq $nj`; do
      mkdir -p $data/split${n}
    done
  fi
  feat_scps=$(for n in `seq $nj`; do echo ${data_combine}/split${n}/feats.scp; done)
  utils/split_scp.pl ${data_combine}/feats.scp $feat_scps
fi

if [ $stage -le 2 ]; then
  echo "$0: Dump Egs"
  $cmd $dir/log/dump_egs.log \
    nnet3-xvector-get-egs-seg \
      ${data}/utt2spk ${data}/spk2num scp:${data_combine}/split1/feats.scp \
      ark,t:${data_combine}/split1/eg.ark
fi
