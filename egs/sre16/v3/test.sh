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
  #Generate global spk2label file
  awk -v id=0 '{print $1, id++}' ${data}/spk2utt > ${data}/spk2label
  #Generate data_li/feat.scp and utt2spk
  $cmd $dir/log/allocate_examples_test.log \
    allocate_egs_seg.py \
      --num-repeats=3 \
      --min-frames-per-chunk=500 \
      --max-frames-per-chunk=1000 \
      --kinds-of-length=4 \
      --data-dir=$data \
      --output-dir=${data}/temp || exit 1
  #Generate data_li/spk2utt and utt2label
  for dir in `dir ${data}/temp`; do
    utils/utt2spk_to_spk2utt.pl ${data}/temp/$dir/utt2spk > ${data}/temp/$dir/spk2utt
    utils/sym2int.pl -f 2 ${data}/spk2label ${data}/temp/$dir/utt2spk \
      > ${data}/temp/$dir/utt2label
  done
fi

if [ $stage -le 1 ]; then
  echo "$0: Combine, shuffle and split list"
  #combine feats.scp and shuffle
  mkdir -p ${data}/temp/combine
  cat ${data}/temp/data_*/feats.scp > ${data_combine}/feats.scp.bak
  utils/shuffle_list.pl ${data_combine}/feats.scp.bak > ${data_combine}/feats.scp
  #combine utt2spk
  cat ${data}/temp/data_*/utt2spk > ${data_combine}/utt2spk
  #generate spk2utt and utt2label
  utils/utt2spk_to_spk2utt.pl ${data_combine}/utt2spk > ${data_combine}/spk2utt
  utils/sym2int.pl -f 2 ${data}/spk2label ${data_combine}/utt2spk \
    > ${data_combine}/utt2label

  # split feats.scp
  directories=$(for n in `seq $nj`; do echo ${data_combine}/split${n}; done)
  if ! mkdir -p $directories >&/dev/null; then
    for n in `seq $nj`; do
      mkdir -p $data/split${n}
    done
  fi
  feat_scps=$(for n in `seq $nj`; do echo ${data_combine}/split${n}/feats.scp; done)
  utils/split_scp.pl ${data_combine}/feats.scp $feat_scps
  #According to split{n}/feats.scp, generate utt2spk, spk2utt, utt2label
  for n in `seq $nj`; do
    utils/filter_scp.pl ${data_combine}/split${n}/feats.scp ${data_combine}/utt2spk \
      > ${data_combine}/split${n}/utt2spk
    utils/utt2spk_to_spk2utt.pl ${data_combine}/split${n}/utt2spk \
      > ${data_combine}/split${n}/spk2utt
    utils/sym2int.pl -f 2 ${data}/spk2label ${data_combine}/split${n}/utt2spk \
      > ${data_combine}/split${n}/utt2label
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Dump Egs"
  #Note: if --num-pdfs options is not supply, you must use global utt2label,
  #because we will compute the num-pdfs from utt2label file.
  $cmd $dir/log/dump_egs.log \
    nnet3-xvector-get-egs-seg \
      ${data_combine}/utt2label scp:${data_combine}/split1/feats.scp \
      ark,t:${data_combine}/split1/eg.ark
fi
