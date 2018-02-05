#!/bin/bash

# Copyright 2017 Pegah Ghahremani
# Apache 2.0

# This script splits the train,test and cv data for sre-8-10
# using fold_info_file.

# TODO: I need to add parts for downloading the data and preparing allfolds.

[ -f ./path.sh ] && . ./path.sh; # source the path.

fold_info_file=data/sre_08_10_folds.json
# End configuration section.

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <test-fold-indx> <data-dir> ouput-dir>";
  echo "e.g.: $0 1 data/train data/train/fold1"
  echo "Options: "
  echo "--fold-info-file             # file with utt information for train,"
  echo "                               test and cv for different folds";
  exit 1;
fi

test_fold=$1
data=$2
out_data=$3

train_data=$out_data/train
test_data=$out_data/test
cv_data=$out_data/cv

mkdir -p $out_data
mkdir -p $train_data $test_data $cv_data || exit 1;
seq 20 70 > $train_data/classes.txt
cp {$train_data,$test_data}/classes.txt
rm -rf $out_data/split_info

if [ -f $fold_info_file ]; then
  local/split_age_data.py --test-fold $test_fold --fold-info-file $fold_info_file \
    --output-dir $out_data/split_info || exit 1;
else
  echo "It expects the $fold_info_file to exist to generate test and train subset." && exit 1;
fi

utils/subset_data_dir.sh --utt-list $out_data/split_info/utt_train.txt $data $train_data || exit 1;

utils/subset_data_dir.sh --utt-list $out_data/split_info/utt_test.txt $data $test_data || exit 1;

utils/subset_data_dir.sh --utt-list $out_data/split_info/utt_cv.txt $data $cv_data || exit 1;

# find shift to make zero-based age labels
age_offset=`cat $data/utt2age | cut -d" " -f2 | sort -n | head -1`
max_age=`cat $data/utt2age | cut -d" " -f2 | sort -n | tail -1`
# age2label used to map labels to zero-based labels during generating egs.
for i in `seq $age_offset $max_age`;do echo $i $[$i-$age_offset]; done > $train_data/label2age || exit 1;

# prepare utt2num_frames
#feat-to-len scp:$train_data/feats.scp ark,t:$train_data/utt2frames
