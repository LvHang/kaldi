#!/bin/bash

# Note: this works only on pre-downloaded data on the CLSP servers
data=/export/a05/dgalvez/

data_url=www.openslr.org/resources/31
lm_url=www.openslr.org/resources/11

. ./cmd.sh
. ./path.sh

stage=0
. utils/parse_options.sh

set -euo pipefail

mkdir -p $data

for part in dev-clean-2 train-clean-5; do
  local/download_and_untar.sh $data $data_url $part
done

if [ $stage -le 0 ]; then
  local/download_lm.sh $lm_url data/local/lm
fi

#prepare the data. Generate dirs: lang, dev_clean_2, train_clean_5, train_500short 
if [ $stage -le 0 ]; then
  # format the data as Kaldi data directories
  for part in dev-clean-2 train-clean-5; do
    # use underscore-separated names in data directories.
    local/data_prep.sh $data/LibriSpeech/$part data/$(echo $part | sed s/-/_/g)
  done

  local/prepare_dict.sh --stage 3 --nj 30 --cmd "$train_cmd" \
    data/local/lm data/local/lm data/local/dict_nosp

  utils/prepare_lang.sh data/local/dict_nosp \
    "<UNK>" data/local/lang_tmp_nosp data/lang_nosp

  local/format_lms.sh --src-dir data/lang_nosp data/local/lm
  # Create ConstArpaLm format language model for full 3-gram and 4-gram LMs
  utils/build_const_arpa_lm.sh data/local/lm/lm_tglarge.arpa.gz \
    data/lang_nosp data/lang_nosp_test_tglarge
  
  mfccdir=mfcc
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/b{07,14,16,17}/$USER/kaldi-data/egs/librispeech/s5/$mfcc/storage \
      $mfccdir/storage
  fi

  for part in dev_clean_2 train_clean_5; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$part exp/make_mfcc/$part $mfccdir
    steps/compute_cmvn_stats.sh data/$part exp/make_mfcc/$part $mfccdir
  done

  # Get the shortest 500 utterances first because those are more likely
  # to have accurate alignments.
  utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short
fi

#Stage2: prepare a noise dir(maybe a speicial noise dataset). In mini_librispeech,
#we just use trainset directly.
if [ $stage -le 1 ]; then
  cp -r data/train_clean_5 data/noise
  #for the noise dir, we prepare a file utt2dur_fix. Each line is "utt_id dur-0.2"
  #This file is used in "fvector-chunk.cc". It will be store into a vector in binary code.
  #For each target chunk, we randomly select two utt_id form vector, and the 
  #corresponding start point.
  utils/data/get_utt2dur.sh data/noise  # wav-to-duration
  cat data/noise/utt2dur | awk '{print $1,$2-0.2}' > data/noise/utt2dur_fix
fi

if [ $stage -le 2 ]; then
#generate fvector egs and train model.
local/fvector/run_fvector.sh --train-stage $stage --data data/train_clean_5 --noise-data data/noise \
  --egs-dir exp/fvector/egs --fvector-dir exp/fvector
fi

