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
if [ $stage -le 1 ]; then
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

  utils/subset_data_dir.sh --shortest data/train_clean_5 500 data/train_500short
fi

#Stage2: prepare a noise dir(maybe a speicial noise dataset). In mini_librispeech,
#we just use trainset directly.
if [ $stage -le 2 ]; then
  cp -r data/train_clean_5 data/noise
  #for the noise dir, we prepare a file utt2dur_fix. Each line is "utt_id dur-0.2"
  #This file is used in "fvector-chunk.cc". It will be store into a vector in binary code.
  #For each target chunk, we randomly select two utt_id form vector, and the 
  #corresponding start point.
  utils/data/get_utt2dur.sh data/noise  # wav-to-duration
  cat data/noise/utt2dur | awk '{print $1 $2-0.2}' > data/noise/utt2dur_fix
fi

#stage3: get the (120ms) chunks from wav.scp and noise.scp. And compose 1 source
# chunk and 2 noise chunks into a matrix.
if [ $stage -le 3 ]; then
  fvector-chunk --chunk-size=120 scp:data/train_clean_5/wav.scp scp:data/noise/wav.scp \
    data/noise/utt2dur_fix ark,scp:data/train_clean_5/chunks.ark,data/train_clean_5/chunks.scp
fi

#stage4: Deal with the chunk one-by-one, add the noise.
if [ $stage -le 4 ]; then
  fvector-add-noise scp:data/train_clean_5/chunks.scp \
    scp,ark:data/train_clean_5/perturbed_chunks.scp,data/train_clean_5/perturbed_chunks.ark
fi

#stage5: convert the chunk data into Nnet3eg
if [ $stage -le 5 ]; then
  #It will be implement in a bash script like steps/nnet3/get_egs.sh
  #The core of the script is following
  fvector-get-egs scp:data/train_clean_5/perturbed_chunks.scp ark:egs.ark
fi

if [ $stage -le 6 ]; then
  #The other step is simliar with xvector. 
  #Use nnet3-shuffle-egs | nnet3-merge-egs to combine the separate Nnet3eg into
  #minibatch
  #And Train the plda nnetwork.
fi
