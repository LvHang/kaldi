#!/bin/bash
# Copyright 2016  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# The script is used to generate the egs which will be used in fvector framework.
# So far, the script achieves the duration files of train dataset and noise
# dataset seperately. Then, with the duration files, it will generate the range
# file which is used to control the process about adding additive noise. 
# At the same time, it will generate the mapping between wav and perturbedwav.

# Begin Configuration section.
stage=0
cmd=run.pl
nj=4
# Begain Configuration.
min_additive_noise_len=2.0       # the minimum duration of each noise file in seconds.
num_ranges_per_wav=4             # the number of noise ranges for each wav.
min_snr=-5                       # the minimum snr value in dB.
max_snr=-15                      # the maximum snr value in dB.
seed=-1                          # set the random seed.
variable_len_additive_noise=true #If true, generate the variable-length range files.
                                 #If false, generate the fixed-length range files.
# End Configuration options.

echo "$0 $@" # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "usage: steps/nnet3/fvector/add_noise.sh <data-dir> <noise-dir> <range-dir>"
  echo "e.g.:  steps/nnet3/fvector/add_noise.sh data/train data/noise ranges"
  echo "main options (for others, see top of script file)"
  echo "  --min-additive-noise-len <second>                # limit the minimum length of noise" 
  echo "  --num-ranges-per-wav <n>                         # number of noise range kinds"
  echo "  --variable-len-additive-noise (true|false)       # decide fixed/variable version"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs"
fi

data=$1  # contain wav.scp
noise=$2 # contain noise.scp 
dir=$3   # eg: data/perturbed

# remove the segments so that the duration corresponding to recording-id
if [ -f $data/segments ]; then
  mv $data/segments $data/segments_backup
  if [ -f $data/utt2dur ]; then
    mv $data/utt2dur $data/utt2dur.backup
    utils/data/get_utt2dur.sh $data
  else
    utils/data/get_utt2dur.sh $data
  fi
  mv $data/segments_backup $data/segments
else
  if [ ! -f $data/utt2dur ]; then
    # get original clean wav's duration
    utils/data/get_utt2dur.sh $data
  fi 
fi

# remove the segments so that the duration corresponding to recording-id
if [ -f $noise/segments ]; then
  mv $noise/segments $noise/segments_backup
  if [ -f $noise/utt2dur ]; then
    mv $noise/utt2dur $noise/utt2dur.backup
    utils/data/get_utt2dur.sh $noise
  else
    utils/data/get_utt2dur.sh $noise
  fi
  mv $noise/segments_backup $noise/segments
else
  if [ ! -f $noise/utt2dur ]; then
    # get original clean wav's duration
    utils/data/get_utt2dur.sh $noise
  fi 
fi


mkdir -p $dir/log
if [ $stage -le 0 ]; then
  echo "$0: generate $num_kind_rage kinds of noise range for each original wav"
  $cmd $dir/log/generate_noise_range.log \
    steps/nnet3/fvector/generate_noise_range.py \
      --num-ranges-per-wav=$num_ranges_per_wav \
      --min-additive-noise-len=$min_additive_noise_len \
      --min-snr=$min_snr \
      --max-snr=$max_snr \
      --variable-len-additive-noise $variable_len_additive_noise \
      --seed=$seed \
      $data/utt2dur $noise/utt2dur $dir/ranges $dir/wav2perturbedwav
fi

if [ $stage -le 1 ]; then
  echo "$0: generate perturbed_wav_specifier"
  $cmd $dir/log/generate_perturb_wav_specifier.log \
    steps/nnet3/fvector/generate_perturb_wav_specifier.py \
      --noise=$noise/wav.scp \
      $data/wav.scp $dir/ranges $dir/wav2perturbedwav $dir/wav.scp
fi

if [ $stage -le 2 ]; then
  echo "$0: generate other files in data directory"
  #reco2file_and_channel
  cat $dir/wav2perturbedwav | cut -d ' ' -f 1 | paste -d ' ' - $dir/wav2perturbedwav > $dir/perturb_recording_map
  steps/nnet3/fvector/apply_map_one2mult.pl -f 1 $dir/perturb_recording_map <$data/reco2file_and_channel >$dir/reco2file_and_channel
  if [ -f $data/segments ]; then
    awk -v num=$num_ranges_per_wav '{
      printf("%s %s",$1, $1);
      for(i=1; i<= num; i++){ printf(" %s%s-%s","perturb", i, $1); }
      printf("\n");
    }' <$data/segments > $dir/perturb_utt_map
    cat $dir/perturb_recording_map > $dir/perturb_map
    cat $dir/perturb_utt_map >> $dir/perturb_map
    #segments
    steps/nnet3/fvector/apply_map_one2mult.pl -f 1 $dir/perturb_map <$data/segments >$dir/segments
    #text
    steps/nnet3/fvector/apply_map_one2mult.pl -f 1 $dir/perturb_map <$data/text >$dir/text
    #utt2spk
    steps/nnet3/fvector/apply_map_one2mult.pl -f 1 $dir/perturb_map <$data/utt2spk >$dir/utt2spk
    #spk2utt
    utt2spk_to_spk2utt.pl <$dir/utt2spk | sort > $dir/spk2utt
  else #no segments->wav indexed by utterence-id/<recording-id> is equal to <utt-id>
    cp $dir/perturb_recording_map $dir/perturb_map
    #segments
    steps/nnet3/fvector/apply_map_one2mult.pl -f 1 $dir/perturb_map <$data/segments >$dir/segments
    #text
    steps/nnet3/fvector/apply_map_one2mult.pl -f 1 $dir/perturb_map <$data/text >$dir/text
    #utt2spk
    steps/nnet3/fvector/apply_map_one2mult.pl -f 1 $dir/perturb_map <$data/utt2spk >$dir/utt2spk
    #spk2utt
    utt2spk_to_spk2utt.pl <$dir/utt2spk | sort > $dir/spk2utt
  fi
fi

if [ -f $data/utt2dur.backup ]; then
  mv $data/utt2dur.backup $data/utt2dur
fi
if [ -f $noise/utt2dur.backup ]; then
  mv $noise/utt2dur.backup $noise/utt2dur
fi

exit 0
