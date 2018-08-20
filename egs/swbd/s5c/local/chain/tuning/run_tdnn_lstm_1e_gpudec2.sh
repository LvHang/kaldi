#!/bin/bash

# configs for 'chain' gpu decoding
# NOTICE: 1. we need CUDA9.0 installed with correct driver version
#         2. a GPU not earlier than K20
#         3. as libkaldi-decoder.so needs the CUDA runtime libraries, other libraries using
#            libkaldi-decoder.so also needs them.

stage=19
nj=4
data=data/eval2000_hires

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

dir=tmp/tdnn_lstm_1e_sp/
model_dir=exp/chain/tdnn_lstm_1e_sp

# model training
if [ $stage -le 19 ]; then
  local/chain/tuning/run_tdnn_lstm_1e.sh || exit 1;
fi

# acoustic inference and GPU decoding
if [ $stage -le 20 ]; then
  mkdir -p $dir/log
  graphdir=$model_dir/graph_sw1_tg
  decdir=$dir/decode_`basename $graphdir`/
  mkdir -p $decdir
  subset=1

  sdata=$data/split$nj;
  [[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
  echo $nj > $dir/num_jobs

  queue.pl JOB=1:$nj $dir/log/decode.JOB.log \
  nnet3-compute-gpu --utt2spk=ark:$sdata/JOB/utt2spk --online-ivectors=scp:exp/nnet3/ivectors_eval2000/ivector_online.scp \
    --online-ivector-period=10 --allow-partial=true --num-threads=2 --word-symbol-table=$graphdir/words.txt \
    --extra-left-context=50 --extra-right-context=0 --extra-left-context-initial=0 --extra-right-context-final=0 \
    --frame-subsampling-factor=3 --frames-per-chunk=140 --beam=13 --gpu-raction=0.1 --determinize-lattice=false \
    $model_dir/final.mdl $graphdir/HCLG.fst scp:$sdata/JOB/feats.scp \
    "ark:|lattice-scale --acoustic-scale=10.0 ark:- ark:- | gzip -c >  $decdir/lat.$subset.gz" \
    ark,t:$decdir/words.$subset.txt ark,t:$decdir/ali.$subset.txt 
fi

