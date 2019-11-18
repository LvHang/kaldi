#!/bin/bash

# Copyright 2012-2015  Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.

# This script does decoding with a neural-net.

# Begin configuration section.
stage=1
nj=4 # number of decoding jobs.
acwt=1.0  # Just a default value, used for adaptation and beam-pruning..
          # can be used in 'chain' systems to scale acoustics by 1.0
post_decode_acwt=10.0  # can be used in 'chain' systems to scale acoustics by 10 so the
                       # regular scoring script works.
cmd=run.pl
beam=15.0
max_active=7000
min_active=200
lattice_beam=8.0 # Beam we use in lattice generation.

iter=final
use_gpu=false # If true, will use a GPU
use_batch=false # If true, will use nnet3-compute-batch to compute the posterior

scoring_opts=
skip_diagnostics=false
skip_scoring=false

ivector_scale=1.0
frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1
online_ivector_dir=
minimize=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
  echo "e.g.:   steps/nnet3/decode_posterior.sh --nj 8 \\"
  echo "--online-ivector-dir exp/nnet2_online/ivectors_test_eval92 \\"
  echo "    exp/chain/tdnn_1d_sp/graph_tgsmall data/test_eval92_hires exp/chain/tdnn_1d_sp/decode_bg_eval92"
  echo ""
  echo "This script will generate the posteriors in first or use them directly"
  echo "if exists. Then it will decode from the posteriors."
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 15.0"
  echo "  --iter <iter>                            # Iteration of model to decode; default is final."
  echo "  --scoring-opts <string>                  # options to local/score.sh"
  echo "  --use-gpu <true|false>                   # default: false.  If true, we recommend"
  echo "                                           # to use large --num-threads as the graph"
  echo "                                           # search becomes the limiting factor."
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.
model=$srcdir/$iter.mdl


extra_files=
if [ ! -z "$online_ivector_dir" ]; then
  steps/nnet2/check_ivectors_compatible.sh $srcdir $online_ivector_dir || exit 1
  extra_files="$online_ivector_dir/ivector_online.scp $online_ivector_dir/ivector_period"
fi

utils/lang/check_phones_compatible.sh {$srcdir,$graphdir}/phones.txt || exit 1

for f in $graphdir/HCLG.fst $model $extra_files; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
cmvn_opts=`cat $srcdir/cmvn_opts` || exit 1;
batch_string=
if $use_gpu; then
  if $use_batch; then
    batch_string="-batch"
  fi
  queue_opt="--gpu 1"
fi

## Check data
mkdir -p $dir/log
data_ok=false
if [ -z $data/feats.scp ]; then
  [[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
  echo $nj > $dir/num_jobs
  data_ok=true
  stage=0
fi
if [ $sdata/1/posteriors.scp ]; then
  echo -n "Use the posteriors directly. Bear in mind, I just check the first "
  echo "sub-dir. If you modify something, you need to re-generate it."
  data_ok=true
  stage=1
fi
if ! $data_ok; then
  echo "$O: Neither posterior.scp nor feats.scp exists."
  exit 1;
fi

posteriors="ark,scp:$sdata/JOB/posterior.ark,$sdata/JOB/posterior.scp"
posteriors_rspecifier="scp:$sdata/JOB/posterior.scp"
if [ $stage -le 0 ]; then
  ## Set up features. Generate the posteriors.
  if [ -f $srcdir/online_cmvn ]; then online_cmvn=true
  else online_cmvn=false; fi

  if ! $online_cmvn; then
    echo "$0: feature type is raw"
    feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- |"
  else
    feats="ark,s,cs:apply-cmvn-online $cmvn_opts --spk2utt=ark:$sdata/JOB/spk2utt $srcdir/global_cmvn.stats scp:$sdata/JOB/feats.scp ark:- |"
  fi

  if [ ! -z "$online_ivector_dir" ]; then
    ivector_period=$(cat $online_ivector_dir/ivector_period) || exit 1;
    ivector_opts="--online-ivectors=scp:$online_ivector_dir/ivector_online.scp --online-ivector-period=$ivector_period"
  fi

  frame_subsampling_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    # e.g. for 'chain' systems
    frame_subsampling_opt="--frame-subsampling-factor=$(cat $srcdir/frame_subsampling_factor)"
  fi

  $cmd $queue_opt JOB=1:$nj $dir/log/decode.JOB.log \
    nnet3-compute$batch_string $ivector_opts $frame_subsampling_opt \
     --frames-per-chunk=$frames_per_chunk \
     --extra-left-context=$extra_left_context \
     --extra-right-context=$extra_right_context \
     --extra-left-context-initial=$extra_left_context_initial \
     --extra-right-context-final=$extra_right_context_final \
     --acoustic-scale=$acwt --use-gpu=$use_gpu \
     --use-priors=true \
     "$model" "$feats" "$posteriors" || exit 1;
fi


if [ $stage -le 1 ]; then
  if [ "$post_decode_acwt" == 1.0 ]; then
    lat_wspecifier="ark:|gzip -c >$dir/lat.JOB.gz"
  else
    lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/lat.JOB.gz"
  fi
  # Change it to 'latgen-faster-mapped-mapversion'
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
    latgen-faster-mapped --acoustic-scale=$acwt --allow-partial=true \
      --beam=$beam --lattice-beam=$lattice_beam --max-active=$max_active \
      --min-active=$min_active --word-symbol-table=$graphdir/words.txt \
      $srcdir/final.mdl $graphdir/HCLG.fst "$posteriors_rspecifier" "$lat_wspecifier" || exit 1;
fi

if [ $stage -le 2 ]; then
  if ! $skip_diagnostics ; then
    [ ! -z $iter ] && iter_opt="--iter $iter"
    steps/diagnostic/analyze_lats.sh --cmd "$cmd" $iter_opt $graphdir $dir
  fi
fi


# The output of this script is the files "lat.*.gz"-- we'll rescore this at
# different acoustic scales to get the final output.
if [ $stage -le 3 ]; then
  if ! $skip_scoring ; then
    [ ! -x local/score.sh ] && \
      echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
    echo "score best paths"
    [ "$iter" != "final" ] && iter_opt="--iter $iter"
    local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
    echo "score confidence and timing with sclite"
  fi
fi
echo "Decoding done."
exit 0;
