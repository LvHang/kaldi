#!/bin/bash
# Copyright 2016   David Snyder
# Apache 2.0.
#
# TODO

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
trials=data/sre10_test/trials

# NOTE: Suppose you're testing on SRE10
local/make_sre_2010_test.pl /export/corpora5/SRE/SRE2010/eval/ data/
local/make_sre_2010_train.pl /export/corpora5/SRE/SRE2010/eval/ data/

# NOTE: Suppose old NIST SREs are your training data.
local/make_sre.sh data

# NOTE: In the SLT paper we used 20 dim MFCCs. You could also try hires MFCCs.
# If using the hires MFCCs, you might want to find a way to avoid writing spliced
# features to the disk, otherwise they will be too big. One solution is to use
# a TDNN architecture instead of a feedforward DNN. But this makes it difficult
# to deal with nonspeech frames in a nice way.
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_train exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
  data/sre10_test exp/make_mfcc $mfccdir

for name in sre sre10_train sre10_test; do
  utils/fix_data_dir.sh data/${name}
done

# NOTE: Here we just use a simple energy-based VAD to identify speech frames. If your
# test data is noisy, you might want to try using something more powerful.
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_train exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
  data/sre10_test exp/make_vad $vaddir

for name in sre sre10_train sre10_test; do
  utils/fix_data_dir.sh data/${name}
done

# NOTE: The default options for this script demonstrates the features used in the SLT paper.
local/xvector/prepare_xvector_feats.sh --nj 40 \
  --cmd "$train_cmd --mem 6G" \
  data/sre data/xvector_sre exp/xvector_feats/

utils/fix_data_dir.sh data/xvector_sre
# These files get automatically created by scripts called in local/xvector/run_sre_a1.sh, so
# remove them if they already exist.
rm -rf data/xvector_sre/utt2len
rm -rf data/xvector_sre/utt2dur
rm -rf data/xvector_sre/utt2reco

# NOTE: This script shows the DNN configuration used in the SLT paper.
# Notice that this uses a regular feed-foward archtiecture, instead of the
# TDNN architecture you see in local/xvector/run_sre.sh.
# It's likely you'll have to play around with some of the options for creating egs,
# so that you don't have too many or too few archives. Also, you might need to
# adjust the minibatch size if you start running out of memory.
#
# Also, this script trains the DNN on 10-30s segments. If you're interested in 1-30s
# segments, you'll want to use the final model from here to initialize a new model
# that trains on 1-30s chunks (see script below for where to set that). It's hard to
# train the DNN directly on 1-30s seconds.
local/xvector/run_sre_a1.sh --feat-dim 180 --data data/xvector_sre

# TODO: Evaluation follows from that, can add this later.
