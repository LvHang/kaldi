#!/bin/bash

stage=0
cmd=run.pl
nj=60
data=data/swbd_sre_combined_no_sil
dir=exp/xvector_nnet_10kinds/egs

# some options
num_heldout_utts=1000      # number of utterances held out for training subset
                           # and validation set
min_frames_per_chunk=200
max_frames_per_chunk=400

frames_per_kind=1043000000 # 1 billion * 10 ~= David's setup. For details, please
                           # check the specification of allocate_egs_seg_v2.py.
                           # frames_per_kind * kinds_of_length_train = total_frames

kinds_of_length_train=10   # number of the kinds of lengths in train set.

kinds_of_length_valid=3    # number of distinct lengths in training subset and
                           # validation set
num_train_archives=149     # match David's setup
# end of options


. ./path.sh
set -e


for f in $data/utt2spk $data/utt2num_frames $data/feats.scp ; do
  [ ! -f $f ] && echo "$0: expected file $f" && exit 1;
done

feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;

mkdir -p $dir/log $dir/info $dir/temp $dir/train_set/temp $dir/valid_set/temp $dir/train_subset/temp
temp=$dir/temp

echo $feat_dim > $dir/info/feat_dim
echo '0' > $dir/info/left_context
echo $min_frames_per_chunk > $dir/info/right_context
echo '1' > $dir/info/frames_per_eg

if [ $stage -le 0 ]; then
  echo "$0: Preparing train and validation lists"
  # Pick a list of heldout utterances for validation egs
  cat $data/utt2spk | utils/shuffle_list.pl | head -$num_heldout_utts > $dir/valid_set/utt2spk || exit 1;
  cp $dir/valid_set/utt2spk $temp/uttlist_valid
  utils/filter_scp.pl $dir/valid_set/utt2spk $data/utt2num_frames > $dir/valid_set/utt2num_frames
  utils/filter_scp.pl $dir/valid_set/utt2spk $data/feats.scp > $dir/valid_set/feats.scp

  # The remaining utterances are used for training egs
  utils/filter_scp.pl --exclude $dir/valid_set/utt2spk $data/utt2spk > $dir/train_set/utt2spk
  cp $dir/train_set/utt2spk $temp/uttlist_train
  utils/filter_scp.pl --exclude $dir/valid_set/utt2spk $data/utt2num_frames > $dir/train_set/utt2num_frames
  utils/filter_scp.pl --exclude $dir/valid_set/utt2spk $data/feats.scp > $dir/train_set/feats.scp

  # Pick a subset of the training list for diagnostics
  cat $dir/train_set/utt2spk | utils/shuffle_list.pl | head -$num_heldout_utts > $dir/train_subset/utt2spk || exit 1;
  cp $dir/train_subset/utt2spk $temp/uttlist_train_subset
  utils/filter_scp.pl $temp/uttlist_train_subset <$data/utt2num_frames > $dir/train_subset/utt2num_frames
  utils/filter_scp.pl $temp/uttlist_train_subset <$data/feats.scp > $dir/train_subset/feats.scp

  # Create a mapping from utterance to speaker ID (an integer)
  awk -v id=0 '{print $1, id++}' $data/spk2utt > $temp/spk2int
  utils/sym2int.pl -f 2 $temp/spk2int $data/utt2spk > $temp/utt2int
  utils/filter_scp.pl $dir/train_set/utt2spk $temp/utt2int > $dir/train_set/utt2int
  utils/filter_scp.pl $dir/valid_set/utt2spk $temp/utt2int > $dir/valid_set/utt2int
  utils/filter_scp.pl $dir/train_subset/utt2spk $temp/utt2int > $dir/train_subset/utt2int
  #Above, prepare the "utt2num_frames, utt2spk and utt2int" for each data set.
fi

num_pdfs=$(awk '{print $2}' $temp/utt2int | sort | uniq -c | wc -l)
num_train_frames=$(awk '{n += $2} END{print n}' <$dir/train_set/utt2num_frames)
num_train_subset_frames=$(awk '{n += $2} END{print n}' <$dir/train_subset/utt2num_frames)
#For here, the options are hardcode as we want to compare with the result of David
echo $num_train_frames > $dir/info/num_frames
echo $num_train_archives > $dir/info/num_archives
echo $num_pdfs > $dir/info/num_pdfs

if [ $nj -gt $num_train_archives ]; then
  echo "$0: Reducing num-jobs $nj to number of training archives $num_train_archives"
  nj=$num_train_archives
fi
#In David's setup, there are 149 archives. Each archives contains 70 million frames.
#There are about 100 kinds of different length
#So each "segment length" should have 70million frames * 149 / 100 = 104.3million

temp_dir=$dir/train_set/temp
if [ $stage -le 1 ]; then
  echo "$0: Allocating training egs"
  #Generate data_li/feat.scp and utt2spk
  $cmd $dir/log/allocate_examples.log \
    sid/nnet3/xvector/allocate_egs_seg_v2.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --kinds-of-length=$kinds_of_length_train \
      --frames-per-kind=$frames_per_kind \
      --data-dir=$dir/train_set \
      --output-dir=$temp_dir || exit 1
  #fix dir and generate the 'utt2int' file.
  for subdir in `dir $temp_dir`; do
    utils/fix_data_dir.sh $temp_dir/$subdir
    utils/sym2int.pl -f 2 $data/spk2int $temp_dir/$subdir/utt2spk \
      > $temp_dir/$subdir/utt2int
  done
fi

if [ $stage -le 2 ]; then
  echo "$0: Combine, shuffle and split list for training set"
  #combine feats.scp and shuffle
  mkdir -p $temp_dir/combine
  data_combine=$temp_dir/combine

  utils/combine_data.sh $data_combine $temp_dir/data_*
  utils/shuffle_list.pl $data_combine/feats.scp > $data_combine/shuffled_feats.scp
  utils/sym2int.pl -f 2 ${data}/spk2int ${data_combine}/utt2spk \
    > ${data_combine}/utt2int

  feat_scps=$(for n in `seq $num_train_archives`; do echo $data_combine/shuffled_feats.${n}.scp; done)
  utils/split_scp.pl ${data_combine}/shuffled_feats.scp $feat_scps
fi

if [ $stage -le 3 ]; then
  echo "$0: Dump Egs for training set"
  if [ -e $dir/storage ]; then
    echo "$0: creating data links"
    utils/create_data_link.pl $(for x in $(seq $num_train_archives); do echo $dir/egs.$x.ark; done)
  fi
  #Note: if --num-pdfs options is not supply, you must use global utt2int,
  #because we will compute the num-pdfs from utt2label file.
  $cmd --max-jobs-run $nj JOB=1:$num_train_archives $dir/log/dump_egs.JOB.log \
    nnet3-xvector-get-egs-seg \
      --num-pdfs=$num_pdfs \
      $temp_dir/combine/utt2int \
      scp:$temp_dir/combine/shuffled_feats.JOB.scp \
      ark,scp:$dir/egs.JOB.ark,$dir/egs.JOB.scp
fi

temp_dir=$dir/valid_set/temp
if [ $stage -le 4 ]; then
  echo "$0: Dealing with validation set"
  #Generate data_li/feat.scp and utt2spk
  $cmd $dir/log/allocate_examples_validate.log \
    sid/nnet3/xvector/allocate_egs_seg_v2.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --kinds-of-length=$kinds_of_length_valid \
      --generate-validate=true \
      --randomize-chunk-length=false \
      --data-dir=$dir/valid_set \
      --output-dir=$temp_dir || exit 1
  #Generate data_li/spk2utt and utt2label
  for subdir in `dir $temp_dir`; do
    utils/fix_data_dir.sh $temp_dir/$subdir
    utils/sym2int.pl -f 2 $data/spk2int $temp_dir/$subdir/utt2spk \
      > $temp_dir/$subdir/utt2int
  done

  echo "$0: combine and shuffle (valid)"
  #combine feats.scp and shuffle
  mkdir -p $temp_dir/combine
  data_combine=$temp_dir/combine

  utils/combine_data.sh $data_combine $temp_dir/data_*
  utils/shuffle_list.pl $data_combine/feats.scp > $data_combine/shuffled_feats.scp
  utils/sym2int.pl -f 2 ${data}/spk2int ${data_combine}/utt2spk \
    > ${data_combine}/utt2int

  feat_scps=$(for n in `seq $kinds_of_length_valid`; do echo $data_combine/shuffled_feats.${n}.scp; done)
  utils/split_scp.pl ${data_combine}/shuffled_feats.scp $feat_scps

  echo "$0: dump egs (valid)"
  #dump egs
  $cmd --max-jobs-run $nj JOB=1:$kinds_of_length_valid $dir/log/dump_egs_valid.JOB.log \
    nnet3-xvector-get-egs-seg \
      --num-pdfs=$num_pdfs \
      $temp_dir/combine/utt2int \
      scp:$temp_dir/combine/shuffled_feats.JOB.scp \
      ark,scp:$dir/valid_egs.JOB.ark,$dir/valid_egs.JOB.scp
  cat $dir/valid_egs.*.scp > $dir/valid_diagnostic.scp
fi

temp_dir=$dir/train_subset/temp
if [ $stage -le 5 ]; then
  echo "$0: Dealing with train_diagnostic set"
  #Generate data_li/feat.scp and utt2spk
  $cmd $dir/log/allocate_examples_diagnostic.log \
    sid/nnet3/xvector/allocate_egs_seg_v2.py \
      --min-frames-per-chunk=$min_frames_per_chunk \
      --max-frames-per-chunk=$max_frames_per_chunk \
      --kinds-of-length=$kinds_of_length_valid \
      --generate-validate=true \
      --randomize-chunk-length=false \
      --data-dir=$dir/train_subset \
      --output-dir=$temp_dir || exit 1
  #Generate data_li/spk2utt and utt2label
  for subdir in `dir $temp_dir`; do
    utils/fix_data_dir.sh $temp_dir/$subdir
    utils/sym2int.pl -f 2 $data/spk2int $temp_dir/$subdir/utt2spk \
      > $temp_dir/$subdir/utt2int
  done

  echo "$0: combine and shuffle (diagnostic)"
  #combine feats.scp and shuffle
  mkdir -p $temp_dir/combine
  data_combine=$temp_dir/combine

  utils/combine_data.sh $data_combine $temp_dir/data_*
  utils/shuffle_list.pl $data_combine/feats.scp > $data_combine/shuffled_feats.scp
  utils/sym2int.pl -f 2 ${data}/spk2int ${data_combine}/utt2spk \
    > ${data_combine}/utt2int

  feat_scps=$(for n in `seq $kinds_of_length_valid`; do echo $data_combine/shuffled_feats.${n}.scp; done)
  utils/split_scp.pl ${data_combine}/shuffled_feats.scp $feat_scps

  echo "$0: dump egs (diagnostic)"
  #dump egs
  $train_cmd --max-jobs-run $nj JOB=1:$kinds_of_length_valid $dir/log/dump_egs_diagnostic.JOB.log \
    nnet3-xvector-get-egs-seg \
      --num-pdfs=$num_pdfs \
      $temp_dir/combine/utt2int \
      scp:$temp_dir/combine/shuffled_feats.JOB.scp \
      ark,scp:$dir/train_diagnostic_egs.JOB.ark,$dir/train_diagnostic_egs.JOB.scp
  cat $dir/train_diagnostic_egs.*.scp > $dir/train_diagnostic.scp
  ln -s $dir/train_diagnostic.scp combine.scp
fi
