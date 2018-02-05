#!/bin/bash

# Copyright 2017 Pegah Ghahremani
# Apache 2.0

# This script evaluates age detection model trained using nnet3 model.

[ -f ./path.sh ] && . ./path.sh; # source the path.

. parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

if [ $# != 2 ]; then
  echo "Usage: $0 [options] <data-dir> <xvector-data>";
  echo "e.g.: $0 data/sre_08_10/1/test exp/xvector_age/1/test"
fi

data=$1
xvector_data=$2

for x in $xvector_data/spk_xvector.scp $xvector_data/xvector.scp $data/utt2age; do
  [ ! -f $x ] && echo "$0: file $x does not exist" && exit 1;
done

copy-vector scp:$xvector_data/xvector.scp ark,t:- | \
  awk '{max=$3;m_ind=0; for (i=3;i<NF;i++) if($i>max) {max=$i;m_ind=i-3 };print $1,m_ind}' > $xvector_data/utt2max

age2class=$data/age2class

#mean_abs_error=`paste <(awk '{print $3}' $xvector_data/utt2max) <(awk -v offset=$age_offset '{print $2-offset}' $data/utt2age) > $xvector_data/true.precicted.age
#cat $xvector_data/true.precicted.age | awk '{ sum +=( $1-$2 >= 0 ? $1-$2 : $2-$1 ); n++ } END { if (n > 0) print sum / n; }'`

# Generate mapping from class to min age correspond to class.
# generate class2age
./utils/utt2spk_to_spk2utt.pl $age2class > $data/class2age || exit 1;
utils/apply_map.pl -f 2 $data/class2age < $xvector_data/utt2max > $xvector_data/utt2predicted.age
paste $data/utt2age $xvector_data/utt2predicted.age > $xvector_data/tmp_f
awk '{print $2" "$4}' $xvector_data/tmp_f > $xvector_data/true.predicted.age
#paste <(awk '{print $2}' $data/utt2age) <(awk '{print $2}' $xvector_data/utt2predicted.age)  > $xvector_data/true.predicted.age
mean_abs_error=`cat $xvector_data/true.predicted.age | awk '{ sum +=( $1-$2 >= 0 ? $1-$2 : $2-$1 ); n++ } END { if (n > 0) print sum / n; }'`

echo "mean abs error |y_i - y'_i| is $mean_abs_error." > $xvector_data/mean_abs_error.txt

accuracy=`cat $xvector_data/true.predicted.age | awk '{ sum +=( $1==$2 ? 1 : 0 ); n++ } END { if (n > 0) print sum / n; }'`
tot_data=`wc -l $xvector_data/true.predicted.age | awk '{print $1}'`
echo "Overall accuracy is $accuracy" > $xvector_data/accuracy
num_classes=`wc -l $data/class2age | awk '{print $1}'`
for n in $(seq 0 $[$num_classes-1]);do
  accuracy=`cat $xvector_data/true.predicted.age | awk -v n=$n '{if ($1 == n) print $1" "$2}' | awk -v t=$tot_data '{ sum +=( $1==$2 ? 1 : 0 ); n++ } END { if (n > 0) print sum / n", "n / t", "n; }'`
  miclassified=`cat $xvector_data/true.predicted.age | awk -v n=$n '{if ($1 == n) print $2}' | sort | uniq -c | sed  's/^ *//g' | tr ' ' ':' | tr '\n' ' ,'`
  echo "accuracy, % of data, num of test data for class $n = $accuracy; (num-predicted utts with class label : class label: $miclassified)" >> $xvector_data/accuracy
done

