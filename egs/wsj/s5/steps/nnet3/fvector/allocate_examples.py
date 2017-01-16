#!/usr/bin/env python

# This script, for use when training fvectors, decides for you which examples
# will come from which utterances, and at what point.

# You call it as (e.g.)
#
#  allocate_examples.py --frames-per-chunk=200  --frames-per-iter=1000000 \
#   --num-archives=169 --num-jobs=24  exp/xvector_a/egs/temp/utt2len.train exp/xvector_a/egs
#
# and this program outputs certain things to the temp directory (exp/xvector_a/egs/temp in this case)
# that will enable you to dump the chunks for xvector training.  What we'll eventually be doing is invoking
# the following program with something like the following args:
#
#  nnet3-fvector-get-egs [options] exp/xvector_a/temp/ranges.1  scp:data/train/feats.scp \
#    ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark \
#    ark:exp/xvector_a/egs/egs_temp.3.ark
#
# where exp/xvector_a/temp/ranges.1 contains something like the following:
#
#   <utt{i}-p{j}> <utt{i}-p{k}> 0 1 50 200
#
# where each line is interpreted as follows:
#  <source-utterance1> <source-utterance2> <relative-archive-index> <absolute-archive-index> <offset> <frame-length>
#
#  Note: <relative-archive-index> is the zero-based offset of the archive-index
# within the subset of archives that a particular ranges file corresponds to;
# and <absolute-archive-index> is the 1-based numeric index of the destination
# archive among the entire list of archives, which will form part of the
# archive's filename (e.g. egs/egs.<absolute-archive-index>.ark);
# <absolute-archive-index> is only kept for debug purposes so you can see which
# archive each line corresponds to.
#
# The list of archives corresponding to ranges.n will be written to output.n, 
# so in exp/xvector_a/temp/outputs.1 we'd have:
#
#  ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark ark:exp/xvector_a/egs/egs_temp.3.ark
#
# The number of these files will equal 'num-jobs'.  If you add up the word-counts of
# all the outputs.* files you'll get 'num-archives'.  The number of frames in each archive
# will be about the --frames-per-iter.
#

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, random


parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and archive_chunk_lengths files "
                                 "in preparation for dumping egs for xvector training.",
                                 epilog="Called by steps/nnet3/xvector/get_egs.sh")
parser.add_argument("--prefix", type=str, default="",
                   help="Adds a prefix to the output files. This is used to distinguish between the train "
                   "and diagnostic files.")
parser.add_argument("--frames-per-chunk", type=int, default=100,
                    help="The number of frames-per-chunk used for any archive")
parser.add_argument("--frames-per-iter", type=int, default=1000000,
                    help="Target number of frames for each archive")
parser.add_argument("--num-archives", type=int, default=-1,
                    help="Number of archives to write")
parser.add_argument("--num-jobs", type=int, default=-1,
                    help="Number of jobs we're going to use to write the archives; the ranges.* "
                    "and outputs.* files are indexed by job.  Must be <= the --num-archives option.")
parser.add_argument("--seed", type=int, default=1,
                    help="Seed for random number generator")

# now the positional arguments
parser.add_argument("utt2len",
                    help="utt2len file of the features to be used as input (format is: "
                    "<utterance-id> <approx-num-frames>)")
parser.add_argument("oriutt2allutt",
                    help="oriutt2allutt to be used as input (format is: "
                    "<ori-utt-id> <ori-utt-id> <p1-utt-id> ... <pn-utt-id>)")
parser.add_argument("egs_dir",
                    help="Name of egs directory, e.g. exp/xvector_a/egs")

print(' '.join(sys.argv))

args = parser.parse_args()

if not os.path.exists(args.egs_dir + "/temp"):
    os.makedirs(args.egs_dir + "/temp")

## Check arguments.
if args.frames_per_chunk <= 1:
    sys.exit("--frames-per-chunk is invalid.")
if args.frames_per_iter < 1000:
    sys.exit("--frames-per-iter is invalid.")
if args.num_archives < 1:
    sys.exit("--num-archives is invalid")
if args.num_jobs > args.num_archives:
    sys.exit("--num-jobs is invalid (must not exceed num-archives)")

random.seed(args.seed)

f = open(args.utt2len, "r");
if f is None:
    sys.exit("Error opening utt2len file " + str(args.utt2len));
utt_ids = []
lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in utt2len file " + line);
    utt_ids.append(a[0])
    lengths.append(int(a[1]))
f.close()

num_utts = len(utt_ids)
max_length = max(lengths)

if args.frames_per_chunk * 3 > max_length:
    sys.exit("--max-frames-per-chunk={0} is not valid: it must be no more "
             "than a third of the maximum length {1} from the utt2len file ".format(
            args.max_frames_per_chunk, max_length))

# create the map form ori-utt-id to all kinds of utt-id. The ori-utt-id is the 
# index, which is same with the elements in utt_ids[]
f = open(args.oriutt2allutt, "r");
if f is None:
    sys.exit("Error opening oriutt2allutt file " + str(args.oriutt2allutt));
utt_map = {}
for line in f:
    a = line.split()
    if len(a) < 3:
        sys.exit("bad line in oriutt2allutt file " + line);
    tmp_list = []
    for i in range(1, len(a)):
        tmp_list.append(a[i])
    tuple_list = tuple(tmp_list)
    utt_map[a[0]]=tuple_list
f.close()

    
# this function returns a random integer utterance index, limited to utterances
# above a minimum length in frames, with probability proportional to its length.
def RandomUttAtLeastThisLong(min_length):
    while True:
        i = random.randrange(0, num_utts)
        # read the next line as 'with probability lengths[i] / max_length'.
        # this allows us to draw utterances with probability with
        # prob proportional to their length.
        if lengths[i] > min_length and random.random() < lengths[i] / float(max_length):
            return i


# given an utterance length utt_length (in frames) and two desired chunk lengths
# (length1 and length2) whose sum is <= utt_length,
# this function randomly picks the starting points of the chunks for you.
# the chunks may appear randomly in either order.
def GetRandomOffsets(utt_length, length):
    if length > utt_length:
        sys.exit("code error: tot-length > utt-length")
    free_length = utt_length - length
    offset = random.randrange(0, free_length + 1)
    return offset


# this function randomly choose two utt-id form utt_map depending on ori-utt-id
def ChoosePairs(ori_utt_id):
    this_tuple = utt_map[ori_utt_id]
    while True:
        first_index = random.randint(0, len(this_tuple) - 1)
        second_index = random.randint(0, len(this_tuple) - 1)
        if first_index != second_index:
            break
    utt_a = this_tuple[first_index]
    utt_b = this_tuple[second_index]
    return (utt_a, utt_b)


# each element of all_egs (one per archive) is
# an array of 2-tuples (utterance-index, offset)
all_egs= []

prefix = ""
if args.prefix != "":
  prefix = args.prefix + "_"

for archive_index in range(args.num_archives):
    tot_length = 2 * args.frames_per_chunk
    this_num_egs = (args.frames_per_iter / tot_length) + 1
    this_egs = [ ] # this will be an array of 2-tuples (utterance-index, start-frame).
    for n in range(this_num_egs):
        utt_index = RandomUttAtLeastThisLong(args.frames_per_chunk)
        utt_len = lengths[utt_index]
        offset = GetRandomOffsets(utt_len, args.frames_per_chunk)
        this_egs.append( (utt_index, offset) )
    all_egs.append(this_egs)

# work out how many archives we assign to each job in an equitable way.
num_archives_per_job = [ 0 ] * args.num_jobs
for i in range(0, args.num_archives):
    num_archives_per_job[i % args.num_jobs]  = num_archives_per_job[i % args.num_jobs] + 1


cur_archive = 0
for job in range(args.num_jobs):
    this_ranges = []
    this_archives_for_job = []
    this_num_archives = num_archives_per_job[job]

    for i in range(0, this_num_archives):
        this_archives_for_job.append(cur_archive)
        for (utterance_index, offset) in all_egs[cur_archive]:
            this_ranges.append( (utterance_index, i, offset) )
        cur_archive = cur_archive + 1
    f = open(args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1), "w")
    if f is None:
        sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1))
    for (utterance_index, i, offset) in sorted(this_ranges):
        archive_index = this_archives_for_job[i]
        this_utt_id = utt_ids[utterance_index]
        #Random select two utt-id
        (utt_a, utt_b) = ChoosePairs(this_utt_id)
        print("{0} {1} {2} {3} {4} {5}".format(utt_a,
                                           utt_b,
                                           i,
                                           archive_index + 1,
                                           offset
                                           args.frames_per_chunk,
              file=f)
    f.close()

    f = open(args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1), "w")
    if f is None:
        sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1))
    print( " ".join([ str("{0}/" + prefix + "egs_temp.{1}.ark").format(args.egs_dir, n + 1) for n in this_archives_for_job ]),
           file=f)
    f.close()


print("allocate_examples.py: finished generating " + prefix + "ranges.* and " + prefix + "outputs.* files")

