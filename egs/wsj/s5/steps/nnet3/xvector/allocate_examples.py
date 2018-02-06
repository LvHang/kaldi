#!/usr/bin/env python

# This script, for use when training xvectors, decides for you which examples
# will come from which utterances, and at what point.

# You call it as (e.g.)
#
#  allocate_examples.py --min-frames-per-chunk=50 --max-frames-per-chunk=200  --frames-per-iter=1000000 \
#   --num-archives=169 --num-jobs=24  exp/xvector_a/egs/temp/utt2len.train exp/xvector_a/egs
#
# and this program outputs certain things to the temp directory (exp/xvector_a/egs/temp in this case)
# that will enable you to dump the chunks for xvector training.  What we'll eventually be doing is invoking
# the following program with something like the following args:
#
#  nnet3-xvector-get-egs [options] exp/xvector_a/temp/ranges.1  scp:data/train/feats.scp \
#    ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark \
#    ark:exp/xvector_a/egs/egs_temp.3.ark
#
# where exp/xvector_a/temp/ranges.1 contains something like the following:
#
#   utt1  3  0  0   65  112  110
#   utt1  0  2  160 50  214  180
#   utt2  ...
#
# where each line is interpreted as follows:
#  <source-utterance> <relative-archive-index> <absolute-archive-index> <start-frame-index1> <num-frames1> <start-frame-index2> <num-frames2>
#
#  Note: <relative-archive-index> is the zero-based offset of the archive-index
# within the subset of archives that a particular ranges file corresponds to;
# and <absolute-archive-index> is the 1-based numeric index of the destination
# archive among the entire list of archives, which will form part of the
# archive's filename (e.g. egs/egs.<absolute-archive-index>.ark);
# <absolute-archive-index> is only kept for debug purposes so you can see which
# archive each line corresponds to.
#
# and for each line we create an eg (containing two possibly-different-length chunks of data from the
# same utterance), to one of the output archives.  The list of archives corresponding to
# ranges.n will be written to output.n, so in exp/xvector_a/temp/outputs.1 we'd have:
#
#  ark:exp/xvector_a/egs/egs_temp.1.ark ark:exp/xvector_a/egs/egs_temp.2.ark ark:exp/xvector_a/egs/egs_temp.3.ark
#
# The number of these files will equal 'num-jobs'.  If you add up the word-counts of
# all the outputs.* files you'll get 'num-archives'.  The number of frames in each archive
# will be about the --frames-per-iter.
#
# This program will also output to the temp directory a file called archive_chunk_lengths which gives you
# the pairs of frame-lengths associated with each archives. e.g.
# 1   60  180
# 2   120  85
# the format is:  <archive-index> <num-frames1> <num-frames2>.
# the <num-frames1> and <num-frames2> will always be in the range
# [min-frames-per-chunk, max-frames-per-chunk].



# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, random


parser = argparse.ArgumentParser(description="Writes ranges.*, outputs.* and archive_chunk_lengths files "
                                 "in preparation for dumping egs for xvector training.",
                                 epilog="Called by steps/nnet3/xvector/get_egs.sh")
parser.add_argument("--prefix", type=str, default="",
                   help="Adds a prefix to the output files. This is used to distinguish between the train "
                   "and diagnostic files.")
parser.add_argument("--min-frames-per-chunk", type=int, default=50,
                    help="Minimum number of frames-per-chunk used for any archive")
parser.add_argument("--max-frames-per-chunk", type=int, default=300,
                    help="Maximum number of frames-per-chunk used for any archive")
parser.add_argument("--randomize-chunk-length", type=str,
                    help="If true, randomly pick a chunk length in [min-frames-per-chunk, max-frames-per-chunk]."
                    "If false, the chunk length varies from min-frames-per-chunk to max-frames-per-chunk"
                    "according to a geometric sequence.",
                    default="true", choices = ["false", "true"])
parser.add_argument("--frames-per-iter", type=int, default=1000000,
                    help="Target number of frames for each archive")
parser.add_argument("--num-archives", type=int, default=-1,
                    help="Number of archives to write");
parser.add_argument("--num-jobs", type=int, default=-1,
                    help="Number of jobs we're going to use to write the archives; the ranges.* "
                    "and outputs.* files are indexed by job.  Must be <= the --num-archives option.");
parser.add_argument("--seed", type=int, default=1,
                    help="Seed for random number generator")

# now the positional arguments
parser.add_argument("utt2len",
                    help="utt2len file of the features to be used as input (format is: "
                    "<utterance-id> <approx-num-frames>)");
parser.add_argument("egs_dir",
                    help="Name of egs directory, e.g. exp/xvector_a/egs");

print(' '.join(sys.argv))

args = parser.parse_args()

if not os.path.exists(args.egs_dir + "/temp"):
    os.makedirs(args.egs_dir + "/temp")

## Check arguments.
if args.min_frames_per_chunk <= 1:
    sys.exit("--min-frames-per-chunk is invalid.")
if args.max_frames_per_chunk < args.min_frames_per_chunk:
    sys.exit("--max-frames-per-chunk is invalid.")
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

if args.max_frames_per_chunk * 3 > max_length:
    sys.exit("--max-frames-per-chunk={0} is not valid: it must be no more "
             "than a third of the maximum length {1} from the utt2len file ".format(
            args.max_frames_per_chunk, max_length))

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

# this function returns a random integer drawn from the range
# [min-frames-per-chunk, max-frames-per-chunk], but distributed log-uniform.
def RandomChunkLength():
    log_value = (math.log(args.min_frames_per_chunk) +
                 random.random() * math.log(args.max_frames_per_chunk /
                                            args.min_frames_per_chunk))
    ans = int(math.exp(log_value) + 0.45)
    return ans

# This function returns an integer in the range
# [min-frames-per-chunk, max-frames-per-chunk] according to a geometric
# sequence. For example, suppose min-frames-per-chunk is 50,
# max-frames-per-chunk is 200, and args.num_archives is 3. Then the
# lengths for archives 0, 1, and 2 will be 50, 100, and 200.
def DeterministicChunkLength(archive_id):
  ans = int(math.pow(float(args.max_frames_per_chunk) /
                     args.min_frames_per_chunk, float(archive_id) /
                     (args.num_archives-1)) * args.min_frames_per_chunk + 0.5)
  return ans



# given an utterance length utt_length (in frames) and two desired chunk lengths
# (length1 and length2) whose sum is <= utt_length,
# this function randomly picks the starting points of the chunks for you.
# the chunks may appear randomly in either order.
def GetRandomOffsets(utt_length, length1, length2):
    tot_length = length1 + length2
    if tot_length > utt_length:
        sys.exit("code error: tot-length > utt-length")
    free_length = utt_length - tot_length

    # We want to randomly divide free_length into 3 pieces a, b, c
    # so that we'll have:
    # utt_length = a + length1 + b + length2 + c
    # or
    # utt_length = a + length2 + b + length1 + c
    # where the order shows you in what order the pieces come (note:
    # we may randomly switch the left and right chunks later on.
    # so we want random a,b,c such that a + b + c = free_length;
    # we can achieve this elegantly as follows:
    while True:
        a = random.randrange(0, free_length + 1)
        b = random.randrange(0, free_length + 1)
        c = free_length - a - b
        if c >= 0:
            break
    if random.random() < 0.5: # chunk with length1 is earlier
        offset1 = a
        offset2 = a + length1 + b
    else:            # chunk with length2 is earlier
        offset2 = a
        offset1 = a + length2 + b
    return (offset1, offset2)


# archive_chunk_lengths and all_archives will be arrays of dimension
# args.num_archives.  archive_chunk_lengths contains 2-tuples
# (left-num-frames, right-num-frames).
archive_chunk_lengths = []  # archive
# each element of all_egs (one per archive) is
# an array of 3-tuples (utterance-index, offset1, offset2)
all_egs= []

prefix = ""
if args.prefix != "":
  prefix = args.prefix + "_"

info_f = open(args.egs_dir + "/temp/" + prefix + "archive_chunk_lengths", "w")
if info_f is None:
    sys.exit(str("Error opening file {0}/temp/" + prefix + "archive_chunk_lengths").format(args.egs_dir));
for archive_index in range(args.num_archives):
    print("Processing archive {0}".format(archive_index + 1))
    if args.randomize_chunk_length == "true":
        # don't constrain the lengths to be the same
        length1 = RandomChunkLength();
        length2 = RandomChunkLength();
    else:
        length1 = DeterministicChunkLength(archive_index);
        length2 = length1
    print("{0} {1} {2}".format(archive_index + 1, length1, length2), file=info_f)
    archive_chunk_lengths.append( (length1, length2) )
    tot_length = length1 + length2
    this_num_egs = (args.frames_per_iter / tot_length) + 1
    this_egs = [ ] # this will be an array of 3-tuples (utterance-index, left-start-frame, right-start-frame).
    for n in range(this_num_egs):
        utt_index = RandomUttAtLeastThisLong(tot_length)
        utt_len = lengths[utt_index]
        (offset1, offset2) = GetRandomOffsets(utt_len, length1, length2)
        this_egs.append( (utt_index, offset1, offset2) )
    all_egs.append(this_egs)
info_f.close()

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
        for (utterance_index, offset1, offset2) in all_egs[cur_archive]:
            this_ranges.append( (utterance_index, i, offset1, offset2) )
        cur_archive = cur_archive + 1
    f = open(args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1), "w")
    if f is None:
        sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1))
    for (utterance_index, i, offset1, offset2) in sorted(this_ranges):
        archive_index = this_archives_for_job[i]
        print("{0} {1} {2} {3} {4} {5} {6}".format(utt_ids[utterance_index],
                                           i,
                                           archive_index + 1,
                                           offset1,
                                           archive_chunk_lengths[archive_index][0],
                                           offset2,
                                           archive_chunk_lengths[archive_index][1]),
              file=f)
    f.close()

    f = open(args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1), "w")
    if f is None:
        sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "outputs." + str(job + 1))
    print( " ".join([ str("{0}/" + prefix + "egs_temp.{1}.ark").format(args.egs_dir, n + 1) for n in this_archives_for_job ]),
           file=f)
    f.close()


print("allocate_examples.py: finished generating " + prefix + "ranges.* and " + prefix + "outputs.* files")

