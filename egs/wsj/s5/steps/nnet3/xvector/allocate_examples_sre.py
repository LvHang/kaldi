#!/usr/bin/env python

# TODO

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
parser.add_argument("--unique-recordings", type=str,
                    help="TODO",
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
parser.add_argument("utt2reco",
                    help="utt2reco TODO");

parser.add_argument("utt2spk",
                    help="utt2spk TODO");

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
    sys.exit("Error opening utt2len file " + str(args.utt2len))
utt_ids = []
lengths = []
utt2len = {}
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in utt2len file " + line)
    utt_ids.append(a[0])
    lengths.append(int(a[1]))
    utt2len[a[0]] = int(a[1])
f.close()

num_utts = len(utt_ids)
max_length = max(lengths)

f = open(args.utt2reco, "r");
if f is None:
    sys.exit("Error opening utt2reco file " + str(args.utt2reco))
utt2reco = {}
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in utt2reco file " + line)
    utt2reco[a[0]] = a[1]
f.close()

f = open(args.utt2spk, "r");
if f is None:
    sys.exit("Error opening utt2spk file " + str(args.utt2spk))
utt2spk = {}
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in utt2spk file " + line)
    utt2spk[a[0]] = a[1]
f.close()

reco2utt = {}
spk2utt = {}

for utt in utt2spk:
  spk = utt2spk[utt]
  if spk not in spk2utt:
    spk2utt[spk] = [utt]
  else:
    spk2utt[spk].append(utt)

for utt in utt2reco:
  reco = utt2reco[utt]
  if reco not in reco2utt:
    reco2utt[reco] = [utt]
  else:
    reco2utt[reco].append(utt)

spk2reco = {}
for spk in spk2utt:
  spk2reco[spk] = []
  for utt in spk2utt[spk]:
     reco = utt2reco[utt]
     if reco not in spk2reco[spk]:
       spk2reco[spk].append(reco)

spk2maxlen = {}
for spk in spk2utt:
  spk2maxlen[spk] = -1
  for utt in spk2utt[spk]:
    if utt2len[utt] > spk2maxlen[spk]:
      spk2maxlen[spk] = utt2len[utt]

reco2maxlen = {}
for reco in reco2utt:
  reco2maxlen[reco] = -1
  for utt in reco2utt[reco]:
    if utt2len[utt] > reco2maxlen[reco]:
      reco2maxlen[reco] = utt2len[utt]


num_reco = len(reco2utt.keys())
num_spk = len(spk2utt.keys())


if args.max_frames_per_chunk > max_length:
    sys.exit("--max-frames-per-chunk={0} is not valid: it must be no more "
             "than the maximum length {1} from the utt2len file ".format(
            args.max_frames_per_chunk, max_length))

# TODO
def RandomUttAtLeastThisLong(spkr, exclude_reco, min_length):
    #print("Top of random utt")
    if exclude_reco != None:
        this_utts_set = set(spk2utt[spkr]) - set(reco2utt[exclude_reco])
    else:
        this_utts_set = set(spk2utt[spkr])
    this_utts = list(this_utts_set)
    this_num_utts = len(this_utts)
    this_max_length = spk2maxlen[spkr]
    while True:
        i = random.randint(0, this_num_utts-1)
        #print("i={0}".format(i))
        utt = this_utts[i]
        #print("utt={0}".format(utt))

        # read the next line as 'with probability lengths[i] / max_length'.
        # this allows us to draw utterances with probability with
        # prob proportional to their length.
        #print("utt2len={0}  min_length={1} max_length={2}".format(utt2len[utt], min_length, this_max_length))
        if utt2len[utt] > min_length and random.random() < utt2len[utt] / float(this_max_length):
            #print("in return")
            return utt

#TODO
def RandomSpkr(exclude_spkrs, max_len):
  too_short = []
  for spk in spk2maxlen:
    if spk2maxlen[spk] < max_len:
      too_short.append(spk)

  spkr_set = set(spk2utt.keys()) - set(exclude_spkrs)
  spkr_set = spkr_set - set(too_short)

  return random.sample(spkr_set, 1)[0]


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
  if args.max_frames_per_chunk == args.min_frames_per_chunk:
    return args.max_frames_per_chunk
  elif args.num_archives == 1:
    return int(args.max_frames_per_chunk);
  else:
    return int(math.pow(float(args.max_frames_per_chunk) /
                     args.min_frames_per_chunk, float(archive_id) /
                     (args.num_archives-1)) * args.min_frames_per_chunk + 0.5)


# given an utterance length utt_length (in frames) and two desired chunk lengths
# (length1 and length2) whose sum is <= utt_length,
# this function randomly picks the starting points of the chunks for you.
# the chunks may appear randomly in either order.
def GetRandomOffset(utt_length, length):
    if length > utt_length:
        sys.exit("code error: length > utt-length")
    free_length = utt_length - length

    offset = random.randint(0, free_length)
    return offset

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

singletons = []
for spk in spk2reco:
  if len(spk2reco[spk]) == 1:
    singletons.append(spk)

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
    this_egs = [ ] # this will be an array of 4-tuples (utt1, utt2, left-start-frame, right-start-frame).
    exclude_spkrs = []
    exclude_spkrs.extend(singletons)
    spk2num_reco_long = {}
    for spk in spk2reco:
      spk2num_reco_long[spk] = 0
      for reco in spk2reco[spk]:
        if reco2maxlen[reco] > max(length1, length2):
         #print("nax len for reco: " + reco + " is " + str(reco2maxlen[reco]))
         # print("spk is " + spk)
          spk2num_reco_long[spk] += 1

    if args.unique_recordings == "true":
      for spk in spk2num_reco_long:
        if spk2num_reco_long[spk] < 2:
          exclude_spkrs.append(spk)
    else:
      for spk in spk2num_reco_long:
        if spk2num_reco_long[spk] < 1:
          exclude_spkrs.append(spk)

    exclude_spkrs = list(set(exclude_spkrs))
    for n in range(this_num_egs):
        print("len(exclude_spkrs) = " + str(len(exclude_spkrs)))
        print("len(spk2utt.keys()) = " + str(len(spk2utt.keys())))
        if len(exclude_spkrs) >= len(spk2utt.keys()):
            print("Run out of speakers")
            break

        spkr = RandomSpkr(exclude_spkrs, max(length1, length2))
        exclude_spkrs.append(spkr)
        utt1 = RandomUttAtLeastThisLong(spkr, None, length1)
        if args.unique_recordings == "true":
          exclude_reco = utt2reco[utt1]
        else:
          exclude_reco = None

        utt2 = RandomUttAtLeastThisLong(spkr, exclude_reco, length2)
        #utt2 = RandomUttAtLeastThisLong(spkr, None, length2)
        utt_len1 = utt2len[utt1]
        utt_len2 = utt2len[utt2]

        offset1 = GetRandomOffset(utt_len1, length1)
        offset2 = GetRandomOffset(utt_len2, length2)

        this_egs.append( (utt1, utt2, offset1, offset2) )
        print("utt1 = " + str(utt1) + " utt2 = " + str(utt2))
    all_egs.append(this_egs)
info_f.close()

# work out how many archives we assign to each job in an equitable way.
num_archives_per_job = [ 0 ] * args.num_jobs
for i in range(0, args.num_archives):
    num_archives_per_job[i % args.num_jobs] = num_archives_per_job[i % args.num_jobs] + 1

cur_archive = 0
for job in range(args.num_jobs):
    this_ranges = []
    this_archives_for_job = []
    this_num_archives = num_archives_per_job[job]

    for i in range(0, this_num_archives):
        this_archives_for_job.append(cur_archive)
        for (utt1, utt2, offset1, offset2) in all_egs[cur_archive]:
            this_ranges.append( (utt1, utt2, i, offset1, offset2) )
        cur_archive = cur_archive + 1
    f = open(args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1), "w")
    if f is None:
        sys.exit("Error opening file " + args.egs_dir + "/temp/" + prefix + "ranges." + str(job + 1))
    for (utt1, utt2, i, offset1, offset2) in sorted(this_ranges):
        archive_index = this_archives_for_job[i]
        print("{0} {1} {2} {3} {4} {5} {6} {7}".format(utt1,
                                           utt2,
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

