#!/usr/bin/env python

# The function use to generate range file for fvector
# This is the fixed-length version
# The format is <wav_t_start> <wav_t_end> <noise_uttid> <noise_t_start> <noise_t_end> <snr>

# For <wav_t_start> <wav_t_end>
# We except the last fragement, the length will be a fixed value T.

# For <noise_uttid>
# It is randomly selected from noise list, which is longer than --min-additive-noise-len

# For <noise_t_start> <noise_t_end>
# If the noise file is longer than fixed value. We randomly select the start point and 
# the length will be fixed value T.
# If the noise file is shorter than T. We select the whole noise.

# The <snr> control the rate of signal and noise. In the other word, scale the amplitude of noise.

from __future__ import print_function
import re, os, argparse, sys, math, warnings, random

parser = argparse.ArgumentParser(description="Generate N noise range files for each original wav. The file"
                                 "which created by this python code will be supplied to fixed-length add "
                                 "additive noise program.",
                                 epilog="Called by steps/nnet3/fvector/add_noise.sh")
parser.add_argument("--num-kind-range", type=int, default=4,
                    help="the number of noise range files")
parser.add_argument("--min-additive-noise-len", type=float, default=2.0,
                    help="the minimum duration/length of each noise file")
parser.add_argument("--min-snr", type=int, default=0,
                    help="the minimum Signal-to-Noise Rate, the default=0")
parser.add_argument("--max-snr", type=int, default=-10,
                    help="the maximum Signal-to-Noise Rate, the default=-10")
parser.add_argument("--seed", type=int, default=-1,
                    help="Seed for random number generator")

# now the positional arguments
parser.add_argument("wav_utt2dur",
                    help="utt2dur file of the original wav to be used as input (format is: "
                    "<utterance-id> <duration>")
parser.add_argument("noise_utt2dur",
                    help="utt2dur file of the noise wav to be used as input (format is: "
                    "<utterance-id> <duration>")
parser.add_argument("rangs_dir",
                    help="Name of ranges directory, exp/fxvector/ranges")

print(' '.join(sys.argv))

args = parser.parse_args()

## Check arguments
if args.min_snr < args.max_snr:
    sys.exit("For SNR, the less numerical value is, the larger noise is. So --min-snr bigger "
             "than --max-snr in numerical value.")

random.seed(args.seed)

# deal with the original wav utt2dur
# the information was stored in wav_utt_ids[], wav_lengths[] and wav_num_utts
f = open(args.wav_utt2dur, "r")
if f is None:
    sys.exit("Error opening wav_utt2dur file " + str(args.wav_utt2dur))
wav_utt_ids = []
wav_lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("Bad line in wav_utt2dur file " + line)
    if float(a[1]) < args.min_additive_noise_len:
	sys.exit("ERROR: The wav length is shorter than --min-additive-noise-len")
    wav_utt_ids.append(a[0])
    wav_lengths.append(float(a[1]))
f.close()

wav_num_utts = len(wav_utt_ids)

# deal with the noise wav utt2dur
# remove the noise whose length < --min-additive-noise-len
num_error = 0
num_done = 0
f = open(args.noise_utt2dur, "r")
if f is None:
    sys.exit("Error opening wav_utt2dur file " + str(args.wav_utt2dur))
noise_utt_ids = []
noise_lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in noise_utt2dur file " + line);
    if float(a[1]) < args.min_additive_noise_len:
        num_error += 1
        continue
    noise_utt_ids.append(a[0])
    noise_lengths.append(float(a[1]))
    num_done += 1
f.close()
noise_num_utts = len(noise_utt_ids)
noise_str =  "Warning: There are " + str(num_error) + " noise files length smaller than " + \
             "--min-additive-noise-len, we remove it from the noise list. Now, there are " + \
             str(num_done) + " noise file."
sys.stdout.write( noise_str + '\n')

num_error = 0
num_done = 0
# generate the range file for each original wav file
for i in range(0, wav_num_utts):
    # decide the number of noises which will be add to 
    current_wav_len = wav_lengths[i]
    max_num_additive_noise = int(current_wav_len / args.min_additive_noise_len)
    
    if max_num_additive_noise > noise_num_utts:
        print( "Warning: The number of noise files or the --min-additive-noise-len is too small" )
        num_error += 1
        continue

    # We generate $num_kind_range ranges
    for j in range(0, args.num_kind_range):
        # select a number from [1 ... max_num_additive_noise]
        num_additive_noise = random.randint(1, max_num_additive_noise)
    
        # decide the length of each noise, minus 0.01 to prevent overstep
        len_additive_noise = float('{:.2f}'.format(current_wav_len / num_additive_noise)) - 0.01

        # create a file to record the ranges
        f = open(args.rangs_dir + "/" + str(wav_utt_ids[i]) + ".range." + str(j), "w")
        if f is None:
            sys.exit("Error open file " + args.rangs_dir + "/" + str(wav_utt_ids[i]) + ".ranges." + str(j))
        # generate range file
        # format: wav_t_start, wav_t_end, noise_name, noise_t_start, noise_t_end, snr
        for k in range(0, num_additive_noise - 1):
            wav_t_start = float('{:.2f}'.format(k * len_additive_noise))
            wav_t_end = wav_t_start + len_additive_noise
			
            noise_index = random.randrange(0, noise_num_utts)
            current_noise_name = noise_utt_ids[noise_index]
            current_noise_len = noise_lengths[noise_index]
            if current_noise_len <= len_additive_noise:
	        noise_t_start = 0.0
		noise_t_end = current_noise_len
	    else :
	        noise_start_bound = float('{:.2f}'.format(current_noise_len - len_additive_noise))
                noise_t_start = float('{:.2f}'.format(random.uniform(0, noise_start_bound)))
                noise_t_end = noise_t_start + len_additive_noise

            current_snr = random.randrange(args.max_snr, args.min_snr)

            print("{0} {1} {2} {3} {4} {5}".format(wav_t_start,
                                                   wav_t_end,
                                                   current_noise_name,
                                                   noise_t_start,
                                                   noise_t_end,
                                                   current_snr),
                  file=f)
	    # deal with the last noise, which cover the rest
        k = num_additive_noise - 1
	wav_t_start = float('{:.2f}'.format(k * len_additive_noise))
        wav_t_end = current_wav_len

	noise_index = random.randrange(0, noise_num_utts)
        current_noise_name = noise_utt_ids[noise_index]
        current_noise_len = noise_lengths[noise_index]

	if current_noise_len <= (wav_t_end - wav_t_start):
	    noise_t_start = 0.0
	    noise_t_end = current_noise_len
	else :
	    noise_start_bound = float('{:.2f}'.format(current_noise_len - wav_t_end + wav_t_start))
            noise_t_start = float('{:.2f}'.format(random.uniform(0, noise_start_bound)))
            noise_t_end = noise_t_start + wav_t_end - wav_t_start		
		
	current_snr = random.randrange(args.max_snr, args.min_snr)

        print("{0} {1} {2} {3} {4} {5}".format(wav_t_start,
                                               wav_t_end,
                                               current_noise_name,
                                               noise_t_start,
                                               noise_t_end,
                                               current_snr),
              file=f)
		
        f.close()
	num_done += 1
		
print('''generate_fixed_length_range.py: finished generate the range files for all wav. Compare with our expect, it lacks %d files. Now we totally have %d noise range files.''' %(num_error, num_done) )
