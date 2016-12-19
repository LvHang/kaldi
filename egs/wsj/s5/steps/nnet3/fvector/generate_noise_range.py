#!/usr/bin/env python

from __future__ import print_function
import re, os, argparse, sys, math, warnings, random

parser = argparse.ArgumentParser(description="Generate n kinds of noise range for each original wav"
                                 epilog="Called by steps/nnet3/fvector/lh_add_noise.sh")
parser.add_argument("--num-kind-range", type=int, default=4,
                    help="the number of kinds of noise ranges")
parser.add_argument("--min-additive-noise-len", type=float, default=2.0,
                    help="the minimum duration of each noise file")
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
f = open(args.wav_utt2dur, "r")
if f is None:
    sys.exit("Error opening wav_utt2dur file " + str(args.wav_utt2dur))
wav_utt_ids = []
wav_lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in wav_utt2dur file " + line)
    wav_utt_ids.append(a[0])
    wav_lengths.append(a[1])
f.close()

wav_num_utts = len(wav_utt_ids)

# deal with the noise wav utt2dur
f = open(args.noise_utt2dur, "r")
if f is None:
    sys.exit("Error opening wav_utt2dur file " + str(args.wav_utt2dur))
noise_utt_ids = []
noise_lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("bad line in noise_utt2dur file " + line);
    if a[1] <  args.min_additive_noise_len:
        sys.exit("bad line in noise_utt2dur file " + line);
    noise_utt_ids.append(a[0])
    noise_lengths.append(a[1])
f.close()

noise_num_utts = len(noise_utt_ids)

# generate the range file for each original wav file
for i in range(0, wav_num_utts):
   
    # decide the number of noises which will be add to 
    current_wav_len = wav_lengths[i]
    max_num_additive_noise = int(current_wav_len / args.min_additive_noise_len)
    upperbound_num_additive_noise = min(max_num_additive_noise, noise_num_utts)

    # select a number from [1 ... upperbound_num_additive_noise]
    num_additive_noise = random.randrange(1, upperbound_num_additive_noise + 1)
    
    # decide the length of each noise, minus 0.01 to prevent overstep
    len_additive_noise = float('{:.2f}'.format(current_wav_len / num_additive)) - 0.01

    # We generate $num_kind_range ranges
    for j in range(0, args.num_kind_range):
 
        # create a file to record the ranges
        f = open(args.rangs_dir + "/" + str(wav_utt_ids[i]) + ".range." + str(j), "w")
        if f is None:
            sys.exit("Error open file " + args.rangs_dir + "/" + str(wav_utt_ids[i]) + ".ranges." + str(j))
        
        # generate range file
        # format: wav_t_start, wav_t_end, noise_name, noise_t_start, noise_t_end, snr
        for k in range(0, num_additive_noise):
            wav_t_start = flat('{:.2f}'.format(k * len_additive_noise))
            
            noise_index = random.randrange(0, noise_num_utts)
            current_noise_name = noise_utt_ids[noise_index]
            current_noise_len = noise_lengths[noise_index]
            
            upperbound_add_len = min(len_additive_noise, current_noise_len)
            current_add_len = float('{:.2f}'.format(random.randrange(0, upperbound_add_len, 0.01)))
            
            noise_start_bound = float('{:.2f}'.format(current_noise_len - current_add_len))
            noise_t_start = float('{:.2f}'.format(random.randrange(0, noise_start_bound)))
            noise_t_end = noise_t_start + current_add_len

            wav_t_end = wav_t_start + current_add_len

            current_snr = random.randrange(args.max_snr, args.min_snr)

            print("{0} {1} {2} {3} {4} {5} {6}".format(wav_t_start,
                                                       wav_t_end,
                                                       current_noise_name,
                                                       noise_t_start,
                                                       noise_t_end,
                                                       current_snr,
                  file=f)
        f.close()
        
print("generate_noise_range.py: finished generate the range files for all wav")        


