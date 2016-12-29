#!/usr/bin/env python

# The function use to generate range-file for fvector
# Each line of the range-file corrsponds to a kind of perturbed wav. In each line,
# we use comma to seperate different addnoise range. The format of each addnoise
# range is <wav_t_start>:<wav_t_end>:<noise_uttid>:<noise_t_start>:<noise_t_end>:<snr>
# The line which starts with the asterisk(*) is the differences between two versions.

# For the fixed-length version:
# *For <wav_t_start> <wav_t_end>
# *Except the last fragement, the length will be a fixed value T.
# For <noise_uttid>
# It is randomly selected from noise list, which is longer than --min-additive-noise-len
# *For <noise_t_start> <noise_t_end>
# *If the noise file is longer than fixed value. We randomly select the start point and 
# *the length will be fixed value T.
# *If the noise file is shorter than T. We select the whole noise.
# The <snr> control the rate of signal and noise. In the other word, scale the amplitude of noise.
# The snr will be randomly selected form the range (max-snr, min-snr).

# For the variable-length version:
# *For <wav_t_start> <wav_t_end>
# *Except the last fragement, the length will be random.
# For <noise_uttid>
# It is randomly selected from noise list, which is longer than --min-additive-noise-len
# *For <noise_t_start> <noise_t_end>
# *If the noise file is longer than wav length. We randomly select the start point and 
# *the length will be the same as wav length.
# *If the noise file is shorter than T. We select the whole noise.
# For <snr>, it was used to control the amplitude of noise
# It will be randomly selected from the range (max-snr, min-snr)


from __future__ import print_function
import re, os, argparse, sys, math, warnings, random

parser = argparse.ArgumentParser(description="Generate a noise range-file which contains "
                                 "N lines corresponding to the number of kinds for each original wav. "
                                 "The file which created by this python code will be supplied to "
                                 "add additive noise program.",
                                 epilog="Called by steps/nnet3/fvector/add_noise.sh")
parser.add_argument("--num-kind-range", type=int, default=4,
                    help="the number of expected addnoise kinds")
parser.add_argument("--min-additive-noise-len", type=float, default=2.0,
                    help="the minimum duration/length of each noise file")
parser.add_argument("--min-snr", type=int, default=-5,
                    help="the minimum Signal-to-Noise Rate, the default=0")
parser.add_argument("--max-snr", type=int, default=-15,
                    help="the maximum Signal-to-Noise Rate, the default=-10")
parser.add_argument("--seed", type=int, default=-1,
                    help="Seed for random number generator")
parser.add_argument("--variable-len-additive-noise", type=str,
                    help="If true, generate the variable-length range files for each original wavform file."
                    "If false, generate the fixed-length range files for each original wavform file.",
                    default="false", choices = ["false", "true"])

# now the positional arguments
parser.add_argument("wav2dur",
                    help="wav2dur file of the original wav to be used as input (format is: "
                    "<utterance-id> <duration>")
parser.add_argument("noise2dur",
                    help="noise2dur file of the noise wav to be used as input (format is: "
                    "<utterance-id> <duration>")
parser.add_argument("range_dir",
                    help="Name of ranges directory, exp/fxvector/ranges")

print(' '.join(sys.argv))

args = parser.parse_args()

## Check arguments
if args.min_snr < args.max_snr:
    sys.exit("For SNR, the less numerical value is, the larger noise is. So --min-snr bigger "
             "than --max-snr in numerical value.")

random.seed(args.seed)

# deal with the original wav utt2dur
# the information was stored in wav_ids[], wav_lengths[] and wav_num_utts
f = open(args.wav2dur, "r")
if f is None:
    sys.exit("Error opening wav2dur file " + str(args.wav2dur))
wav_ids = []
wav_lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("Bad line in wav2dur file " + line)
    if float(a[1]) < args.min_additive_noise_len:
	sys.exit("ERROR: The wav length is shorter than --min-additive-noise-len")
    wav_ids.append(a[0])
    wav_lengths.append(float(a[1]))
f.close()

wav_num_utts = len(wav_ids)

# deal with the noise wav utt2dur
# remove the noise whose length < --min-additive-noise-len
num_error = 0
num_done = 0
f = open(args.noise2dur, "r")
if f is None:
    sys.exit("Error opening noise2dur file " + str(args.noise2dur))
noise_ids = []
noise_lengths = []
for line in f:
    a = line.split()
    if len(a) != 2:
        sys.exit("Bad line in noise2dur file " + line);
    if float(a[1]) < args.min_additive_noise_len:
        num_error += 1
        continue
    noise_ids.append(a[0])
    noise_lengths.append(float(a[1]))
    num_done += 1
f.close()
noise_num_utts = len(noise_ids)
noise_str =  "Warning: There are " + str(num_error) + " noise files length smaller than " + \
             "--min-additive-noise-len, we remove it from the noise list. Now, there are " + \
             str(num_done) + " noise file."
sys.stdout.write( noise_str + '\n')

# This function generates the fixed-length range files
def GenerateFixedLengthRangeFile():
    num_fixed_error = 0
    num_fixed_done = 0

    for i in range(0, wav_num_utts):
        # decide the number of noises which will be add to 
        current_wav_len = wav_lengths[i]
        max_num_additive_noise = int(current_wav_len / args.min_additive_noise_len)
    
        if max_num_additive_noise > noise_num_utts:
            print( "Warning: The number of noise files or the --min-additive-noise-len is too small" )
            num_fixed_error += 1
            continue

        # create a file to record the ranges
        f = open(args.range_dir + "/" + str(wav_ids[i]) + ".noiserange", "w")
        if f is None:
            sys.exit("Error open file " + args.range_dir + "/" + str(wav_ids[i]) + ".noiserange")

        # We generate $num_kind_range ranges
        for j in range(0, args.num_kind_range):
            # print the perturbed wav id in the beginning of line
            print("{0}-{1}".format(wav_ids[i], "perturbed-"+str(j+1)), end=" ", file=f)
            
            # select a number from [1 ... max_num_additive_noise]
            num_additive_noise = random.randint(1, max_num_additive_noise)
    
            # decide the length of each noise, minus 0.01 to prevent overstep
            additive_noise_len = float('{:.2f}'.format(current_wav_len / num_additive_noise)) - 0.01

            # generate one line of file
            # format: wav_t_start:wav_t_end:noise_name:noise_t_start:noise_t_end:snr,
            for k in range(0, num_additive_noise - 1):
                wav_t_start = float('{:.2f}'.format(k * additive_noise_len))
                wav_t_end = wav_t_start + additive_noise_len
			
                noise_index = random.randrange(0, noise_num_utts)
                current_noise_name = noise_ids[noise_index]
                current_noise_len = noise_lengths[noise_index]
                if current_noise_len <= additive_noise_len:
	            noise_t_start = 0.0
		    noise_t_end = current_noise_len
	        else :
	            noise_start_bound = float('{:.2f}'.format(current_noise_len - additive_noise_len))
                    noise_t_start = float('{:.2f}'.format(random.uniform(0, noise_start_bound)))
                    noise_t_end = noise_t_start + additive_noise_len

                current_snr = random.randrange(args.max_snr, args.min_snr)

                print("{0}:{1}:{2}:{3}:{4}:{5}".format(wav_t_start,
                                                       wav_t_end,
                                                       current_noise_name,
                                                       noise_t_start,
                                                       noise_t_end,
                                                       current_snr),
                      end=",",file=f)
	    # deal with the last noise, which cover the rest
            k = num_additive_noise - 1
	    wav_t_start = float('{:.2f}'.format(k * additive_noise_len))
            wav_t_end = current_wav_len

	    noise_index = random.randrange(0, noise_num_utts)
            current_noise_name = noise_ids[noise_index]
            current_noise_len = noise_lengths[noise_index]

	    if current_noise_len <= (wav_t_end - wav_t_start):
	        noise_t_start = 0.0
	        noise_t_end = current_noise_len
	    else :
	        noise_start_bound = float('{:.2f}'.format(current_noise_len - wav_t_end + wav_t_start))
                noise_t_start = float('{:.2f}'.format(random.uniform(0, noise_start_bound)))
                noise_t_end = noise_t_start + wav_t_end - wav_t_start		
		
	    current_snr = random.randrange(args.max_snr, args.min_snr)

            print("{0}:{1}:{2}:{3}:{4}:{5}".format(wav_t_start,
                                                   wav_t_end,
                                                   current_noise_name,
                                                   noise_t_start,
                                                   noise_t_end,
                                                   current_snr),
                  file=f)
	    num_fixed_done += 1
        f.close()
    print('''generate_fixed_length_range.py: finished generate the range files for all wav. Compare with our expect, it lacks %d files. Now we totally have %d noise range files.''' %(num_fixed_error, num_fixed_done) )

# This function generates the variable-length range files
def GenerateVariableLengthRangeFile():
    num_variable_error = 0
    num_variable_done = 0

    for i in range(0, wav_num_utts):

        # check the noise list has enough sample or not
        current_wav_len = wav_lengths[i]
        max_num_additive_noise = int(current_wav_len / args.min_additive_noise_len)
    
        if max_num_additive_noise > noise_num_utts:
            print( "Warning: The number of noise files or the --min-additive-noise-len is too small" )
            num_variable_error += 1
            continue

        # create a file to record the ranges
        f = open(args.range_dir + "/" + str(wav_ids[i]) + ".noiserange", "w")
        if f is None:
            sys.exit("Error open file " + args.range_dir + "/" + str(wav_ids[i]) + ".noiserange")
        
        # We generate $num_kind_range ranges
        for j in range(0, args.num_kind_range):
            # print the perturbed wav id in the beginning of line
            print("{0}-{1}".format(wav_ids[i], "perturbed-"+str(j+1)), end=" ", file=f)

            # generate range file
            # format: wav_t_start:wav_t_end:noise_name:noise_t_start:noise_t_end:snr,
            the_rest = current_wav_len
            wav_t_start = 0.0
            wav_t_end = 0.0
            while (the_rest > float(args.min_additive_noise_len)):
	        # firstly, we randomly choose a kind of noise and snr
	        noise_index = random.randrange(0, noise_num_utts)
                current_noise_name = noise_ids[noise_index]
                current_noise_len = noise_lengths[noise_index]
	        current_snr = random.randrange(args.max_snr, args.min_snr)
			
                # Secondly, we randomly select a fragement of the noise file.
                noise_start_bound = float('{:.2f}'.format(current_noise_len - float(args.min_additive_noise_len)))
                noise_t_start = float('{:.2f}'.format(random.uniform(0, noise_start_bound)))
	        noise_end_upperbound = float('{:.2f}'.format(noise_t_start + float(args.min_additive_noise_len)))
	        noise_end_lowerbound = float('{:.2f}'.format(min((noise_t_start + the_rest), current_noise_len)))
                noise_t_end = float('{:.2f}'.format(random.uniform(noise_end_upperbound, noise_end_lowerbound)))
	        current_noise_length = noise_t_end - noise_t_start
			
	        # Thirdly, we generate the start and end point of wav
                wav_t_start = wav_t_end #the new start is the end of the last.
	        wav_t_end = wav_t_start + current_noise_length
			
	        # Forthly, update the_rest
	        the_rest = the_rest - current_noise_length
	    
                # Fifthly, print
                print("{0}:{1}:{2}:{3}:{4}:{5}".format(wav_t_start,
                                                       wav_t_end,
                                                       current_noise_name,
                                                       noise_t_start,
                                                       noise_t_end,
                                                       current_snr),
                      end=",",file=f)
	    # deal with the bit of wav
	    # firstly, we randomly choose a kind of noise and snr
	    noise_index = random.randrange(0, noise_num_utts)
            current_noise_name = noise_ids[noise_index]
            current_noise_len = noise_lengths[noise_index]
	    current_snr = random.randrange(args.max_snr, args.min_snr)
		
	    # Secondly, we randomly select a fragement of the noise file.
            noise_start_bound = float('{:.2f}'.format(current_noise_len - the_rest))
            noise_t_start = float('{:.2f}'.format(random.uniform(0, noise_start_bound)))
	    noise_t_end = noise_t_start + the_rest
	    current_noise_length = noise_t_end - noise_t_start
		
	    # Thirdly, we generate the start and end point of wav
            wav_t_start = wav_t_end #the new start is the end of the last.
	    wav_t_end = wav_t_start + current_noise_length
		
	    # Forthly, print
            print("{0}:{1}:{2}:{3}:{4}:{5}".format(wav_t_start,
                                                   wav_t_end,
                                                   current_noise_name,
                                                   noise_t_start,
                                                   noise_t_end,
                                                   current_snr),
                  file=f)		
	    num_variable_done += 1
	f.close()
    print('''generate_fixed_length_range.py: finished generate the range files for all wav. Compare with our expect, it lacks %d files. Now we totally have %d noise range files.''' %(num_variable_error, num_variable_done) )

if args.variable_len_additive_noise == "true":
    GenerateVariableLengthRangeFile()
else:
    GenerateFixedLengthRangeFile()
