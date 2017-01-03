#!/usr/bin/env python

# This function is used to generate the perturbed_wav.scp with the inputs as
# wav.scp, wav2perturbedwav, ranges

# The final format is :
# wav1 sph2pipe -f wav -p -c 1 $path/wav1.sph |
# wav1-p1 sph2pipe -f wav -p -c 1 $path/wav1.sph | nnet3-fvector-perturb-signal 
# --noise-scp=scp:noise.scp noise-range="range-p1-for-wav1" - |

from __future__ import print_function
import re, os, argparse, sys, math, warnings, random

parser = argparse.ArgumentParser(description="Generate a mapping file which use to map the wav to  "
                                 "Corresponding pertrubedwav",
                                 epilog="Called by steps/nnet3/fvector/add_noise.sh")
parser.add_argument("--noise", type=str,
                    help="To assign the noise.scp. You must make sure it is same with "
                    "the noise.scp which is used to generate range_file.")
# now the positional arguments
parser.add_argument("wav_scp",
                    help="The orginial wav.scp which contains all the original wav "
                    "The format is: <recording-id> <extended-file>.")
parser.add_argument("range_file",
                    help="The file contains the range information which is used to "
                    "control the process of adding noise. The format is : "
                    "<perturbedwavid> <range-information>.")
parser.add_argument("wav2perturbedwav",
                    help="This file contains the mapping between wav and perturbedwav.")
parser.add_argument("perturbed_wav_scp",
                    help="The file is used to store the perturbed wav sperifier.")

print(' '.join(sys.argv))

args = parser.parse_args()

# Extract the information form the wav_scprding_ids = []
wav_recording_ids = []
wav_extended_files = []
f = open(args.wav_scp, "r")
if f is None:
    sys.exit("Error opening wav.scp file")
for line in f:
    # remove the "\n" in the end of each line
    line.split("\n")
    a = line.split()
    wav_recording_ids.append(a[0])
    del a[0]
    wav_extended_files.append(' '.join(a))
f.close()

# Extract the infromation from the range_file
perturbed_range_ids = []
perturbed_range_contents = []
f = open(args.range_file, "r")
if f is None:
    sys.exit("Error opening range_file")
for line in f:
    # remove the "\n" in the end of each line
    line.split("\n")
    a = line.split()
    if len(a) != 2:
        sys.exit("Bad line \"" + line + "\" in file: " + str(args.range_file))
    perturbed_range_ids.append(a[0])
    perturbed_range_contents.append(a[1])
f.close()

# generate the mapping file through iterating all terms in the wav2perturbedwav
f = open(args.wav2perturbedwav, "r")
if f is None:
    sys.exit("Error opening wav2perturbedwav")
# make a store file.
g = open(args.perturbed_wav_scp, "w")
if g is None:
    sys.exit("Error opening perturbed_wav_specifier")

# start the loop
for line in f:
    # remove the "\n" in the end of each line
    line.split("\n")
    wav_list = line.split()
    current_wav_id = wav_list[0]
    current_wav_index = wav_recording_ids.index(current_wav_id)

    # print the original wav
    print("{0} {1}".format(current_wav_id, wav_extended_files[current_wav_index]), file=g)
    
    for i in range(1, len(wav_list)):
        current_perturbed_wav_id = wav_list[i]
        current_perturbed_wav_index = perturbed_range_ids.index(current_perturbed_wav_id)
        print('''{0} {1} nnet3-fvector-perturb-signal --noise-scp=scp:{2} --noise=\"{3}\" - |'''.format(
            current_perturbed_wav_id,
            wav_extended_files[current_wav_index],
            args.noise,
            perturbed_range_contents[current_perturbed_wav_index]),file=g)
g.close()
f.close()
print("Finished generating the perturb_wav.scp")
