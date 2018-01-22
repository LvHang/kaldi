#!/usr/bin/env python3

# Copyright 2017 Johns Hopkins University (author: Hang Lyu)

# This version depend on num_egs_per_speaker_per_length.

from __future__ import print_function
import re, os, argparse, sys, math, warnings, random

def get_args():
    parser = argparse.ArgumentParser(description="In previous xvector setup, "
            "the different examples in the same archive are equvalent length."
            "Now, we hope the length of different examples in the same archive is "
            "different. This script modifies the feature.scp in original data directory "
            "to generate some new feats.scp which corresponds "
            "to a specific length. After then, we can combine the new files, "
            "shuffle the list and split it into pieces. At last, we use "
            "nnet3-xvector-get-egs-seg to dump egs. The format of new feats_li.scp is "
            "<uttid-startpoint-len> <extend-filename-of-features[startpoint, startpoint+len-1]>.",
            epilog="Called by sid/nnet3/xvector/get_egs_seg.sh")
    parser.add_argument("--min-frames-per-chunk", type=int, default=50,
            help="Minimum number of frames-per-chunk used for any archive.")
    parser.add_argument("--max-frames-per-chunk", type=int, default=300,
            help="Maximum number of frames-per-chunk used for any archive")
    parser.add_argument("--kinds-of-length", type=int, default=4,
            help="Number of length types.")
    parser.add_argument("--num-egs-per-speaker-per-length", type=int, default=-1,
            help="Number of egs for per speaker per kind of length")
    parser.add_argument("--generate-validate", type=str,
            help="If true, for each utterance-id, we just generate one segment."
            "This method will be used in generating validate egs.",
            default="false", choices = ["false", "true"])
    parser.add_argument("--average-method", type-str,
            help="If true, randomly select a utterance of a speaker. If false,"
            "divid it equally. For example, a sepaker has 8 utterance and "
            "--num-egs-per-speaker-per-length is 50. 50/8=6.25, that means each "
            "utterance will generate 6 egs and it has 25% to generate extra one.",
            default="false", choice = ["false", "true"])
    parser.add_argument("--randomize-chunk-length", type=str,
            help="If true, randomly pick a chunk length in [min-frames-per-chunk, max-frames-per-chunk]."
            "If false, the chunk length varies from min-frames-per-chunk to "
            "max-frames-per-chunk according to a geometric sequence.",
            default="true", choices = ["false", "true"])
    parser.add_argument("--seed", type=int, default=123,
            help="Seed for random number generator.")
    parser.add_argument("--data-dir", type=str, required=True,
            help="The location of original data directory which contains "
            "feature.scp ,utt2num_frames and utt2spk.")
    parser.add_argument("--output-dir", type=str, required=True,
            help="The name of output-dir. In it, there are 'kinds-of-length' directories. "
            "In each subdirectory, it will contains the new generated feats_li.scp and new utt2spk.")

    print(' '.join(sys.argv), file=sys.stderr)
    print(sys.argv, file=sys.stderr)
    args = parser.parse_args()
    args = process_args(args)
    return args


def process_args(args):
    if args.num_repeats < 1:
        raise Exception("--num-repeats should have a minimum value of 1")
    if not os.path.exists(args.data_dir):
        raise Exception("This script expects --data-dir to exist")
    if not os.path.exists(args.data_dir+"/utt2num_frames"):
        raise Exception("This script expects the utt2num_frames file to exist")
    if not os.path.exists(args.data_dir+"/feats.scp"):
        raise Exception("This script expects the original feats.scp to exist")
    if not os.path.exists(args.data_dir+"/utt2spk"):
        raise Exception("This script expects the original utt2spk to exist")
    if args.min_frames_per_chunk <= 1:
        raise Exception("--min-frames-per-chunk is invalid")
    if args.max_frames_per_chunk < args.min_frames_per_chunk:
        raise Exception("--max-frames-per-chunk is invalid")
    if args.kinds_of_length < 1:
        raise Exception("--kinds-of-length is invalid")
    if ((args.max_frames_per_chunk-args.min_frames_per_chunk) < args.kinds_of_length) :
        raise Exception("--kinds-of-length is too large")
    return args


# Create utt2spk
def get_utt2spk(utt2spk_filename):
    utt2spk = {}
    spk2utt = {}
    f = open(utt2spk_filename, "r")
    if f is None:
        sys.exit("Error opening utt2spk file " + str(utt2spk_filename))
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2spk file " + line)
        utt = tokens[0]
        spk = tokens[1]
        utt2spk[utt] = spk
        if spk not in spk2utt:
            spk2utt[spk] = [utt]
        else:
            spk2utt[spk].append(utt)
    f.close()
    return utt2spk, spk2utt
# Done utt2spk


# Create utt2len
def get_utt2len(utt2len_filename):
    utt2len = {}
    f = open(utt2len_filename, "r")
    if f is None:
        sys.exit("Error opening utt2len file " + str(utt2len_filename))
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2len file " + line)
        utt2len[tokens[0]] = int(tokens[1])
    f.close()
    return utt2len
# Done utt2len


# Get all original utt_id and utt2fea dict
def get_utt2fea(utt2fea_filename):
    utt2fea = {}
    f = open(utt2fea_filename, "r")
    if f is None:
        sys.exit("Error opening utt2num file " + str(utt2fea_filename))
    utt_ids = []
    for line in f:
        tokens = line.split()
        if len(tokens) != 2:
            sys.exit("bad line in utt2len file " + line)
        utt2fea[tokens[0]] = tokens[1]
        utt_ids.append(tokens[0])
    f.close()
    return utt_ids, utt2fea
# Done utt2fea


# Randomly generate n length_type
def random_chunk_length(min_frames_per_chunk, max_frames_per_chunk, kinds_of_len):
    ans = []
    while(len(ans) < kinds_of_len):
      x=random.randint(min_frames_per_chunk, max_frames_per_chunk)
      if x not in ans:
          ans.append(x)
    return ans
# Done


# This function returns a geometric sequence in the range
# [min-frames-per-chunk, max-frames-per-chunk]. For example, suppose min-frames-per-chunk
# is 50, and max-frames-per-chunk is 200, and kinds-of-len is 3. The output will
# be [50, 100, 200]
def deterministic_chunk_length(min_frames_per_chunk, max_frames_per_chunk, kinds_of_len):
    ans = []
    if min_frames_per_chunk == max_frames_per_chunk:
        ans = [ max_frames_per_chunk ] * kinds_of_len
    elif kinds_of_len == 1:
        ans = [ max_frames_per_chunk ]
    else:
        for i in range(0, kinds_of_len):
            ans.append(
                int(math.pow(float(max_frames_per_chunk)/min_frames_per_chunk,
                    float(i)/(kinds_of_len-1)) * min_frames_per_chunk + 0.5))
    return ans
# Done


def main():
    args = get_args()
    random.seed(args.seed)
    #utt2len is a dict: key=uttid, value=num_of_frames
    utt2len_filename = args.data_dir + "/utt2num_frames"
    utt2len = get_utt2len(utt2len_filename)
    #utt2fea is a dict: key=uttid, value=extend_filename_of_featue
    #utt_list is a list cotains all the original uttid
    fea_filename = args.data_dir + "/feats.scp"
    utt_list, utt2fea = get_utt2fea(fea_filename)
    #utt2spk is a dict: key=uttid, value=spkid
    utt2spk_filename = args.data_dir + "/utt2spk"
    utt2spk, spk2utt = get_utt2spk(utt2spk_filename)
    spks = spk2utt.keys()

    length_types = []
    # Generate l1 ... ln
    if args.randomize_chunk_length == "true":
        length_types = random_chunk_length(args.min_frames_per_chunk, 
                args.max_frames_per_chunk, args.kinds_of_length)
    else:
        length_types = deterministic_chunk_length(args.min_frames_per_chunk,
                args.max_frames_per_chunk, args.kinds_of_length)

    if (args.generate_validate == "false") :
        for i in range(0, args.kinds_of_length):
            num_err = 0
            this_length = length_types[i]

            this_dir_name =  args.output_dir + "/data_" + str(this_length)
            this_fea_filename = this_dir_name + "/feats.scp.temp"
            if not os.path.exists(this_dir_name):
                os.makedirs(this_dir_name)
        
            fea_f = open(this_fea_filename, "w")
            if fea_f is None:
                sys.exit(str("Error opening file {0}").format(this_fea_filename))
        
            this_utt2spk_filename = this_dir_name + "/utt2spk.temp"
            spk_f = open(this_utt2spk_filename, "w")
            if spk_f is None:
                sys.exit(str("Error opening file {0}").format(this_utt2spk_filename))

            for j in range(len(spks)):
                if (args.average_method == "false") :
                    this_speaker = spks[j]
                    this_utts = spk2utt[this_speaker]
                    this_num_utts = len(this_utts)
                    for k in range(len(args.num_egs_per_speaker_per_length)):
                        x = random.randint(0, this_num_utts-1)
                        this_utt_id = this_utts[x]
                        this_spk_id = utt2spk[this_utt_id]
                        this_extend = utt2fea[this_utt_id]
                        this_utt_len = utt2len[this_utt_id]
                        # check the error
                        if this_utt_len < this_length:
                            num_err = num_err + 1
                            if num_err > (0.1 * len(spks) * args.num_egs_per_speaker_per_length):
                                raise Exception(str(this_length) + "is not a suitable length")
                        else :
                            this_utt_boundary = int(this_utt_len - this_length)
                            start_point = random.randint(0, this_utt_boundary)
                            new_utt_id = this_utt_id + '_' + str(start_point) + '_' + str(this_length)
                            new_extend = this_extend + '[' + str(start_point) + ':' + str(start_point+this_length-1) + ']'
                            print("{0} {1}".format(new_utt_id,
                                           new_extend),
                                  file=fea_f)
                            print("{0} {1}".format(new_utt_id,
                                           this_spk_id),
                                  file=spk_f)
                else :
                    # Ensure which speaker, then get the number of utterance of this speaker.
                    # For each utterance, generate "num_this_segments" segments.
                    this_spekaer = spks[j]
                    this_utts = spk2utt[this_speaker]
                    this_num_utts = len(this_utts)
                    num_this_segments = round(args.num_egs_per_speaker_per_length / this_utts, 2)
                    #integral part and decimal part
                    integral_part = int(num_this_segments)
                    decimal_part = num_this_segments - integral_part
                    
                    for k in range(this_num_utts):
                        this_utt_id = this_utts[k]
                        this_spk_id = utt2spk[this_utt_id]
                        this_extend = utt2fea[this_utt_id]
                        this_utt_len = utt2len[this_utt_id]
                        # check the error
                        if this_utt_len < this_length:
                            num_err = num_err + 1
                            if num_err > (0.1 * len(spks) * args.num_egs_per_speaker_per_length):
                                raise Exception(str(this_length) + "is not a suitable length")
                        else :
                            this_utt_boundary = int(this_utt_len - this_length)
                            for l in range(integral_part):
                                start_point = random.randint(0, this_utt_boundary)
                                new_utt_id = this_utt_id + '_' + str(start_point) + '_' + str(this_length)
                                new_extend = this_extend + '[' + str(start_point) + ':' + str(start_point+this_length-1) + ']'
                                print("{0} {1}".format(new_utt_id,
                                                       new_extend),
                                      file=fea_f)
                                print("{0} {1}".format(new_utt_id,
                                                       this_spk_id),
                                      file=spk_f)
                            if ((random.randint(0,100)) * 1.0 /100.0) < decimal_part :
                                start_point = random.randint(0, this_utt_boundary)
                                new_utt_id = this_utt_id + '_' + str(start_point) + '_' + str(this_length)
                                new_extend = this_extend + '[' + str(start_point) + ':' + str(start_point+this_length-1) + ']'
                                print("{0} {1}".format(new_utt_id,
                                                       new_extend),
                                      file=fea_f)
                                print("{0} {1}".format(new_utt_id,
                                                       this_spk_id),
                                      file=spk_f)
            fea_f.close()
            spk_f.close()
    else:
        #For each utterance, generate one segment. It always be used in validate or diagnose.
        for i in range(0, args.kinds_of_length):
            num_err = 0
            this_length = length_types[i]
            num_this_segments = 1

            this_dir_name =  args.output_dir + "/data_" + str(this_length)
            this_fea_filename = this_dir_name + "/feats.scp"
            if not os.path.exists(this_dir_name):
                os.makedirs(this_dir_name)
        
            fea_f = open(this_fea_filename, "w")
            if fea_f is None:
                sys.exit(str("Error opening file {0}").format(this_fea_filename))
        
            this_utt2spk_filename = this_dir_name + "/utt2spk"
            spk_f = open(this_utt2spk_filename, "w")
            if spk_f is None:
                sys.exit(str("Error opening file {0}").format(this_utt2spk_filename))
        
            for j in range(len(utt_list)):
                this_utt_id = utt_list[j]
                this_spk_id = utt2spk[this_utt_id]
                this_extend = utt2fea[this_utt_id]
                this_utt_len = utt2len[this_utt_id]
                if this_utt_len < this_length:
                    num_err = num_err + 1
                    if num_err > (0.1 * len(utt_list)):
                        raise Exception(str(this_length) + "is not a suitable length")
                    break
                else:
                    this_utt_boundary = int(this_utt_len - this_length)
                    start_point = random.randint(0, this_utt_boundary)
                    new_utt_id = this_utt_id + '_' + str(start_point) + '_' + str(this_length)
                    new_extend = this_extend + '[' + str(start_point) + ':' + str(start_point+this_length-1) + ']'
                    print("{0} {1}".format(new_utt_id,
                                           new_extend),
                        file=fea_f)
                    print("{0} {1}".format(new_utt_id,
                                           this_spk_id),
                        file=spk_f)
                   
            fea_f.close()
            spk_f.close()
    print("allocate_egs_seg.py: finished")
# Done main       



if __name__ == "__main__":
    main()
