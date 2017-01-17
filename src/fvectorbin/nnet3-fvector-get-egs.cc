// fvectorbin/nnet3-fvector-get-egs.cc

// Copyright 2012-2016  Johns Hopkins University (author:  Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <sstream>

#include "util/common-utils.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {

// A struct for holding information about the position and
// duration of each pair of chunks.
struct FvectorChunkPairInfo {
  std::string pair_name;
  std::string utt_a;
  std::string utt_b;
  int32 output_archive_id;
  int32 start_frame;
  int32 num_frames;
};

// Process the range input file and store it as a map from utterance
// name to vector of ChunkPairInfo structs.
static void ProcessRangeFile(const std::string &range_rxfilename,
                             std::vector<FvectorChunkPairInfo *> *pairs) {
  Input range_input(range_rxfilename);
  if (!range_rxfilename.empty()) {
    std::string line;
    while (std::getline(range_input.Stream(), line)) {
      FvectorChunkPairInfo *pair = new FvectorChunkPairInfo();
      std::vector<std::string> fields;
      SplitStringToVector(line, " \t\n\r", true, &fields);
      if (fields.size() != 6) {
        KALDI_ERR << "Expected 6 fields in line of range file, got "
                  << fields.size() << " instead.";
      }

      std::string utt_a = fields[0],
                  utt_b = fields[1],
                  start_frame_str = fields[4],
                  num_frames_str = fields[5];

      if (!ConvertStringToInteger(fields[2], &(pair->output_archive_id)) ||
          !ConvertStringToInteger(start_frame_str, &(pair->start_frame)) ||
          !ConvertStringToInteger(num_frames_str, &(pair->num_frames))) {
        KALDI_ERR << "Expected integer for output archive in range file.";
      }
      pair->pair_name = utt_a + "-" + utt_b + "-" + start_frame_str + "-"
                        + num_frames_str;
      pair->utt_a = utt_a;
      pair->utt_b = utt_b;

      pairs->push_back(pair);
    }
  }
}

// Delete the dynamically allocated memory.
static void Cleanup(std::vector<FvectorChunkPairInfo *> *pairs,
                    std::vector<NnetExampleWriter *> *writers) {
  for (std::vector<NnetExampleWriter *>::iterator
      it = writers->begin(); it != writers->end(); ++it) {
    delete *it;
  }
  for (std::vector<FvectorChunkPairInfo *>::iterator it = pairs->begin();
       it != pairs->end(); ++it) {
    delete *it;
  }
}

} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Get examples for training an nnet3 neural network for the fvector\n"
        "system.  Each output example contains a pair of feature chunks from\n"
        "the specified utterance.  The location and length of the feature chunks\n"
        "are specified in the 'ranges' file.  Each line is interpreted as\n"
        "follows:\n"
        "<source-utterance1> <source-utterance2> <relative-output-archive-index> "
        "<absolute-archive-index>  <start-frame> <num-frames1> "
        "<start-frame-index2> <num-frames2>\n"
        "where <relative-output-archive-index> is interpreted as a zero-based\n"
        "index into the wspecifiers specified on the command line (<egs-0-out>\n"
        "and so on), and <absolute-archive-index> is ignored by this program.\n"
        "For example:\n"
        "  utt1-p1 utt1-p2 3  13  5   65\n"
        "  utt2    utt2-pn 0  10  160 50\n"
        "\n"
        "Usage:  nnet3-fvector-get-egs [options] <ranges-filename> "
        "<features-rspecifier> <egs-0-out> <egs-1-out> ... <egs-N-1-out>\n"
        "\n"
        "For example:\n"
        "nnet3-fvector-get-egs ranges.1 \"$feats\" ark:egs_temp.1.ark"
        "  ark:egs_temp.2.ark ark:egs_temp.3.ark\n";

    bool compress = true;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string range_rspecifier = po.GetArg(1);
    std::string feature_rspecifier = po.GetArg(2);
    std::vector<NnetExampleWriter *> example_writers;

    for (int32 i = 3; i <= po.NumArgs(); i++) {
      example_writers.push_back(new NnetExampleWriter(po.GetArg(i)));
    }

    std::vector<FvectorChunkPairInfo *> pairs;
    // deal with the ranges file and initalize the vector
    ProcessRangeFile(range_rspecifier, &pairs);

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

    int32 num_error = 0,
          num_egs_written = 0;

    for (std::vector<FvectorChunkPairInfo *>::iterator iter = pairs.begin();
         iter != pairs.end(); iter++) {

      FvectorChunkPairInfo *pair = *iter;
      // get the features
      if (!feature_reader.HasKey(pair->utt_a) || !feature_reader.HasKey(pair->utt_b)) {
        num_error++;
        KALDI_WARN << "The feature " << pair->utt_a << " or " << pair->utt_b
                   << " is not found.";
        continue;
      }
      const Matrix<BaseFloat> &feats_a = feature_reader.Value(pair->utt_a);
      const Matrix<BaseFloat> &feats_b = feature_reader.Value(pair->utt_b);
      int32 num_rows = feats_a.NumRows(),
            feat_dim = feats_a.NumCols();
      if (num_rows < (pair->start_frame + pair->num_frames)) {
        num_error++;
        KALDI_WARN << "Unable to create examples for utterance " << pair->pair_name
                   << ". Requested chunk boundary is the "
                   << (pair->start_frame + pair->num_frames)
                   << "th frmae, but utterance has only " << num_rows << " frames.";
        continue;
      } else {
        SubMatrix<BaseFloat> chunk1(feats_a, pair->start_frame,
                                    pair->num_frames, 0, feat_dim),
                             chunk2(feats_b, pair->start_frame,
                                    pair->num_frames, 0, feat_dim);
        NnetIo nnet_io1 = NnetIo("input", 0, chunk1),
               nnet_io2 = NnetIo("input", 0, chunk2);
        for (std::vector<Index>::iterator indx_it = nnet_io1.indexes.begin();
            indx_it != nnet_io1.indexes.end(); ++indx_it) {
          indx_it->n = 0;
        }
        for (std::vector<Index>::iterator indx_it = nnet_io2.indexes.begin();
            indx_it != nnet_io2.indexes.end(); ++indx_it) {
          indx_it->n = 1;
        }
        NnetExample eg;
        eg.io.push_back(nnet_io1);
        eg.io.push_back(nnet_io2);
        if (compress)
          eg.Compress();

        if (pair->output_archive_id >= example_writers.size()) {
          KALDI_ERR << "Requested output index exceeds number of specified "
                    << "output files.";
        }
        example_writers[pair->output_archive_id]->Write(pair->pair_name, eg);
        num_egs_written += 1;
      }
    }
    Cleanup(&pairs, &example_writers);

    KALDI_LOG << "Finished generating examples, "
              << "successfully wrote " << num_egs_written << " examples; "
              << num_error << " files had errors.";
    return (num_egs_written == 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
