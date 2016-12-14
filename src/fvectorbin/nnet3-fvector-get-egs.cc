// fvectorbin/nnet3-fvector-get-egs.cc

// Copyright 2016  Johns Hopkins University (author:  Daniel Povey)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"

namespace kaldi {
namespace nnet3 {


static void ProcessFile(const MatrixBase<BaseFloat> &feats,
                        const std::string &utt_id,
                        bool compress,
                        int32 left_context,
                        int32 right_context,
                        int32 frames_per_eg,
                        int64 *num_frames_written,
                        int64 *num_egs_written,
                        NnetExampleWriter *example_writer) {
  for (int32 t = 0; t < feats.NumRows(); t += frames_per_eg) {

    // actual_frames_per_eg is the number of frames in center.
    // At the end of the file we pad with zero posteriors
    // so that all examples have the same structure (prevents the need
    // for recompilations).
    int32 actual_frames_per_eg = std::min(frames_per_eg,
                                          feats.NumRows() - t);

    int32 tot_frames = left_context + frames_per_eg + right_context;

    Matrix<BaseFloat> input_frames(tot_frames, feats.NumCols(), kUndefined);
    
    // Set up "input_frames".
    for (int32 j = -left_context; j < frames_per_eg + right_context; j++) {
      int32 t2 = j + t;
      if (t2 < 0) t2 = 0;
      if (t2 >= feats.NumRows()) t2 = feats.NumRows() - 1;
      SubVector<BaseFloat> src(feats, t2),
                           dest(input_frames, j + left_context);
      dest.CopyFromVec(src);
    }

    NnetExample eg;
    
    // call the regular input "input".
    eg.io.push_back(NnetIo("input", -left_context, input_frames));
   
    if (compress) { eg.Compress();}
      
    std::ostringstream os;
    os << utt_id << "-" << t;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    *num_frames_written += actual_frames_per_eg;
    *num_egs_written += 1;

    example_writer->Write(key, eg);
  }
}


} // namespace nnet3
} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Get frame-by-frame examples of data for nnet3 neural network training.\n"
        "Essentially this is a format change from features into a special frame-by-frame format.\n"
        "This program handles the common case where you have some input features\n"
        "and convert them to fvector examples format\n"
        "Note: In fvector version, there is no need for iVectors, posterior and labels.\n"
        "\n"
        "Usage:  nnet3-fvector-get-egs [options] <features-rspecifier> <egs-out>\n"
        "\n"
        "An example [where $feats expands to the actual features]:\n"
        "nnet3-fvector-get-egs --left-context=12 --right-context=9 --compress=true \"$feats\" \\\n"
        "\"ark:train.egs\"\n";
        

    bool compress = true;
    int32 left_context = 0, right_context = 0, num_frames = 1;
    
    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");
    po.Register("left-context", &left_context, "Number of frames of left "
                "context the neural net requires.");
    po.Register("right-context", &right_context, "Number of frames of right "
                "context the neural net requires.");
    po.Register("num-frames", &num_frames, "Number of frames is central");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
                examples_wspecifier = po.GetArg(2);

    // Read in all the training files.
    SequentialBaseFloatMatrixReader feat_reader(feature_rspecifier);
    NnetExampleWriter example_writer(examples_wspecifier);
    
    int32 num_done = 0;
    int64 num_frames_written = 0, num_egs_written = 0;
    
    for (; !feat_reader.Done(); feat_reader.Next()) {
      std::string key = feat_reader.Key();
      const Matrix<BaseFloat> &feats = feat_reader.Value();
      ProcessFile(feats, key, compress, left_context, right_context,
                  num_frames, &num_frames_written, &num_egs_written,
                  &example_writer);
      num_done++;
    }

    KALDI_LOG << "Finished generating examples, "
              << "successfully processed " << num_done
              << " feature files, wrote " << num_egs_written << " examples, "
              << " with " << num_frames_written << " egs in total.";
    return (num_egs_written == 0 || num_done == 0 ? 1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
