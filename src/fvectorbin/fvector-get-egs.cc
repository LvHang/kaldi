// fvectorbin/fvector-get-egs.cc

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

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;

    const char *usage =
        "Get examples for training an nnet3 neural network for the fvector\n"
        "system.  Each output example contains a pair of feature chunks.\n"
        "Usage:  fvector-get-egs [options] <chunk-rspecifier> <egs-wspecifier>\n"
        "For example:\n"
        "fvector-get-egs scp:perturbed_chunks.scp ark:egs.ark";

    bool compress = true;

    ParseOptions po(usage);
    po.Register("compress", &compress, "If true, write egs in "
                "compressed format.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1);
    NnetExampleWriter example_writer(po.GetArg(2));


    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

    int32 num_read = 0,
          num_egs_written = 0;
    for (; feature_reader.Done(); feature_reader.Next(), num_read++) {
      std::string key = feature_reader.Key();
      const Matrix<BaseFloat> &feats = feature_reader.Value();
      //Please take care. Here, the 'feats' is a 2-lines matrix which is generated
      //by fvector-add-noise.cc. The 2-lines matrix represents two perturbed 
      //vectors(e.g 100ms wavform) which come from the same source signal.
      //chunk1 and chunk2 corresponds to one line respectively.
      SubMatrix<BaseFloat> chunk1(feats, 0, 1, 0, feats.NumCols()),
                           chunk2(feats, 1, 1, 0, feats.NumCols());
      NnetIo nnet_io1 = NnetIo("input", 0, chunk1),
             nnet_io2 = NnetIo("input", 0, chunk2);
      //modify the n index, so that in a mini-batch Nnet3Example, the adjacent
      //two NnetIos come from the same source signal.
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
      if (compress) {
        eg.Compress();
      }
      example_writer.Write(key, eg);
      num_egs_written += 1;
    }
    KALDI_LOG << "Finished generating examples, "
              << "successfully convert " << num_egs_written << " chunks into examples out of "
              << num_read << " chunks";
    return (num_egs_written == 0);
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
