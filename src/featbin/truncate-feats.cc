// featbin/truncate-feats.cc

// Copyright 2016 David Snyder

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace std;

    const char *usage = "TODO";

    ParseOptions po(usage);

    int32 max_len = 100, 
          num_done = 0,
          num_err = 0;
    bool binary = true;
    po.Register("max-len", &max_len, "TODO");
    po.Register("binary", &binary, "If true, output files in binary "
                "(only relevant for single-file operation, i.e. no tables)");

    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }
    string rspecifier = po.GetArg(1), 
           wspecifier = po.GetArg(2);
    SequentialBaseFloatMatrixReader feat_reader(rspecifier);
    BaseFloatMatrixWriter feat_writer(wspecifier);

    for (; !feat_reader.Done(); feat_reader.Next()) {
      string utt = feat_reader.Key();
      const Matrix<BaseFloat> feat = feat_reader.Value();
      Matrix<BaseFloat> feat_out(std::min(feat.NumRows(), max_len), feat.NumCols());
      if (feat_out.NumRows() == feat.NumRows()) {
        feat_out.CopyFromMat(feat);
      } else {
        SubMatrix<BaseFloat> sub_feat(feat, 0, max_len, 0, feat.NumCols());
        feat_out.CopyFromMat(sub_feat);
      }
      feat_writer.Write(utt, feat_out);
      num_done++;
    }

    KALDI_LOG << "Done " << num_done << " utts, errors on "
              << num_err;

    return (num_done == 0 ? -1 : 0);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
