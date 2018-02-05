// featbin/compute-gamma-filters.cc

// Copyright 2016 Pegah Ghahremani

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
#include "feat/gammatone.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage = 
        "Generate gammatone filters.\n"
        "Usage: compute-gamma-filters [options...] <gamma-filters-output>\n";
    // 
    ParseOptions po(usage);
    GammatoneOptions gamma_opts;
    gamma_opts.Register(&po);
    po.Read(argc, argv);
    if (po.NumArgs() != 1) {
      po.PrintUsage();
      exit(1);
    }
    std::string output_filter_file = po.GetArg(1);
    BaseFloatMatrixWriter kaldi_writer;
    Matrix<BaseFloat> gamma_filters;
    GenerateGammatoneFilters(gamma_opts, &gamma_filters);
    Output ko(output_filter_file, false);
    gamma_filters.Write(ko.Stream(), false);
    return 1;
    KALDI_LOG << "Generated gammatone filters.";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}
