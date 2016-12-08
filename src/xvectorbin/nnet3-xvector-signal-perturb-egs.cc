// xvectorbin/nnet3-xvector-signal-perturb-egs.cc

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
#include "feat/signal-distort.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h" 

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;   
    typedef kaldi::int64 int64;


    const char *usage =
        "Corrupts  the examples supplied via input pipe with different type of distortions\n"
        "such as additive noise, negation, random time shifts or random distortion.\n"
        "Usage: nnet3-xvector-signal-perturb-egs [options...] <egs-especifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-xvector-signal-perturb-egs --max-shift=0.2"
        " --max-speed-perturb=0.1 --negation=true --add-noise=noise.scp --snr=10\n"
        "ark:input.egs akr:distorted.egs\n";

    ParseOptions po(usage);
    XvectorPerturbOptions perturb_opts;
    perturb_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string examples_rspecifier = po.GetArg(1),
      examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
     
    NnetExampleWriter example_writer(examples_wspecifier);

    int64 num_read = 0, num_written = 0;

    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      std::string key = example_reader.Key();
      const NnetExample &input_eg = example_reader.Value();
      const NnetIo &input_eg_io = input_eg.io[0];
      NnetExample *perturb_eg = new NnetExample();
      Matrix<BaseFloat> perturb_eg_mat, 
        input_eg_mat;
      input_eg_io.features.CopyToMat(&input_eg_mat);      
      
      PerturbExample(perturb_opts, input_eg_mat, &perturb_eg_mat);
 
      perturb_eg->io.resize(1.0);
      perturb_eg->io[0].features.SwapFullMatrix(&perturb_eg_mat);
      example_writer.Write(key, *perturb_eg);
      num_written++;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
