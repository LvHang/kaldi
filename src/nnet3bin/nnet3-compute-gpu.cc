// nnet3bin/nnet3-compute-gpu.cc

// Copyright 2018 Hang Lyu

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
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "tree/context-dep.h"
#include "fstext/fstext-lib.h"
#include "cudamatrix/cu-device.h"
#include "util/kaldi-thread.h"
#include "decoder/cuda-decoder-utils.h"
#include "decoder/cuda-lattice-decoder.h"
#include "decoder/decoder-controller.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    using fst::Fst;
    using fst::StdArc;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;

    const char *usage =
        "Read in the features. Propagate the features through raw neural "
        "network model in a batch way, and output the result chunk by chunk. \n"
        "At the same time, use multiple decoding threads to generate lattices "
        "with the chunks. \n"
        "Note: These chunks will be store in gpu so that we can avoid the "
        "PCI-E bottleneck.\n"
        "Usage: nnet3-compute-gpu [options] <nnet-in> <trans-model-in> "
        "<fst-in-str> <features-rspecifier> <lattice-wspecifier> "
        "[ words-wspecifier [alignments-wspecifier] ]\n";

    ParseOptions po(usage);
    Timer timer;
    
    // The following is batch computer options
    NnetSimpleComputationOptions opts;
    opts.acoustic_scale = 1.0;  // by default do no scaling in this recipe.

    bool apply_exp = false, use_priors = true;
    std::string use_gpu = "yes";

    std::string ivector_rspecifier,
                online_ivector_rspecifier,
                utt2spk_rspecifier;
    int32 online_ivector_period = 0;
    bool ensure_exact_final_context = false;
    int32 minibatch_size = 128;
    opts.Register(&po);

    po.Register("ivectors", &ivector_rspecifier, "Rspecifier for "
                "iVectors as vectors (i.e. not estimated online); per "
                "utterance by default, or per speaker if you provide the "
                "--utt2spk option.");
    po.Register("utt2spk", &utt2spk_rspecifier, "Rspecifier for "
                "utt2spk option used to get ivectors per speaker");
    po.Register("online-ivectors", &online_ivector_rspecifier, "Rspecifier for "
                "iVectors estimated online, as matrices.  If you supply this,"
                " you must set the --online-ivector-period option.");
    po.Register("online-ivector-period", &online_ivector_period, "Number of "
                "frames between iVectors in matrices supplied to the "
                "--online-ivectors option");
    po.Register("apply-exp", &apply_exp, "If true, apply exp function to "
                "output");
    po.Register("use-gpu", &use_gpu,
                "yes|no|optional|wait, only has effect if compiled with CUDA");
    po.Register("use-priors", &use_priors, "If true, subtract the logs of the "
                "priors stored with the model (in this case, "
                "a .mdl file is expected as input).");
    po.Register("minibatch-size", &minibatch_size, "Specify the number of "
                "utterances to be process in parallel.");
    po.Register("ensure-exact-final-context", &ensure_exact_final_context, "It "
                "controls whether we deal with those utterances whose length "
                "are shorter than chunk size specially.");

    // The following is decoder options
    CudaLatticeDecoderConfig config;
    bool allow_partial = false;
    int32 num_threads = 1;
    std::string word_syms_filename;
    po.Register("allow-partial", &allow_partial, "If true, produce output even "
                "if end state was not reached.");
    po.Register("num-threads", &num_threads, "Number of actively processing "
                "threads to run in parallel.");
    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for "
                "words [for debug output].");
    
    // The following is controller options
    int32 num_max_chunks = 512;
    int32 num_max_utts = 10;
    po.Register("num-max-chunks", &num_max_chunks, "The maximum number of "
                "chunks in GPU memory.");
    po.Register("num-max-utts", &num_max_utts, "The maximum number of "
                "utterances is processing in BatchComputer.");

    po.Read(argc, argv);

    if (po.NumArgs() < 5 || po.NumArgs() > 7) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1),
                trans_model_in_filename = po.GetArg(2),
                fst_in_str = po.GetArg(3),
                feature_rspecifier = po.GetArg(4),
                lattice_wspecifier = po.GetArg(5),
                words_wspecifier = po.GetOptArg(6),
                alignment_wspecifier = po.GetOptArg(7);

    // Read in the neural network
    Nnet raw_nnet;
    AmNnetSimple am_nnet;
    if (use_priors) {
      bool binary;
      TransitionModel trans_model;
      Input ki(nnet_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_nnet.Read(ki.Stream(), binary);
    } else {
      ReadKaldiObject(nnet_rxfilename, &raw_nnet);
    }
    Nnet &nnet = (use_priors ? am_nnet.GetNnet() : raw_nnet);
    SetBatchnormTestMode(true, &nnet);
    SetDropoutTestMode(true, &nnet);
    CollapseModel(CollapseModelConfig(), &nnet);

    Vector<BaseFloat> priors;
    if (use_priors)
      priors = am_nnet.Priors();

    // Read in the transition model
    TransitionModel trans_model;
    ReadKaldiObject(trans_model_in_filename, &trans_model);
    
    // Lattice part
    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
                       : lattice_writer.Open(lattice_wspecifier))) {
      KALDI_ERR << "Could not open table for writing lattices: "
                << lattice_wspecifier; 
    }

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") {
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename))) {
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_filename;
      }
    }

#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
    CuDevice::Instantiate().AllowMultithreading();
#endif
    double elapsed = 0.0;
    double tot_like = 0.0;
    int64 frame_count = 0;
    int32 num_success = 0, num_fail = 0;
    
    // GPU version of WFST
    CudaFst decode_fst_cuda;
    // The input FST is just one FST, not a table of FSTs
    Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);
    decode_fst_cuda.Initialize(*decode_fst);
   
    timer.Reset();
    // Initalize Controller
    Controller controller(opts, nnet, priors, online_ivector_period,
                          ensure_exact_final_context, minibatch_size,
                          feature_rspecifier, online_ivector_rspecifier,
                          ivector_rspecifier, utt2spk_rspecifier,
                          decode_fst_cuda, num_threads,
                          config, trans_model, word_syms,
                          config.acoustic_scale, determinize, allow_partial,
                          &alignment_writer, &words_writer,
                          &compact_lattice_writer, &lattice_writer, &tot_like,
                          &frame_count, &num_success, &num_fail, NULL,
                          num_max_chunks, num_max_utts);
    controller.Run();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif
    elapsed = timer.Elapsed();
    KALDI_LOG << "Time taken "<< elapsed << "s";
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
