// bin/lat2gen-biglm-faster-mapped .cc

// Copyright      2018  Zhehuai Chen

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
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "lm/const-arpa-lm.h"
#include "rnnlm/rnnlm-lattice-rescoring.h"
#include "base/timer.h"
#include "decoder/lattice2-biglm-faster-decoder.h"


namespace kaldi {
// Takes care of output.  Returns true on success.
bool DecodeUtterance(Lattice2BiglmFasterDecoder &decoder, // not const but is really an input.
                     DecodableInterface &decodable, // not const but is really an input.
                     const TransitionModel &trans_model,
                     const fst::SymbolTable *word_syms,
                     std::string utt,
                     double acoustic_scale,
                     bool determinize,
                     bool allow_partial,
                     Int32VectorWriter *alignment_writer,
                     Int32VectorWriter *words_writer,
                     CompactLatticeWriter *compact_lattice_writer,
                     LatticeWriter *lattice_writer,
                     double *like_ptr) {  // puts utterance's like in like_ptr on success.
  using fst::VectorFst;

  if (!decoder.Decode(&decodable)) {
    KALDI_WARN << "Failed to decode file " << utt;
    return false;
  }
  if (!decoder.ReachedFinal()) {
    if (allow_partial) {
      KALDI_WARN << "Outputting partial output for utterance " << utt
                 << " since no final-state reached\n";
    } else {
      KALDI_WARN << "Not producing output for utterance " << utt
                 << " since no final-state reached and "
                 << "--allow-partial=false.\n";
      return false;
    }
  }

  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  { // First do some stuff with word-level traceback...
    VectorFst<LatticeArc> decoded;
    decoder.GetBestPath(&decoded);
    if (decoded.NumStates() == 0)
      // Shouldn't really reach this point as already checked success.
      KALDI_ERR << "Failed to get traceback for utterance " << utt;

    std::vector<int32> alignment;
    std::vector<int32> words;
    GetLinearSymbolSequence(decoded, &alignment, &words, &weight);
    num_frames = alignment.size();
    if (words_writer->IsOpen())
      words_writer->Write(utt, words);
    if (alignment_writer->IsOpen())
      alignment_writer->Write(utt, alignment);
    if (word_syms != NULL) {
      std::cerr << utt << ' ';
      for (size_t i = 0; i < words.size(); i++) {
        std::string s = word_syms->Find(words[i]);
        if (s == "")
          KALDI_ERR << "Word-id " << words[i] <<" not in symbol table.";
        std::cerr << s << ' ';
      }
      std::cerr << '\n';
    }
    likelihood = -(weight.Value1() + weight.Value2());
  }

  // Get lattice, and do determinization if requested.
  Lattice lat;
  decoder.GetRawLattice(&lat);
  if (lat.NumStates() == 0)
    KALDI_ERR << "Unexpected problem getting lattice for utterance " << utt;
  fst::Connect(&lat);
  if (determinize) {
    CompactLattice clat;
    if (!DeterminizeLatticePhonePrunedWrapper(
            trans_model,
            &lat,
            decoder.GetOptions().lattice_beam,
            &clat,
            decoder.GetOptions().det_opts))
      KALDI_WARN << "Determinization finished earlier than the beam for "
                 << "utterance " << utt;
    // We'll write the lattice without acoustic scaling.
    if (acoustic_scale != 0.0)
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &clat);
    compact_lattice_writer->Write(utt, clat);
  } else {
    Lattice fst;
    decoder.GetRawLattice(&fst);
    if (fst.NumStates() == 0)
      KALDI_ERR << "Unexpected problem getting lattice for utterance "
                << utt;
    fst::Connect(&fst); // Will get rid of this later... shouldn't have any
    // disconnected states there, but we seem to.
    if (acoustic_scale != 0.0) // We'll write the lattice without acoustic scaling
      fst::ScaleLattice(fst::AcousticLatticeScale(1.0 / acoustic_scale), &fst); 
    lattice_writer->Write(utt, fst);
  }
  KALDI_LOG << "Log-like per frame for utterance " << utt << " is "
            << (likelihood / num_frames) << " over "
            << num_frames << " frames.";
  KALDI_VLOG(2) << "Cost for utterance " << utt << " is "
                << weight.Value1() << " + " << weight.Value2();
  *like_ptr = likelihood;
  return true;
}

}



int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::Fst;
    using fst::StdArc;
    using fst::ReadFstKaldi;

    const char *usage =
        "Generate lattices using on-the-fly composition.\n"
        "User supplies LM used to generate decoding graph, and desired LM;\n"
        "this decoder applies the difference during decoding\n"
        "Usage: lat2gen-biglm-faster-mapped [options] model-in (fst-in|fsts-rspecifier) "
        "oldlm-fst-in newlm-fst-in features-rspecifier"
        " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
    ParseOptions po(usage);
    Timer timer;
    bool allow_partial = false;
    BaseFloat acoustic_scale = 0.1;
    Lattice2BiglmFasterDecoderConfig config;
    int32 max_ngram_order = 4;
    rnnlm::RnnlmComputeStateComputationOptions rnn_opts;
    bool use_carpa = false;
    
    std::string word_syms_filename, word_embedding_rxfilename;
    config.Register(&po);
    rnn_opts.Register(&po);
    po.Register("acoustic-scale", &acoustic_scale, "Scaling factor for acoustic likelihoods");

    po.Register("word-symbol-table", &word_syms_filename, "Symbol table for words [for debug output]");
    po.Register("allow-partial", &allow_partial, "If true, produce output even if end state was not reached.");
    po.Register("use-const-arpa", &use_carpa, "If true, read the old-LM file "
             "as a const-arpa file as opposed to an FST file");
    po.Register("word-embedding-rxfilename", &word_embedding_rxfilename, "If set, use rnnlm");
    po.Register("max-ngram-order", &max_ngram_order,
        "If positive, allow RNNLM histories longer than this to be identified "
        "with each other for rescoring purposes (an approximation that "
        "saves time and reduces output lattice size).");

 
    po.Read(argc, argv);

    if (po.NumArgs() < 6 || po.NumArgs() > 8) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string model_in_filename = po.GetArg(1),
        fst_in_str = po.GetArg(2),
        old_lm_fst_rxfilename = po.GetArg(3),
        new_lm_fst_rxfilename = po.GetArg(4),
        feature_rspecifier = po.GetArg(5),
        lattice_wspecifier = po.GetArg(6),
        words_wspecifier = po.GetOptArg(7),
        alignment_wspecifier = po.GetOptArg(8);
    
    TransitionModel trans_model;
    ReadKaldiObject(model_in_filename, &trans_model);

    VectorFst<StdArc> *old_lm_fst = fst::ReadAndPrepareLmFst(
        old_lm_fst_rxfilename);
    fst::BackoffDeterministicOnDemandFst<StdArc> old_lm_dfst(*old_lm_fst);
    fst::ScaleDeterministicOnDemandFst old_lm_sdfst(-1,
                                                  &old_lm_dfst);

    fst::DeterministicOnDemandFst<StdArc>* new_lm_dfst = NULL;
    VectorFst<StdArc> *new_lm_fst = NULL; 
    ConstArpaLm* const_arpa = NULL;
    CuMatrix<BaseFloat>* word_embedding_mat = NULL;
    kaldi::nnet3::Nnet *rnnlm = NULL;
    const rnnlm::RnnlmComputeStateInfo *info = NULL;

    if (word_embedding_rxfilename!="") {
      rnnlm = new kaldi::nnet3::Nnet();
      word_embedding_mat = new CuMatrix<BaseFloat>();
      ReadKaldiObject(word_embedding_rxfilename, word_embedding_mat);
      ReadKaldiObject(new_lm_fst_rxfilename, rnnlm);
      info = new rnnlm::RnnlmComputeStateInfo(rnn_opts, *rnnlm, *word_embedding_mat);
      new_lm_dfst = new rnnlm::KaldiRnnlmDeterministicFst(max_ngram_order, *info);
    } else if (use_carpa) {
      const_arpa = new ConstArpaLm();
      ReadKaldiObject(new_lm_fst_rxfilename, const_arpa);
      new_lm_dfst = new ConstArpaLmDeterministicFst(*const_arpa);
    } else {
      new_lm_fst = fst::ReadAndPrepareLmFst(
          new_lm_fst_rxfilename);
      new_lm_dfst =
        new fst::BackoffDeterministicOnDemandFst<StdArc>(*new_lm_fst);
    }

    fst::ComposeDeterministicOnDemandFst<StdArc> compose_dfst(&old_lm_sdfst,
                                                              new_lm_dfst);
    fst::CacheDeterministicOnDemandFst<StdArc> cache_dfst(&compose_dfst, 1e7);

    bool determinize = config.determinize_lattice;
    CompactLatticeWriter compact_lattice_writer;
    LatticeWriter lattice_writer;
    if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
           : lattice_writer.Open(lattice_wspecifier)))
      KALDI_ERR << "Could not open table for writing lattices: "
                 << lattice_wspecifier;

    Int32VectorWriter words_writer(words_wspecifier);

    Int32VectorWriter alignment_writer(alignment_wspecifier);

    fst::SymbolTable *word_syms = NULL;
    if (word_syms_filename != "") 
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
        KALDI_ERR << "Could not read symbol table from file "
                   << word_syms_filename;

    double tot_like = 0.0;
    kaldi::int64 frame_count = 0;
    int num_success = 0, num_fail = 0;
    double elapsed = 0;


    if (ClassifyRspecifier(fst_in_str, NULL, NULL) == kNoRspecifier) {
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      // Input FST is just one FST, not a table of FSTs.
      Fst<StdArc> *decode_fst = fst::ReadFstKaldiGeneric(fst_in_str);

      {
        Lattice2BiglmFasterDecoder decoder(*decode_fst, config, &cache_dfst);
        timer.Reset();
    
        for (; !feature_reader.Done(); feature_reader.Next()) {
          std::string utt = feature_reader.Key();
          Matrix<BaseFloat> features (feature_reader.Value());
          feature_reader.FreeCurrent();
          if (features.NumRows() == 0) {
            KALDI_WARN << "Zero-length utterance: " << utt;
            num_fail++;
            continue;
          }
                
          DecodableMatrixScaledMapped decodable(trans_model, features, acoustic_scale);

          double like;
          if (DecodeUtterance(decoder, decodable, trans_model, word_syms,
                              utt, acoustic_scale, determinize, allow_partial,
                              &alignment_writer, &words_writer,
                              &compact_lattice_writer, &lattice_writer,
                              &like)) {
            tot_like += like;
            frame_count += features.NumRows();
            num_success++;
          } else num_fail++;
        }
        elapsed = timer.Elapsed();
      }
      delete decode_fst; // delete this only after decoder goes out of scope.
    } else { // We have different FSTs for different utterances.
      assert(0);
    }
      
    KALDI_LOG << "Time taken "<< elapsed
              << "s: real-time factor assuming 100 frames/sec is "
              << (elapsed*100.0/frame_count);
    KALDI_LOG << "Done " << num_success << " utterances, failed for "
              << num_fail;
    KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
              << frame_count<<" frames.";

    delete word_syms;

    delete const_arpa;
    delete new_lm_fst;
    delete new_lm_dfst;
    delete word_embedding_mat;
    delete rnnlm;
    delete info;

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
