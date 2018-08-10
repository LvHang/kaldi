// decoder/decoder-controller.cc

// Copyright   2018  Hang Lyu

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

#include "decoder/decoder-controller.cc"

Controller::Controller(
    const NnetSimpleComputations &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors,
    int32 online_ivector_period,
    bool ensure_exact_final_context,
    int32 minibatch_size,
    const SequentialBaseFloatMatrixReader &feature_reader,
    const RandomAccessBaseFloatMatrixReader &online_ivector_reader,
    const RandomAccessBaseFloatVectorReaderMapped &ivector_reader,
    int32 num_threads,
    CudaFst &decode_fst_cuda,
    CudaLatticeDecoderConfig &config,
    const TransitionModel &trans_model,
    const fst::SymbolTable *word_syms,
    BaseFloat acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writernt32 num_threads,,
    LatticeWriter *lattice_writer,
    double *like_sum, // on success, adds likelihood to this.
    int64 *frame_sum, // on success, adds #frames to this.
    int32 *num_done, // on success (including partial decode), increments this.
    int32 *num_err,  // on failure, increments this.
    int32 *num_partial // If partial decode(final-state not reached),increment.
    int32 num_max_chunks_; // The maximum number of chunks in GPU memory
    int32 num_max_utts_;  // The maximum number of utterances is processing
    ) 
    : opts_(opts), nnet_(nnet), priors_(priors),
    online_ivector_period_(online_ivector_period),
    ensure_exact_final_context_(ensure_exact_final_context),
    minibatch_size_(minibatch_size), feature_reader_(feature_reader),
    online_ivector_reader_(online_ivector_reader),
    ivector_reader_(ivector_reader), num_threads_(num_threads),
    decode_fst_cuda_(decode_fst_cuda),
    config_(config), trans_model_(trans_model),
    word_syms_(word_syms), acoustic_scale_(acoustic_scale),
    determinize_(determinize), allow_partial_(allow_partial),
    alignments_writer_(alignments_writer),
    words_writer_(words_writer),
    compact_lattice_writer_(compact_lattice_writer),
    lattice_writer_(lattice_writer),
    like_sum_(like_sum), frame_sum_(frame_sum),
    num_done_(num_done), num_err_(num_err),
    num_partial_(num_partial), num_max_chunks_(num_max_chunks),
    num_max_utts_(num_max_utts) { 
  repository_ = new WaitingUtterancesRepository(num_max_utts_);
  // Build the batch_computer_, it will be used to accept input in
  // Controller::Run() and to do compute in a single thread which
  // is wrapped by BatchComputerClass.
  batch_computer_ = new BatchComputer(opts_, nnet_, priors_,
    online_ivector_period, ensure_exact_final_context_, minibatch_size_);  
}


void Controller::Run() {
  // Create num_threads_ threads to deal with decoding.
  DecodeUtteranceLatticeClassCuda c(decode_fst_cuda_, config_, trans_model_,
      word_syms_, acoustic_scale_, determinize_, allow_partial_,
      alignments_writer_, words_writer_, compact_lattice_writer_,
      lattice_writer_, like_sum_, frame_sum_, num_done_, num_err_,
      num_partial_, utt_mutex_, repository_, finished_dec_utts_,
      is_end_, utts_semaphores_);
  MultiThreader<DecodeUtteranceLatticeClassCuda> m(num_threads_, c);

  // Create one thread to do NnetCompute in batch version.
  BatchComputerClass c1(batch_computer_, num_max_chunks_, &chunk_counter_, 
      &finished_inf_utts_, finished_dec_utts_, &is_end_, &utts_semaphores_,
      &repository_, &utt_mutex_);
  MultiThreader<BatchComputerClass> m1(1, c1);

  for (; !feature_reader.Done(); feature_reader.Next()) {
    std::string utt = feature_reader.Key();
    // Note: the reference that Value() returns is only valid until call
    // Next()
    const Matrix<BaseFloat> *features =
      new Matrix<BaseFloat>(feature_reader.Value());
    if (features->NumRows() == 0) {
      KALDI_WARN << "Zero-length utterance: " << utt;
      continue;
    }
    const Matrix<BaseFloat> *online_ivectors = NULL;
    const Vector<BaseFloat> *ivector = NULL;
    if (!ivector_rspecifier.empty()) {
      if (!ivector_reader.HasKey(utt)) {
        KALDI_WARN << "No iVector available for utterance " << utt;
        continue;
      } else {
        ivector = new Vector<BaseFloat>(ivector_reader.Value(utt));
      }
    }
    if (!online_ivector_rspecifier.empty()) {
      if (!online_ivector_reader.HasKey(utt)) {
        KALDI_WARN << "No online iVector available for utterance " << utt;
        continue;
      } else {
        online_ivectors = new Matrix<BaseFloat>(
          online_ivector_reader.Value(utt));
      }
    }
    repository_.AcceptUtterance(utt);
    batch_computer_.AcceptInput(utt, features, ivector, online_ivectors);
  }
  repository_.UtterancesDone();
}
