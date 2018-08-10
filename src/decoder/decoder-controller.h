// decoder/decoder-controller.h

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

#ifndef KALDI_DECODER_DECODER_CONTROLLER_H_
#define KALDI_DECODER_DECODER_CONTROLLER_H_

#include "base/timer.h"
#include "util/kaldi-thread.h"
#include "itf/options-itf.h"
#include "decoder/lattice-faster-decoder.h"
#include "decoder/lattice-simple-decoder.h"
#include "nnet3/nnet-am-decodable-simple.h"

#if HAVE_CUDA == 1
#include "decoder/lattice-faster-decoder-cuda.h"
#endif

namespace kaldi {

  
/** This struct stores the waiting utterances to be used in
    multi-threaded gpu decoding.  */
class WaitingUtteranceRepository {
 public:
  /// The following function is called by the code that reads in the utterances.
  void AcceptUtterance(std::string utt_id) {
    empty_semaphore_.Wait();
    utts_mutex_.Lock();
    utts_.push_back(utt_id);
    utts_mutex_.Unlock();
    full_semaphore_.Signal();
  }

  /// The following function is called by the code that reads in the utterances,
  /// when we're done reading utterances; it signals this way to this class
  /// that the stream is now empty
  void UtterancesDone() {
    for (int32 i = 0; i < buffer_size_; i++)
      empty_semaphore_.Wait();
    utts_mutex_.Lock();
    KALDI_ASSERT(utts_.empty());
    utts_mutex_.Unlock();
    done_ = true;
    full_semaphore_.Signal();
  }

  /// This function is called by the code that does gpu decoding.  If there is
  /// an example available it will provide it, or it will sleep till one is
  /// available.  It returns NULL when there are no utterances left and
  /// UtterancesDone() has been called.
  std::string *ProvideUtterance() {
    full_semaphore_.Wait();
    if (done_) {
      KALDI_ASSERT(utts_.empty());
      full_semaphore_.Signal(); // Increment the semaphore so
      // the call by the next thread will not block.
      return NULL; // no examples to return-- all finished.
    } else {
      utts_mutex_.Lock();
      KALDI_ASSERT(!utts_.empty());
      std::string *ans = utts_.front();
      utts_.pop_front();
      utts_mutex_.Unlock();
      empty_semaphore_.Signal();
      return ans;
    }
  }


  bool Done() {
    if (done_) {
      return true;
    }
    return false;
  }


  WaitingUtterancesRepository(int32 buffer_size = 128):
                                      buffer_size_(buffer_size),
                                      empty_semaphore_(buffer_size_),
                                      done_(false) { }
 private:
  int32 buffer_size_;
  Semaphore full_semaphore_;
  Semaphore empty_semaphore_;
  Mutex utts_mutex_; // mutex we lock to modify examples_.

  std::deque<std::string> utts_;
  bool done_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(WaitingUtteranceRepository);
};


class Controller {
 public:
  // Initializer sets various variables.
  Controller(
    const NnetSimpleComputations &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors,
    int32 online_ivector_period,
    bool ensure_exact_final_context,
    int32 minibatch_size,
    const SequentialBaseFloatMatrixReader &feature_reader,
    const RandomAccessBaseFloatMatrixReader &online_ivector_reader,
    const RandomAccessBaseFloatVectorReaderMapped &ivector_reader,
    CudaFst &decode_fst_cuda,
    CudaLatticeDecoderConfig &config,
    const TransitionModel &trans_model,
    const fst::SymbolTable *word_syms,
    BaseFloat acoustic_scale,
    bool determinize,
    bool allow_partial,
    Int32VectorWriter *alignments_writer,
    Int32VectorWriter *words_writer,
    CompactLatticeWriter *compact_lattice_writer,
    LatticeWriter *lattice_writer,
    double *like_sum, // on success, adds likelihood to this.
    int64 *frame_sum, // on success, adds #frames to this.
    int32 *num_done, // on success (including partial decode), increments this.
    int32 *num_err,  // on failure, increments this.
    int32 *num_partial // If partial decode(final-state not reached),increment.
    );  

  // The main function. In it, we create multi gpu decoding threads with
  // class MultiThreader. We let the BatchComputer read in the features and
  // compute.
  void Run();
 private:
  // Some variables for BatchComputer
  NnetSimpleComputationOptions opts_;
  const Nnet &nnet_;
  const VectorBase<BaseFloat> &priors_;
  int32 online_ivector_period_;
  bool ensure_exact_final_context_;
  int32 minibatch_size_;
  const SequentialBaseFloatMatrixReader &feature_reader_;
  const RandomAccessBaseFloatMatrixReader &online_ivector_reader_;
  const RandomAccessBaseFloatVectorReaderMapped &ivector_reader_;
  BatchComputer* batch_computer_; // ownership of the pointer
  
  // Some variables for Decoder
  const CudaFst &decode_fst_cuda_;
  CudaLatticeDecoderConfig &config_;
  const TransitionModel &trans_model_;
  const fst::SymbolTable *word_syms_;
  BaseFloat acoustic_scale_;
  bool determinize_;
  bool allow_partial_;
  Int32VectorWriter *alignments_writer_;
  Int32VectorWriter *words_writer_;
  CompactLatticeWriter *compace_lattice_writer_;
  LatticeWriter *lattice_writer;
  double *like_sum_;
  int64 *frame_sum_;
  int32 *num_done_;
  int32 *num_err_;
  int32 *num_partial_;

  // Some variables which are used to connect BatchComputer and Decoder
  // This part can be understand as a public territory.
  Mutex *utt_mutex_;
  WaitingUtteranceRepository *repository_;

  int32 num_max_chunks_; // The maximum number of chunks in GPU memory
  int32 num_max_utts_;  // The maximum number of utterances is processing in
                        // BatchComputer
  int32 chunk_counter_ = 0; // It is used to remember the number of chunks is
                            // available now
  int32 num_threads_;  // The number of decoding threads
  // The key is utt_id. The value is a queue which contains all the pointers
  // to each posterior chunk. In BatchComputer, we will new the "CuMatrix"
  // space to store posterior chunk.
  unordered_map<std::string,
    std::queue<const CuMatrix<BaseFloat>* > > finished_inf_utts_;

  // record the number of chunk is processed for each utterance. It will be 
  // pass to BatchComputer to show what point the decoders are at in
  // each utterance.
  unordered_map<std::string, size_t> finished_dec_utts_;
  unordered_map<std::string, bool> is_end_;
  unordered_map<std::string, Semaphore> utts_semaphores_; 
};



} // end namespace kaldi
