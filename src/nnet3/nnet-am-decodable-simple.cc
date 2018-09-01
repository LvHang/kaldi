// nnet3/nnet-am-decodable-simple.cc

// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

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

#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"

namespace kaldi {
namespace nnet3 {


DecodableNnetSimple::DecodableNnetSimple(
    const NnetSimpleComputationOptions &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors,
    const MatrixBase<BaseFloat> &feats,
    CachingOptimizingCompiler *compiler,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    opts_(opts),
    nnet_(nnet),
    output_dim_(nnet_.OutputDim("output")),
    log_priors_(priors),
    feats_(feats),
    ivector_(ivector), online_ivector_feats_(online_ivectors),
    online_ivector_period_(online_ivector_period),
    compiler_(*compiler),
    current_log_post_subsampled_offset_(0) {
  num_subsampled_frames_ =
      (feats_.NumRows() + opts_.frame_subsampling_factor - 1) /
      opts_.frame_subsampling_factor;
  KALDI_ASSERT(IsSimpleNnet(nnet));
  compiler_.GetSimpleNnetContext(&nnet_left_context_, &nnet_right_context_);
  KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
  KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period <= 0 &&
                 "You need to set the --online-ivector-period option!"));
  log_priors_.ApplyLog();
  CheckAndFixConfigs();
}


DecodableAmNnetSimple::DecodableAmNnetSimple(
    const NnetSimpleComputationOptions &opts,
    const TransitionModel &trans_model,
    const AmNnetSimple &am_nnet,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period,
    CachingOptimizingCompiler *compiler):
    compiler_(am_nnet.GetNnet(), opts.optimize_config, opts.compiler_config),
    decodable_nnet_(opts, am_nnet.GetNnet(), am_nnet.Priors(),
                    feats, compiler != NULL ? compiler : &compiler_,
                    ivector, online_ivectors,
                    online_ivector_period),
    trans_model_(trans_model) {
  // note: we only use compiler_ if the passed-in 'compiler' is NULL.
}



BaseFloat DecodableAmNnetSimple::LogLikelihood(int32 frame,
                                               int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdf(transition_id);
  return decodable_nnet_.GetOutput(frame, pdf_id);
}

int32 DecodableNnetSimple::GetIvectorDim() const {
  if (ivector_ != NULL)
    return ivector_->Dim();
  else if (online_ivector_feats_ != NULL)
    return online_ivector_feats_->NumCols();
  else
    return 0;
}

void DecodableNnetSimple::EnsureFrameIsComputed(int32 subsampled_frame) {
  KALDI_ASSERT(subsampled_frame >= 0 &&
               subsampled_frame < num_subsampled_frames_);
  int32 feature_dim = feats_.NumCols(),
      ivector_dim = GetIvectorDim(),
      nnet_input_dim = nnet_.InputDim("input"),
      nnet_ivector_dim = std::max<int32>(0, nnet_.InputDim("ivector"));
  if (feature_dim != nnet_input_dim)
    KALDI_ERR << "Neural net expects 'input' features with dimension "
              << nnet_input_dim << " but you provided "
              << feature_dim;
  if (ivector_dim != std::max<int32>(0, nnet_.InputDim("ivector")))
    KALDI_ERR << "Neural net expects 'ivector' features with dimension "
              << nnet_ivector_dim << " but you provided " << ivector_dim;

  int32 current_subsampled_frames_computed = current_log_post_.NumRows(),
      current_subsampled_offset = current_log_post_subsampled_offset_;
  KALDI_ASSERT(subsampled_frame < current_subsampled_offset ||
               subsampled_frame >= current_subsampled_offset +
               current_subsampled_frames_computed);

  // all subsampled frames pertain to the output of the network,
  // they are output frames divided by opts_.frame_subsampling_factor.
  int32 subsampling_factor = opts_.frame_subsampling_factor,
      subsampled_frames_per_chunk = opts_.frames_per_chunk / subsampling_factor,
      start_subsampled_frame = subsampled_frame,
      num_subsampled_frames = std::min<int32>(num_subsampled_frames_ -
                                              start_subsampled_frame,
                                              subsampled_frames_per_chunk),
      last_subsampled_frame = start_subsampled_frame + num_subsampled_frames - 1;
  KALDI_ASSERT(num_subsampled_frames > 0);
  // the output-frame numbers are the subsampled-frame numbers
  int32 first_output_frame = start_subsampled_frame * subsampling_factor,
      last_output_frame = last_subsampled_frame * subsampling_factor;

  KALDI_ASSERT(opts_.extra_left_context >= 0 && opts_.extra_right_context >= 0);
  int32 extra_left_context = opts_.extra_left_context,
      extra_right_context = opts_.extra_right_context;
  if (first_output_frame == 0 && opts_.extra_left_context_initial >= 0)
    extra_left_context = opts_.extra_left_context_initial;
  if (last_subsampled_frame == num_subsampled_frames_ - 1 &&
      opts_.extra_right_context_final >= 0)
    extra_right_context = opts_.extra_right_context_final;
  int32 left_context = nnet_left_context_ + extra_left_context,
      right_context = nnet_right_context_ + extra_right_context;
  int32 first_input_frame = first_output_frame - left_context,
      last_input_frame = last_output_frame + right_context,
      num_input_frames = last_input_frame + 1 - first_input_frame;
  Vector<BaseFloat> ivector;
  GetCurrentIvector(first_output_frame,
                    last_output_frame - first_output_frame,
                    &ivector);

  Matrix<BaseFloat> input_feats;
  if (first_input_frame >= 0 &&
      last_input_frame < feats_.NumRows()) {
    SubMatrix<BaseFloat> input_feats(feats_.RowRange(first_input_frame,
                                                     num_input_frames));
    DoNnetComputation(first_input_frame, input_feats, ivector,
                      first_output_frame, num_subsampled_frames);
  } else {
    Matrix<BaseFloat> feats_block(num_input_frames, feats_.NumCols());
    int32 tot_input_feats = feats_.NumRows();
    for (int32 i = 0; i < num_input_frames; i++) {
      SubVector<BaseFloat> dest(feats_block, i);
      int32 t = i + first_input_frame;
      if (t < 0) t = 0;
      if (t >= tot_input_feats) t = tot_input_feats - 1;
      const SubVector<BaseFloat> src(feats_, t);
      dest.CopyFromVec(src);
    }
    DoNnetComputation(first_input_frame, feats_block, ivector,
                      first_output_frame, num_subsampled_frames);
  }
}

// note: in the normal case (with no frame subsampling) you can ignore the
// 'subsampled_' in the variable name.
void DecodableNnetSimple::GetOutputForFrame(int32 subsampled_frame,
                                            VectorBase<BaseFloat> *output) {
  if (subsampled_frame < current_log_post_subsampled_offset_ ||
      subsampled_frame >= current_log_post_subsampled_offset_ +
      current_log_post_.NumRows())
    EnsureFrameIsComputed(subsampled_frame);
  output->CopyFromVec(current_log_post_.Row(
      subsampled_frame - current_log_post_subsampled_offset_));
}

void DecodableNnetSimple::GetCurrentIvector(int32 output_t_start,
                                            int32 num_output_frames,
                                            Vector<BaseFloat> *ivector) {
  if (ivector_ != NULL) {
    *ivector = *ivector_;
    return;
  } else if (online_ivector_feats_ == NULL) {
    return;
  }
  KALDI_ASSERT(online_ivector_period_ > 0);
  // frame_to_search is the frame that we want to get the most recent iVector
  // for.  We choose a point near the middle of the current window, the concept
  // being that this is the fairest comparison to nnet2.   Obviously we could do
  // better by always taking the last frame's iVector, but decoding with
  // 'online' ivectors is only really a mechanism to simulate online operation.
  int32 frame_to_search = output_t_start + num_output_frames / 2;
  int32 ivector_frame = frame_to_search / online_ivector_period_;
  KALDI_ASSERT(ivector_frame >= 0);
  if (ivector_frame >= online_ivector_feats_->NumRows()) {
    int32 margin = ivector_frame - (online_ivector_feats_->NumRows() - 1);
    if (margin * online_ivector_period_ > 50) {
      // Half a second seems like too long to be explainable as edge effects.
      KALDI_ERR << "Could not get iVector for frame " << frame_to_search
                << ", only available till frame "
                << online_ivector_feats_->NumRows()
                << " * ivector-period=" << online_ivector_period_
                << " (mismatched --ivector-period?)";
    }
    ivector_frame = online_ivector_feats_->NumRows() - 1;
  }
  *ivector = online_ivector_feats_->Row(ivector_frame);
}


void DecodableNnetSimple::DoNnetComputation(
    int32 input_t_start,
    const MatrixBase<BaseFloat> &input_feats,
    const VectorBase<BaseFloat> &ivector,
    int32 output_t_start,
    int32 num_subsampled_frames) {
  ComputationRequest request;
  request.need_model_derivative = false;
  request.store_component_stats = false;

  bool shift_time = true; // shift the 'input' and 'output' to a consistent
  // time, to take advantage of caching in the compiler.
  // An optimization.
  int32 time_offset = (shift_time ? -output_t_start : 0);

  // First add the regular features-- named "input".
  request.inputs.reserve(2);
  request.inputs.push_back(
      IoSpecification("input", time_offset + input_t_start,
                      time_offset + input_t_start + input_feats.NumRows()));
  if (ivector.Dim() != 0) {
    std::vector<Index> indexes;
    indexes.push_back(Index(0, 0, 0));
    request.inputs.push_back(IoSpecification("ivector", indexes));
  }
  IoSpecification output_spec;
  output_spec.name = "output";
  output_spec.has_deriv = false;
  int32 subsample = opts_.frame_subsampling_factor;
  output_spec.indexes.resize(num_subsampled_frames);
  // leave n and x values at 0 (the constructor sets these).
  for (int32 i = 0; i < num_subsampled_frames; i++)
    output_spec.indexes[i].t = time_offset + output_t_start + i * subsample;
  request.outputs.resize(1);
  request.outputs[0].Swap(&output_spec);

  std::shared_ptr<const NnetComputation> computation = compiler_.Compile(request);
  Nnet *nnet_to_update = NULL;  // we're not doing any update.
  NnetComputer computer(opts_.compute_config, *computation,
                        nnet_, nnet_to_update);

  CuMatrix<BaseFloat> input_feats_cu(input_feats);
  computer.AcceptInput("input", &input_feats_cu);
  CuMatrix<BaseFloat> ivector_feats_cu;
  if (ivector.Dim() > 0) {
    ivector_feats_cu.Resize(1, ivector.Dim());
    ivector_feats_cu.Row(0).CopyFromVec(ivector);
    computer.AcceptInput("ivector", &ivector_feats_cu);
  }
  computer.Run();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  // subtract log-prior (divide by prior)
  if (log_priors_.Dim() != 0)
    cu_output.AddVecToRows(-1.0, log_priors_);
  // apply the acoustic scale
  cu_output.Scale(opts_.acoustic_scale);
  current_log_post_.Resize(0, 0);
  // the following statement just swaps the pointers if we're not using a GPU.
  cu_output.Swap(&current_log_post_);
  current_log_post_subsampled_offset_ = output_t_start / subsample;
}

void DecodableNnetSimple::CheckAndFixConfigs() {
  static bool warned_frames_per_chunk = false;
  int32 nnet_modulus = nnet_.Modulus();
  if (opts_.frame_subsampling_factor < 1 ||
      opts_.frames_per_chunk < 1)
    KALDI_ERR << "--frame-subsampling-factor and --frames-per-chunk must be > 0";
  KALDI_ASSERT(nnet_modulus > 0);
  int32 n = Lcm(opts_.frame_subsampling_factor, nnet_modulus);

  if (opts_.frames_per_chunk % n != 0) {
    // round up to the nearest multiple of n.
    int32 frames_per_chunk = n * ((opts_.frames_per_chunk + n - 1) / n);
    if (!warned_frames_per_chunk) {
      warned_frames_per_chunk = true;
      if (nnet_modulus == 1) {
        // simpler error message.
        KALDI_LOG << "Increasing --frames-per-chunk from "
                  << opts_.frames_per_chunk << " to "
                  << frames_per_chunk << " to make it a multiple of "
                  << "--frame-subsampling-factor="
                  << opts_.frame_subsampling_factor;
      } else {
        KALDI_LOG << "Increasing --frames-per-chunk from "
                  << opts_.frames_per_chunk << " to "
                  << frames_per_chunk << " due to "
                  << "--frame-subsampling-factor="
                  << opts_.frame_subsampling_factor << " and "
                  << "nnet shift-invariance modulus = " << nnet_modulus;
      }
    }
    opts_.frames_per_chunk = frames_per_chunk;
  }
}


DecodableAmNnetSimpleParallel::DecodableAmNnetSimpleParallel(
    const NnetSimpleComputationOptions &opts,
    const TransitionModel &trans_model,
    const AmNnetSimple &am_nnet,
    const MatrixBase<BaseFloat> &feats,
    const VectorBase<BaseFloat> *ivector,
    const MatrixBase<BaseFloat> *online_ivectors,
    int32 online_ivector_period):
    compiler_(am_nnet.GetNnet(), opts.optimize_config, opts.compiler_config),
    trans_model_(trans_model),
    feats_copy_(NULL),
    ivector_copy_(NULL),
    online_ivectors_copy_(NULL),
    decodable_nnet_(NULL) {
  try {
    feats_copy_ = new Matrix<BaseFloat>(feats);
    if (ivector != NULL)
      ivector_copy_ = new Vector<BaseFloat>(*ivector);
    if (online_ivectors != NULL)
      online_ivectors_copy_ = new Matrix<BaseFloat>(*online_ivectors);
    decodable_nnet_ = new DecodableNnetSimple(opts, am_nnet.GetNnet(),
                                              am_nnet.Priors(), *feats_copy_,
                                              &compiler_, ivector_copy_,
                                              online_ivectors_copy_,
                                              online_ivector_period);

  } catch (...) {
    DeletePointers();
    KALDI_ERR << "Error occurred in constructor (see above)";
  }
}

void DecodableAmNnetSimpleParallel::DeletePointers() {
  // delete[] does nothing for null pointers, so we have no checks.
  delete decodable_nnet_;
  decodable_nnet_ = NULL;
  delete feats_copy_;
  feats_copy_ = NULL;
  delete ivector_copy_;
  ivector_copy_ = NULL;
  delete online_ivectors_copy_;
  online_ivectors_copy_ = NULL;
}


BaseFloat DecodableAmNnetSimpleParallel::LogLikelihood(int32 frame,
                                                       int32 transition_id) {
  int32 pdf_id = trans_model_.TransitionIdToPdf(transition_id);
  return decodable_nnet_->GetOutput(frame, pdf_id);
}


BatchComputer::BatchComputer(
    const NnetSimpleComputationOptions &opts,
    const Nnet &nnet,
    const VectorBase<BaseFloat> &priors,
    int32 online_ivector_period,
    bool  ensure_exact_final_context,
    int32 minibatch_size):
    opts_(opts),
    nnet_(nnet),
    output_dim_(nnet_.OutputDim("output")),
    log_priors_(priors),
    online_ivector_period_(online_ivector_period),
    compiler_(nnet_, opts_.optimize_config),
    ensure_exact_final_context_(ensure_exact_final_context),
    minibatch_size_(minibatch_size) {
  // Check the Nnet
  KALDI_ASSERT(IsSimpleNnet(nnet));
  ComputeSimpleNnetContext(nnet, &nnet_left_context_, &nnet_right_context_);
  log_priors_.ApplyLog();
  CheckAndFixConfigs();
  // Prepare ComputationRequest and store a NnetComputation object
  PrepareComputationRequest();
}


void BatchComputer::CheckAndFixConfigs() {
  static bool warned_frames_per_chunk = false;
  int32 nnet_modulus = nnet_.Modulus();
  if (opts_.frame_subsampling_factor < 1 ||
      opts_.frames_per_chunk < 1) {
    KALDI_ERR << "--frame-subsampling-factor and "
              << "--frames-per-chunk must be > 0";
  }
  KALDI_ASSERT(nnet_modulus > 0);
  int32 n = Lcm(opts_.frame_subsampling_factor, nnet_modulus);

  if (opts_.frames_per_chunk % n != 0) {
    // round up to the nearest multiple of n.
    int32 frames_per_chunk = n * ((opts_.frames_per_chunk + n - 1) / n);
    if (!warned_frames_per_chunk) {
      warned_frames_per_chunk = true;
      if (nnet_modulus == 1) {
        // simpler error message.
        KALDI_LOG << "Increasing --frames-per-chunk from "
                  << opts_.frames_per_chunk << " to "
                  << frames_per_chunk << " to make it a multiple of "
                  << "--frame-subsampling-factor="
                  << opts_.frame_subsampling_factor;
      } else {
        KALDI_LOG << "Increasing --frames-per-chunk from "
                  << opts_.frames_per_chunk << " to "
                  << frames_per_chunk << " due to "
                  << "--frame-subsampling-factor="
                  << opts_.frame_subsampling_factor << " and "
                  << "nnet shift-invariance modulus = " << nnet_modulus;
      }
    }
    opts_.frames_per_chunk = frames_per_chunk;
  }
}


void BatchComputer::PrepareComputationRequest() {
  int32 input_dim = nnet_.InputDim("input"),
        ivector_dim = nnet_.InputDim("ivector");
  KALDI_ASSERT(input_dim > 0);

  std::pair<int32, int32> context;
  context = std::make_pair(opts_.extra_left_context + nnet_left_context_,
                           opts_.extra_right_context + nnet_right_context_);
  batch_info_[context] = new BatchInfoQueue();
  context_order_record_.push_back(context);

  if (opts_.extra_left_context_initial != opts_.extra_left_context &&
      opts_.extra_left_context_initial >= 0) {
    context = std::make_pair(opts_.extra_left_context_initial +
                             nnet_left_context_,
                             opts_.extra_right_context + nnet_right_context_);
    batch_info_[context] = new BatchInfoQueue();
    context_order_record_.push_front(context);
  }

  if (opts_.extra_right_context_final != opts_.extra_right_context &&
      opts_.extra_right_context_final >= 0) {
    context = std::make_pair(opts_.extra_left_context + nnet_left_context_,
                             opts_.extra_right_context_final +
                             nnet_right_context_);
    batch_info_[context] = new BatchInfoQueue();
    context_order_record_.push_back(context);
  }

  for (BatchInfoMap::iterator iter =
       batch_info_.begin(); iter != batch_info_.end(); iter++) {
    context = iter->first;
    int32 left_context = context.first,
          right_context = context.second;
    // Actually, when opts_.frame_subsampling_factor > 1,  the chunk size will
    // be less than opts_.frames_per_chunk.
    int32 num_input_rows = left_context + opts_.frames_per_chunk +
      right_context - opts_.frame_subsampling_factor + 1,
          num_output_rows = opts_.frames_per_chunk /
                            opts_.frame_subsampling_factor;

    ComputationRequest* request = new ComputationRequest();
    request->need_model_derivative = false;
    request->store_component_stats = false;
    request->inputs.reserve(2);
    request->outputs.reserve(1);
    std::vector<Index> input_indexes, ivector_indexes, output_indexes;

    for (int32 n = 0; n < minibatch_size_; n++) {
      for (int32 t = 0; t < num_input_rows; t++) {
        input_indexes.push_back(Index(n, t - left_context, 0));
      }
    }
    if (ivector_dim > 0) {
      for (int32 n = 0; n < minibatch_size_; n++) {
        ivector_indexes.push_back(Index(n, 0, 0));
      }
    }
    for (int32 n = 0; n < minibatch_size_; n++) {
      for (int32 i = 0; i < num_output_rows; i++) {
        output_indexes.push_back(Index(n, i * opts_.frame_subsampling_factor,
                                       0));
      }
    }

    request->inputs.push_back(IoSpecification("input", input_indexes));
    if (ivector_dim > 0) {
      request->inputs.push_back(IoSpecification("ivector", ivector_indexes));
    }
    request->outputs.push_back(IoSpecification("output", output_indexes));
    context_to_request_[context] = request;
  }

  if (ensure_exact_final_context_) {
    context = std::make_pair(-1, -1);
    batch_info_[context] = new BatchInfoQueue();
    context_order_record_.push_back(context);
  }
}


void BatchComputer::AcceptInput(
    const std::string &utt_id,
    const Matrix<BaseFloat> *feats,
    const Vector<BaseFloat> *ivector,
    const Matrix<BaseFloat> *online_ivectors) {
  // Check the input fits with the nnet.
  CheckInput(feats, ivector, online_ivectors);

  utt_list_.push_back(utt_id);
  feats_[utt_id] = feats;
  if ( ivector != NULL ) {
    ivectors_[utt_id] = ivector;
  }
  if ( online_ivectors != NULL ) {
    online_ivector_feats_[utt_id] = online_ivectors;
  }
  is_computed_[utt_id] = false;

  // Compute number of output frames
  int32 cur_num_subsampled_frames =
    (feats->NumRows() + opts_.frame_subsampling_factor - 1) /
    opts_.frame_subsampling_factor;

  num_subsampled_frames_[utt_id] = cur_num_subsampled_frames;
  prepared_chunk_record_[utt_id] = 0;
  lasttime_finished_dec_utts_[utt_id] = 0;
  num_chunks_[utt_id] = ceil(feats->NumRows() * 1.0 /
                             opts_.frames_per_chunk);
}


void BatchComputer::CheckInput(const Matrix<BaseFloat> *feats,
                                   const Vector<BaseFloat> *ivector,
                                   const Matrix<BaseFloat> *online_ivectors) {
  KALDI_ASSERT(!(ivector != NULL && online_ivectors != NULL));
  KALDI_ASSERT(!(online_ivectors != NULL && online_ivector_period_ <= 0 &&
                 "You need to set the --online-ivector-period option!"));
  int32 feature_dim = feats->NumCols();
  int32 ivector_dim = 0;
  if (ivector != NULL) {
    ivector_dim = ivector->Dim();
  }
  if (online_ivectors != NULL) {
    ivector_dim = online_ivectors->NumCols();
  }
  int32 nnet_input_dim = nnet_.InputDim("input"),
        nnet_ivector_dim = std::max<int32>(0, nnet_.InputDim("ivector"));
  if (feature_dim != nnet_input_dim)
    KALDI_ERR << "Neural net expects 'input' features with dimension "
              << nnet_input_dim << " but you provided "
              << feature_dim;
  if (ivector_dim != std::max<int32>(0, nnet_.InputDim("ivector")))
    KALDI_ERR << "Neural net expects 'ivector' features with dimension "
              << nnet_ivector_dim << " but you provided " << ivector_dim;
}


void BatchComputer::DoNnetComputationOnes(
  std::unordered_map<std::string,
                     std::queue<const CuMatrix<BaseFloat>*> > *result,
  std::unordered_map<std::string, bool> *is_end,
  std::unordered_map<std::string, Semaphore* > *utts_semaphores) {
  std::pair<int32, int32> current_context(-1, -1);
  if (batch_info_[current_context]->size() == 0) {
    return;
  }
  BatchInfoQueue* current_queue = batch_info_[current_context];
  std::string utt_id;
  int32 first_input_frame, last_input_frame;
  int32 first_subsampled_frame, last_subsampled_frame;
  int32 output_offset;

  // Prepare ComputationRequest
  ComputationRequest request;
  request.need_model_derivative = false;
  request.store_component_stats = false;
  request.inputs.reserve(2);
  request.outputs.reserve(1);
  std::vector<Index> input_indexes, ivector_indexes, output_indexes;
  int32 ivector_dim = nnet_.InputDim("ivector");

  int32 tot_input_rows = 0;
  int32 extra_left_context = opts_.extra_left_context;
  if (opts_.extra_left_context_initial >= 0) {
    extra_left_context = opts_.extra_left_context_initial;
  }
  int32 left_context = nnet_left_context_ + extra_left_context;

  BatchInfoQueue::iterator iter = current_queue->begin();
  for (int32 n = 0; iter != current_queue->end(); iter++, n++) {
    std::tie(utt_id,
             first_input_frame, last_input_frame,
             first_subsampled_frame, last_subsampled_frame,
             output_offset) = *iter;
    int32 num_input_rows = last_input_frame - first_input_frame + 1;
    int32 num_output_rows = last_subsampled_frame - first_subsampled_frame + 1;
    tot_input_rows += num_input_rows;

    for (int32 t = 0; t < num_input_rows; t++) {
      input_indexes.push_back(Index(n, t - left_context, 0));
    }
    if (ivector_dim > 0) {
      ivector_indexes.push_back(Index(n, 0, 0));
    }
    for (int32 i = 0; i < num_output_rows; i++) {
      output_indexes.push_back(Index(n, i * opts_.frame_subsampling_factor, 0));
    }
  }
  request.inputs.push_back(IoSpecification("input", input_indexes));
  if (ivector_dim > 0) {
    request.inputs.push_back(IoSpecification("ivector", ivector_indexes));
  }
  request.outputs.push_back(IoSpecification("output", output_indexes));
  // Prepare Data
  Matrix<BaseFloat> tot_input(tot_input_rows, nnet_.InputDim("input"),
                              kSetZero);
  Matrix<BaseFloat> tot_ivector;
  if (ivector_dim > 0) {
    tot_ivector.Resize(current_queue->size(), ivector_dim, kSetZero);
  }
  BatchInfoQueue::iterator iter_prep = current_queue->begin();
  int32 input_count = 0;
  for (int32 n = 0; iter_prep != current_queue->end(); iter_prep++, n++) {
    std::tie(utt_id,
             first_input_frame, last_input_frame,
             first_subsampled_frame, last_subsampled_frame,
             output_offset) = *iter_prep;

    std::unordered_map<std::string, const Matrix<BaseFloat>* >::iterator
      feats_iter;
    feats_iter = feats_.find(utt_id);
    int32 num_input_frames = last_input_frame - first_input_frame + 1;
    if (first_input_frame >= 0 &&
      last_input_frame < (feats_iter->second)->NumRows()) {
      tot_input.RowRange(input_count, num_input_frames).CopyFromMat(
        (feats_iter->second)->RowRange(first_input_frame, num_input_frames));
    } else {
      int32 tot_input_feats = (feats_iter->second)->NumRows();
      for (int32 i = 0; i < num_input_frames; i++) {
        SubVector<BaseFloat> dest(tot_input, input_count + i);
        int32 t = i + first_input_frame;
        if (t < 0) t = 0;
        if (t >= tot_input_feats) t = tot_input_feats - 1;
        const SubVector<BaseFloat> src(*(feats_iter->second), t);
        dest.CopyFromVec(src);
      }
    }
    std::cout << "input matrix finished Ones" << std::endl;
    // Update ivector matrix
    // If the nnet_ doesn't have ivector, nothing will be returned by
    // GetCurrentIvector. So the ivector.Dim() == 0, and the tot_ivector will
    // not be filled.
    int32 first_output_frame =
      first_subsampled_frame * opts_.frame_subsampling_factor,
          last_output_frame =
      last_subsampled_frame * opts_.frame_subsampling_factor;
    Vector<BaseFloat> ivector;
    GetCurrentIvector(utt_id, first_output_frame,
                      last_output_frame - first_output_frame, &ivector);
    if (ivector.Dim() != 0) {
      tot_ivector.Row(n).CopyFromVec(ivector);
      std::cout << "ivector matrix finished Ones" << std::endl;
    }
    input_count += num_input_frames;
  }
  // Compute
  std::shared_ptr<const NnetComputation> computation =
    compiler_.Compile(request);
  Nnet *nnet_to_update = NULL;  // we're not doing any update
  NnetComputer computer(opts_.compute_config, *computation,
                        nnet_, nnet_to_update);
  CuMatrix<BaseFloat> input_feats_cu(tot_input);
  computer.AcceptInput("input", &input_feats_cu);
  CuMatrix<BaseFloat> ivector_feats_cu;
  // tot_ivector.NumCols() == 0 means that nnet_ doesn't have ivector
  if (tot_ivector.NumCols() != 0) {
    ivector_feats_cu.Resize(tot_ivector.NumRows(), tot_ivector.NumCols());
    ivector_feats_cu.CopyFromMat(tot_ivector);
    computer.AcceptInput("ivector", &ivector_feats_cu);
  }
  computer.Run();
  CuMatrix<BaseFloat> cu_output;
  computer.GetOutputDestructive("output", &cu_output);
  // Get Output
  if (log_priors_.Dim() != 0) {
    cu_output.AddVecToRows(-1.0, log_priors_);
  }
  cu_output.Scale(opts_.acoustic_scale);
  int32 output_count = 0;
  BatchInfoQueue::iterator iter_out = current_queue->begin();
  for (; iter_out != current_queue->end(); iter_out++) {
    std::tie(utt_id,
             first_input_frame, last_input_frame,
             first_subsampled_frame, last_subsampled_frame,
             output_offset) = *iter_out;
    int32 num_rows = last_subsampled_frame - first_subsampled_frame + 1;
    CuMatrix<BaseFloat> *output = new CuMatrix<BaseFloat>(num_rows,
                                                         output_dim_);
    output->CopyFromMat(cu_output.RowRange(output_count, num_rows));
    // Add the result
    (*result)[utt_id].push(output);
    (*utts_semaphores)[utt_id]->Signal();
    output_count += num_rows;
    // Set end symbol
    if ((last_subsampled_frame + 1) ==
         num_subsampled_frames_.find(utt_id)->second) {
      is_end->find(utt_id)->second = true;
      is_computed_.find(utt_id)->second = true;
      // This utterance has been finished. Clear it.
      Clear(utt_id);
    }
  }
  // Clear
  batch_info_[current_context]->clear();

  // comment out may help when you find the ComputationRequest in
  // contest_to_request_ is compiled more than one time. As when
  // CachingOptimizingCompiler compile a request, it will be moved to the end
  // of access_queue_.
  // ComputationRequestMap::iterator iter = context_to_request_.begin(), 
  //                                  end = context_to_request_.end();
  // for (; iter != end; iter++) {
  //   compiler_.Compile(*(iter->second));
  // }
}


void BatchComputer::DoNnetComputation(
  std::unordered_map<std::string,
                     std::queue<const CuMatrix<BaseFloat>*> > *result,
  std::unordered_map<std::string, bool> *is_end,
  std::unordered_map<std::string, Semaphore*> *utts_semaphores) {
  int32 ivector_dim = nnet_.InputDim("ivector");
  // According to the context_order_record_, we do the loop.
  ContextOrderRecord::iterator iter;
  for (iter = context_order_record_.begin();
       iter != context_order_record_.end(); iter++) {
    std::pair<int32, int32> current_context = *iter;
    if (batch_info_[current_context]->size() == 0) {
      break;
    }

    BatchInfoQueue* current_queue = batch_info_[current_context];
    std::string utt_id;
    int32 first_input_frame, last_input_frame;
    int32 first_subsampled_frame, last_subsampled_frame;
    int32 output_offset;

    int32 num_input_rows = current_context.first + opts_.frames_per_chunk +
                           current_context.second -
                           opts_.frame_subsampling_factor + 1;
    Matrix<BaseFloat> tot_input(num_input_rows * minibatch_size_,
                                nnet_.InputDim("input"), kSetZero);
    Matrix<BaseFloat> tot_ivector;
    if (ivector_dim > 0) {
      tot_ivector.Resize(minibatch_size_, ivector_dim, kSetZero);
    }
    // Preapre data
    BatchInfoQueue::iterator iter = current_queue->begin();
    for (int32 n = 0; iter != current_queue->end(); iter++, n++) {
      std::tie(utt_id,
               first_input_frame, last_input_frame,
               first_subsampled_frame, last_subsampled_frame,
               output_offset) = *iter;
      std::cout << utt_id << " " << first_input_frame << " "
                << last_input_frame << " " << first_subsampled_frame << " "
                << last_subsampled_frame << " " << output_offset << std::endl;

      std::unordered_map<std::string, const Matrix<BaseFloat>* >::iterator
        feats_iter;
      feats_iter = feats_.find(utt_id);
      int32 num_input_frames = last_input_frame - first_input_frame + 1;
      KALDI_ASSERT(num_input_frames == num_input_rows);
      if (first_input_frame >= 0 &&
          last_input_frame < (feats_iter->second)->NumRows()) {
        tot_input.RowRange(n * num_input_rows, num_input_frames).CopyFromMat(
          (feats_iter->second)->RowRange(first_input_frame, num_input_frames));
      } else {
        int32 tot_input_feats = (feats_iter->second)->NumRows();
        for (int32 i = 0; i < num_input_frames; i++) {
          SubVector<BaseFloat> dest(tot_input, n * num_input_frames + i);
          int32 t = i + first_input_frame;
          if (t < 0) t = 0;
          if (t >= tot_input_feats) t = tot_input_feats - 1;
          const SubVector<BaseFloat> src(*(feats_iter->second), t);
          dest.CopyFromVec(src);
        }
      }
      // Update ivector matrix
      // If the nnet_ doesn't have ivector, nothing will be returned by
      // GetCurrentIvector. So the ivector.Dim() == 0, and the tot_ivector will
      // not be filled.
      int32 first_output_frame =
        first_subsampled_frame * opts_.frame_subsampling_factor,
            last_output_frame =
        last_subsampled_frame * opts_.frame_subsampling_factor;
      Vector<BaseFloat> ivector;
      GetCurrentIvector(utt_id, first_output_frame,
                        last_output_frame - first_output_frame, &ivector);
      if (ivector.Dim() != 0) {
        tot_ivector.Row(n).CopyFromVec(ivector);
      }
    }
    // Compute
    std::shared_ptr<const NnetComputation> computation =
      compiler_.Compile(*(context_to_request_[current_context]));
    Nnet *nnet_to_update = NULL;  // we're not doing any update
    NnetComputer computer(opts_.compute_config, *computation,
                          nnet_, nnet_to_update);
    CuMatrix<BaseFloat> input_feats_cu(tot_input);
    std::cout << "Input matrix is " << input_feats_cu.NumRows() << "*" 
                                    << input_feats_cu.NumCols() << std::endl;
    std::cout << "Ivector matrix is " << tot_ivector.NumRows() << "*"
                                      << tot_ivector.NumCols() << std::endl; 
    computer.AcceptInput("input", &input_feats_cu);
    CuMatrix<BaseFloat> ivector_feats_cu;
    // tot_ivector.NumCols() == 0 means that nnet_ doesn't have ivector
    if (tot_ivector.NumCols() != 0) {
      ivector_feats_cu.Resize(tot_ivector.NumRows(), tot_ivector.NumCols());
      ivector_feats_cu.CopyFromMat(tot_ivector);
      computer.AcceptInput("ivector", &ivector_feats_cu);
    }
    std::cout << "before computer.run()" << std::endl;
    computer.Run();
    std::cout << "after computer.run()" << std::endl;
    CuMatrix<BaseFloat> cu_output;
    computer.GetOutputDestructive("output", &cu_output);
    // Get Output
    if (log_priors_.Dim() != 0) {
      cu_output.AddVecToRows(-1.0, log_priors_);
    }
    cu_output.Scale(opts_.acoustic_scale);

    int32 num_batch_output_rows = opts_.frames_per_chunk /
                                  opts_.frame_subsampling_factor;
    BatchInfoQueue::iterator iter_out = current_queue->begin();
    for (int32 n = 0; iter_out != current_queue->end(); iter_out++, n++) {
      std::tie(utt_id,
               first_input_frame, last_input_frame,
               first_subsampled_frame, last_subsampled_frame,
               output_offset) = *iter_out;
      int32 num_rows = last_subsampled_frame - first_subsampled_frame + 1;
      CuMatrix<BaseFloat> *output = new CuMatrix<BaseFloat>(num_rows,
                                                            output_dim_);
      output->CopyFromMat(cu_output.RowRange(
              n * num_batch_output_rows + output_offset, num_rows));
      (*utts_semaphores)[utt_id]->Signal();
      // Add the result
      (*result)[utt_id].push(output);
      if ((last_subsampled_frame + 1) ==
           num_subsampled_frames_.find(utt_id)->second) {
        is_computed_.find(utt_id)->second = true;
        is_end->find(utt_id)->second = true;
        // This utterance has been finished. Clear it.
        Clear(utt_id);
      }
    }
    // Clear
    batch_info_[current_context]->clear();
  }
}


void BatchComputer::GetCurrentIvector(std::string utt_id,
                                          int32 output_t_start,
                                          int32 num_output_frames,
                                          Vector<BaseFloat> *ivector) {
  if (ivectors_.find(utt_id) != ivectors_.end()) {
    *ivector = *(ivectors_.find(utt_id)->second);
    return;
  } else if (online_ivector_feats_.find(utt_id) ==
             online_ivector_feats_.end()) {
    return;
  }
  std::unordered_map<std::string, const Matrix<BaseFloat>* >::iterator iter;
  iter = online_ivector_feats_.find(utt_id);
  KALDI_ASSERT(online_ivector_period_ > 0);
  // frame_to_search is the frame that we want to get the most recent iVector
  // for.  We choose a point near the middle of the current window, the concept
  // being that this is the fairest comparison to nnet2.   Obviously we could do
  // better by always taking the last frame's iVector, but decoding with
  // 'online' ivectors is only really a mechanism to simulate online operation.
  int32 frame_to_search = output_t_start + num_output_frames / 2;
  int32 ivector_frame = frame_to_search / online_ivector_period_;
  KALDI_ASSERT(ivector_frame >= 0);
  if (ivector_frame >= (iter->second)->NumRows()) {
    int32 margin = ivector_frame - ((iter->second)->NumRows() - 1);
    if (margin * online_ivector_period_ > 50) {
      // Half a second seems like too long to be explainable as edge effects.
      KALDI_ERR << "Could not get iVector for frame " << frame_to_search
                << ", only available till frame "
                << (iter->second)->NumRows()
                << " * ivector-period=" << online_ivector_period_
                << " (mismatched --ivector-period?)";
    }
    ivector_frame = (iter->second)->NumRows() - 1;
  }
  *ivector = (iter->second)->Row(ivector_frame);
}


void BatchComputer::Clear(std::string utt_id) {
  delete feats_.find(utt_id)->second;
  feats_.erase(utt_id);
  if (ivectors_.find(utt_id) != ivectors_.end()) {
    delete ivectors_.find(utt_id)->second;
    ivectors_.erase(utt_id);
  }
  if (online_ivector_feats_.find(utt_id) != online_ivector_feats_.end()) {
    delete online_ivector_feats_.find(utt_id)->second;
    online_ivector_feats_.erase(utt_id);
  }
  num_subsampled_frames_.erase(utt_id);
  num_chunks_.erase(utt_id);
  utt_list_.remove(utt_id);
  is_computed_.erase(utt_id);
  prepared_chunk_record_.erase(utt_id);
  lasttime_finished_dec_utts_.erase(utt_id);
}


void BatchComputer::Compute(
  bool flush,
  std::unordered_map<std::string, 
                     std::queue<const CuMatrix<BaseFloat>*> > *result,
  std::unordered_map<std::string, bool> *is_end,
  std::unordered_map<std::string, Semaphore* > *utts_semaphores) {
  if (flush) {
    if (!Empty()) {
      if (ensure_exact_final_context_) {
        DoNnetComputationOnes(result, is_end, utts_semaphores);
      }
      DoNnetComputation(result, is_end, utts_semaphores);
    }
  } else {
    if (Ready()) {
      if (ensure_exact_final_context_) {
        DoNnetComputationOnes(result, is_end, utts_semaphores);
      }
      DoNnetComputation(result, is_end, utts_semaphores);
    }
  }
}


BatchComputer::~BatchComputer() {
  BatchInfoMap::iterator iter =
    batch_info_.begin(), end = batch_info_.end();
  for (; iter != end; iter++) {
    delete iter->second;
  }
  ComputationRequestMap::iterator iter2 =
    context_to_request_.begin(), end2 = context_to_request_.end();
  for (; iter2 != end2; iter2++) {
    delete iter2->second;
  }
}


bool BatchComputer::PrepareBatchInfo(
  const std::unordered_map<std::string, size_t> finished_dec_utts) {

  bool flush = false;

  std::vector<std::pair<std::string, int32> > used_chunks;
  std::unordered_map<std::string, int32> remaining_chunks;
  int32 tot_used_chunks = 0;
  int32 tot_remaining_chunks = 0;

  for (std::unordered_map<std::string, int32>::iterator it =
       num_subsampled_frames_.begin(); it != num_subsampled_frames_.end();
       it++) {
    std::string cur_utt_id = it->first;
    int32 cur_used_chunks = finished_dec_utts.at(cur_utt_id) -
                            lasttime_finished_dec_utts_.at(cur_utt_id);
    int32 cur_remaining_chunks = num_chunks_.at(cur_utt_id) -
                                 prepared_chunk_record_.at(cur_utt_id);
    tot_used_chunks += cur_used_chunks;
    tot_remaining_chunks += cur_remaining_chunks;

    used_chunks.push_back(
      std::make_pair(cur_utt_id, cur_used_chunks));
    remaining_chunks[cur_utt_id] = cur_remaining_chunks;
  }


  std::unordered_map<std::string, int32> this_turn_chunks;

  if (tot_remaining_chunks < GetBatchSize()) {
    flush = true;
    this_turn_chunks = remaining_chunks;
  } else { // we will fill the batch to the full.
    // used_chunks will be used to compute the scale, so increment zero.
    for (std::vector<std::pair<std::string, int32> >::iterator it =
         used_chunks.begin(); it != used_chunks.end(); it++) {
      if (it->second == 0) {
        it->second += 1;
        tot_used_chunks += 1;
      }
    }
    // sort from big to small
    std::sort(used_chunks.begin(), used_chunks.end(), CompareByValue());
  
    // if tot_used_chunks > minibatch_size, scale used_chunks to minibatch_size
    // Form here, used_chunks should be regared as a kind of weight
    if (tot_used_chunks > GetBatchSize()) {
      for (std::vector<std::pair<std::string, int32> >::iterator it =
           used_chunks.begin(); it != used_chunks.end(); it++) {
        it->second = it->second * 128 / tot_used_chunks;
      }
    }

    int32 counter = GetBatchSize();
    while (counter > 0) {
      for (std::vector<std::pair<std::string, int32> >::iterator it =
          used_chunks.begin(); it != used_chunks.end() && counter > 0; it++) {
        // Get the minimum value of used_chunks[utt_id], counter and
        // remaining_chunks[utt_id]
        int32 cur_num_chunks = it->second < counter ? it->second : counter;
        if (remaining_chunks[it->first] < cur_num_chunks) {
          cur_num_chunks = remaining_chunks[it->first];
        }
        // update
        remaining_chunks[it->first] -= cur_num_chunks;
        counter -= cur_num_chunks;
        if (this_turn_chunks.find(it->first) != this_turn_chunks.end()) {
          this_turn_chunks[it->first] += cur_num_chunks;
        } else {
          this_turn_chunks[it->first] = cur_num_chunks;
        }
      }
    }
  }
  
  // Update
  lasttime_finished_dec_utts_ = finished_dec_utts;

  for (std::unordered_map<std::string, int32>::iterator it =
       this_turn_chunks.begin(); it != this_turn_chunks.end(); it++) {
    std::string utt_id = it->first;
    int32 subsampling_factor = opts_.frame_subsampling_factor,
          subsampled_frames_per_chunk = opts_.frames_per_chunk /
                                        subsampling_factor,
          num_subsampled_frames = num_subsampled_frames_.find(utt_id)->second;
    KALDI_ASSERT(num_subsampled_frames > 0);

    for (int i = 0; i < it->second; i++) {
      int32 index = i + prepared_chunk_record_[utt_id];
      int32 extra_left_context = opts_.extra_left_context,
            extra_right_context = opts_.extra_right_context;
      // Prepare first_subsampled_frame, last_subsampled_frame. They are the
      // indexes of the output matrix.
      int32 first_subsampled_frame = index * subsampled_frames_per_chunk;
      int32 cur_num_subsampled_frames =
      std::min<int32>(subsampled_frames_per_chunk,
                      num_subsampled_frames - first_subsampled_frame);
      int32 last_subsampled_frame = first_subsampled_frame +
                                    cur_num_subsampled_frames - 1;

      int32 output_offset = subsampled_frames_per_chunk -
                            cur_num_subsampled_frames;
      // Prepare first_input_frame, last_input_frame
      int32 first_output_frame = first_subsampled_frame * subsampling_factor,
            last_output_frame = last_subsampled_frame * subsampling_factor;

      if ( first_output_frame == 0 && opts_.extra_left_context_initial >= 0 ) {
        extra_left_context = opts_.extra_left_context_initial;
      }
      if (last_subsampled_frame == num_subsampled_frames - 1 &&
          opts_.extra_right_context_final >= 0) {
        extra_right_context = opts_.extra_right_context_final;
      }
      // If ensure_exact_final_context_ is false, the "shorter than chunk size"
      // case will be arranged in "(extra_left_context_initial,
      // extra_right_context)" batch type.
      if (!ensure_exact_final_context_ &&
          first_output_frame == 0 &&
          last_subsampled_frame == num_subsampled_frames - 1 ) {
        extra_right_context = opts_.extra_right_context;
      }

      int32 left_context = nnet_left_context_ + extra_left_context;
      int32 right_context = nnet_right_context_ + extra_right_context;

      // first_input_frame can overlap with previous chunk
      int32 last_input_frame = last_output_frame + right_context;
      int32 first_input_frame = last_input_frame +
                                opts_.frame_subsampling_factor - right_context -
                                opts_.frames_per_chunk - left_context;

      // "shorter than chunk size" utterance case
      if (ensure_exact_final_context_ && first_output_frame == 0 &&
          last_subsampled_frame == num_subsampled_frames - 1 ) {
        first_input_frame = first_output_frame - left_context;
        left_context = -1;
        right_context = -1;
      }

      std::pair<int32, int32> context(left_context, right_context);

      // Update class private member
      BatchInfo batch_info = std::make_tuple(utt_id,
          first_input_frame, last_input_frame,
          first_subsampled_frame, last_subsampled_frame, output_offset);
      (batch_info_[context])->push_back(batch_info);
    }
    prepared_chunk_record_[utt_id] += it->second;
  }
  // Debug
  for (BatchInfoMap::iterator it = batch_info_.begin(); it != batch_info_.end();
       it++) {
    std::cout << "Context is (" << (it->first).first << ", "
              << (it->first).second << ")" << std::endl;
    for (BatchInfoQueue::iterator it_q = (*it->second).begin();
         it_q != (*it->second).end(); it_q++) {
      std::string utt_id;
      int32 input_begin, input_end, output_begin, output_end, offset;
      std::tie(utt_id, input_begin, input_end, output_begin,
               output_end, offset) = *it_q;
      std::cout << utt_id << " " << input_begin << " " << input_end << " "
                << output_begin << " " << output_end << " " << offset
                << std::endl;
    }
  }
  return flush;
}

BatchComputerClass::BatchComputerClass(
  BatchComputer* batch_computer,
  std::unordered_map<std::string,
    std::queue<const CuMatrix<BaseFloat>* > > *finished_inf_utts,
  const std::unordered_map<std::string, size_t> &finished_dec_utts,
  std::unordered_map<std::string, bool> *is_end,
  std::unordered_map<std::string, Semaphore* > *utts_semaphores,
  WaitingUtterancesRepository *repository,
  Semaphore *batch_compute_semaphore,
  std::mutex *utt_mutex) :
  batch_computer_(batch_computer), finished_inf_utts_(finished_inf_utts),
  finished_dec_utts_(finished_dec_utts), is_end_(is_end),
  utts_semaphores_(utts_semaphores), repository_(repository),
  batch_compute_semaphore_(batch_compute_semaphore),
  utt_mutex_(utt_mutex) {}

void BatchComputerClass::operator () () {
  while (true) {
    std::cout << thread_id_ << " batch 111" << std::endl;
    batch_compute_semaphore_->Wait();
    std::cout << thread_id_ << " batch 222" << std::endl;
    if (!batch_computer_->Done()) {
      std::cout << thread_id_ << " batch 333" << std::endl;
      bool flush = batch_computer_->PrepareBatchInfo(finished_dec_utts_);
      std::cout << thread_id_ << " batch 444" << std::endl;
      batch_computer_->Compute(flush, finished_inf_utts_, is_end_,
                               utts_semaphores_);
    } else {
      break;
    }
  }
  batch_computer_->Compute(true, finished_inf_utts_, is_end_, utts_semaphores_);
}

} // namespace nnet3
} // namespace kaldi
