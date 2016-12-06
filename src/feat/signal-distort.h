// featbin/signal-distort.h

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

#ifndef KALDI_SIGNAL_DISTORT_H_
#define KALDI_SIGNAL_DISTORT_H_

#include <cassert>
#include <cstdlib>
#include <string>
#include <vector>

#include "base/kaldi-error.h"
#include "matrix/matrix-lib.h"
#include "util/common-utils.h"

#include "feat/resample.h"
#include "matrix/matrix-functions.h"
#include "cudamatrix/cu-matrix.h"

namespace kaldi {

// options class for distorting signals in egs
struct XvectorPerturbOptions {
  BaseFloat max_shift; 
  BaseFloat max_time_stretch;
  int32 frame_dim;
  int32 negation_prop; 
  bool rand_distort;
  std::string noise_egs;
  std::string add_noise;
  BaseFloat snr;

  XvectorPerturbOptions(): max_shift(0.2),
                           max_time_stretch(0.2),
                           frame_dim(80),
                           negation_prop(0.0),
                           rand_distort(false),
                           snr(10) { }
  void Register(OptionsItf *opts) { 
    opts->Register("max-shift", &max_shift, "Maximum random shift relative"
                "to frame length applied to egs.");
    opts->Register("max-speed-perturb", &max_time_stretch,
                   "Max speed perturbation applied on egs.");
    opts->Register("frame-dim", &frame_dim,
                   "The numebr of samples in input frame as product of frame_length by samp_freq.");
    opts->Register("negation-prop", &negation_prop, "This proportion of the input value is randomly negated.");
    opts->Register("noise-egs", &noise_egs, "If supplied, the additive noise is added to input signal.");
    opts->Register("rand_distort", &rand_distort, "If true, the signal is slightly changes"
                   "using some designed FIR filter with no zeros.");
    opts->Register("add-noise", &add_noise, "specify a file contains some noise egs");
    opts->Register("SNR",&snr,"specify a Signal to Noise Ration. We will scale the noise according"
                "to the original signal and SNR. Normally, it's a non-zero number between -30 and 30"
                "default=10");
  }
};

class PerturbXvectorSignal {
 public:
  PerturbXvectorSignal(XvectorPerturbOptions opts): opts_(opts) { };
  inline void SetNoiseEgs(const Matrix<BaseFloat> &noise_egs) {
    noise_egs_ = &noise_egs;
  }
  void ApplyDistortion(const MatrixBase<BaseFloat> &input_egs,
                       Matrix<BaseFloat> *perturb_egs);
 private:
  XvectorPerturbOptions opts_;
  // if we want use many examples in once ApplyDistortion, we can expand the point
  // to a point vector.
  const Matrix<BaseFloat> *noise_egs_;
  // I know we can use noise_egs_ instead of noise_eg parameter in this function,
  // But I keep it. Because we may expand the point to a point vector and choose
  // one kind noise to call ApplyAdditiveNoise.
  void ApplyAdditiveNoise(const MatrixBase<BaseFloat> &input_eg,
                          const Matrix<BaseFloat> &noise_eg,
                          Matrix<BaseFloat> *perturbed_eg);
};


// randomly disturb the input signal using a band-pass filter with no zeros.
void ComputeAndApplyRandDistortion(const MatrixBase<BaseFloat> &input_egs,
                                   Matrix<BaseFloat> *perturb_egs);

// Stretches the time axis for input egs without fixing the pitch value.
// It changes the speed and duration of the input signal without fixing pitch.
// The output  y w.r.t input x is going to be y(t - offset) = x(stretch * (t - offset)),
// where offset is the time index which the signal is stretches along that and the input
// and output are the same for t = offset.
// ArbitraryResample class is used to generate resampled output for different time-stretches.
// The output y is the stretched form of the input, x, and stretch value is randomely generated
// between [1 - max_stretch, 1 + max_stretch].
// y[(m - n + 2 * t)/2] = x[(1 + stretch) * (m - n + 2 * t)/2] for t = 0,..,n   
void TimeStretch(const MatrixBase<BaseFloat> &input_egs,
                 BaseFloat max_time_stretch,
                 Matrix<BaseFloat> *perturb_egs);

void PerturbExample(XvectorPerturbOptions opts,
                    const Matrix<BaseFloat> &input_egs,
                    Matrix<BaseFloat> *perturbed_egs);

} // end of namespace kaldi
#endif // KALDI_SIGNAL_DISTORT_H_
