// feat/gammatone.h

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

#ifndef KALDI_FEAT_GAMMATONE_H_
#define KALDI_FEAT_GAMMATONE_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <utility>
#include <vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"


namespace kaldi {
struct GammatoneOptions {
  BaseFloat amplitude; // The filter amplitude
  BaseFloat phase_carrier;  // The phase carrier (in radians)
  BaseFloat samp_freq; 
  int32 filter_order; // The filter's order
  int32 num_filters;  // number of gammatone filters
  int32 filter_size;  // Size of gammatone filters
  int32 bandwidth_type; // the method used to compute filter bandwidth
  GammatoneOptions():
    amplitude(10e8),
    phase_carrier(0.5236),
    samp_freq(8000),
    filter_order(4),
    num_filters(100),
    filter_size(512),
    bandwidth_type(0) { }

  void Register(OptionsItf *opts) {
    opts->Register("g-amplitude", &amplitude,
                   "The amplitude used for gammatone filters.");
    opts->Register("g-phase-carrier", &phase_carrier,
                   "The phase carrier(in radian) for gammatone filters.");
    opts->Register("samp-freq", &samp_freq,
                   "The sampling frequency to compute gammatone filters.");
    opts->Register("g-filter-order", &filter_order,
                   "The filter oder used to compute gammatone filters.");
    opts->Register("num-filters", &num_filters,
                   "Number of gammatone filters.");
    opts->Register("filter-size", &filter_size,
                   "The size of gammatone filters.");
    opts->Register("bandwidth-type", &bandwidth_type,
                   "The filter bandwidth type. 0 is ERB;"
                   "1 is ERBS and 2 is uniform bandwidth.");
  }

};

// Generate gammatone filters.
// Gammatone filters are linear filters described by an impluse response
// that is product of gamma distribution and sinusoidal tone.
// It is widely used model of auditory filters in the auditory system.
// The impulse response is  g(t) = a t^(n-1) * exp(-2pi * b * t) * cos(2pi * f * t + phi)
// where f is center qrequency and b is the filter's bandwidth in Hz.
// We use approximate equivalent rectangular bandwidth formula to initialize filter's bandwith.
// as ERB(f) = 24.7 * (4.37 * f + 1) , where f is center frequency.
void GenerateGammatoneFilters(GammatoneOptions &gamma_configs,
                              Matrix<BaseFloat> *filters);

} // end of namespace kaldi
#endif // KALDI_FEAT_GAMMATONE_H_
