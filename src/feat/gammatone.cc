// feat/gammatone.cc

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


#include "feat/gammatone.h"

namespace kaldi {

void GenerateGammatoneFilters(GammatoneOptions &gamma_configs,
                              Matrix<BaseFloat> *gamma_filters) {
  int32 num_filters = gamma_configs.num_filters,
    filter_size = gamma_configs.filter_size;
  gamma_filters->Resize(num_filters, filter_size + 1);
  BaseFloat nyquist = gamma_configs.samp_freq / 2.0;
  for (int32 fnum = 0; fnum < num_filters; fnum++) {
    BaseFloat fnum_f = static_cast<BaseFloat>(fnum * 1.0),
      num_filters_f = static_cast<BaseFloat>(num_filters);
    BaseFloat f_bandwidth,
      f_center =  fnum_f / num_filters_f * nyquist;

    if (gamma_configs.bandwidth_type == 0)
      f_bandwidth = 24.7 * (0.00437 * f_center + 1); // Equivalent rectangular bandwidth
    else if (gamma_configs.bandwidth_type == 1)
      f_bandwidth = 21.4 * Log(0.00437 * f_center + 1); // Equivalent rectangular bandwidth
    else if (gamma_configs.bandwidth_type == 2)
      f_bandwidth = nyquist / num_filters_f;
    else
      KALDI_ERR << "Wrong bandwidth type " << gamma_configs.bandwidth_type;

    BaseFloat filter_norm = 0.0;
    for (int32 n = 0; n < filter_size; n++) {
      BaseFloat t = (n * 1.0) / gamma_configs.samp_freq;
      BaseFloat tmp = pow(t, gamma_configs.filter_order - 1) * 
        Exp(-1.0 * M_2PI * f_bandwidth * t) * 
        cos(M_2PI * f_center * t + gamma_configs.phase_carrier);
      filter_norm += (tmp * tmp);
      (*gamma_filters)(fnum, n) = tmp;
    }
    filter_norm = pow(filter_norm, 0.5);
    if (filter_norm != 0.0) 
      gamma_filters->Row(fnum).Scale(gamma_configs.amplitude / filter_norm);
    for (int32 n = 0; n < filter_size; n++)
      if (fabs((*gamma_filters)(fnum, n)) < 10e-6)
        (*gamma_filters)(fnum, n) = 0.0;
    KALDI_VLOG(2) << "filter " << fnum << ", center frequency " << f_center 
              << ", filter bandwidth " << f_bandwidth 
              << ", filter l2_norm " << filter_norm;
  }
}

 
} // end of namespace kaldi
