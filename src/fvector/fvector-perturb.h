#ifndef KALDI_FVECTOR_PERTURB_H_
#define KALDI_FVECTOR_PERTURB_H_

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
struct FvectorPerturbOptions {
  int32 sample_frequency;
  BaseFloat expected_chunk_length;
  BaseFloat max_speed_perturb_rate;
  BaseFloat max_volume_variance;
  BaseFloat max_snr;
  BaseFloat min_snr;
  bool do_volume_variation;
  bool do_speed_perturbation;
  bool do_time_shift;
  bool do_add_noise;
  bool do_reverberation;

  XvectorPerturbOptions(): sample_frequency(8000),
                           expected_chunk_length(100),
                           max_speed_perturb_rate(0.1),
                           max_volume_variance(0.03),
                           max_snr(-5),
                           min_snr(20),
                           do_volume_variation(true),
                           do_speed_perturbation(true),
                           do_time_shift(true),
                           do_add_noise(true),
                           do_reverberation(false) { }
  void Register(OptionsItf *opts) {
    opts->Register("sample-frequency", &sample_frequency, "The sample frequency "
                   "of the wav signal.");
    opts->Register("expected-chunk-length", &expected_chunk_length, "It show the "
                   "length of chunk you expected. e.g. 100ms. That means the length "
                   "of output will correspond to 100ms. At the same time, it will "
                   "affect the speed_perturb_rate, the speed_perturb_rate factor will "
                   "in the range of min{expected-chunk-length/original-length, "
                   "max-speed-perturb-rate}. default=100 ms.");
    opts->Register("max-speed-perturb-rate", &max_speed_perturb_rate,
                   "Max speed perturbation applied on matrix. It will work together "
                   "with expected_chunk_length, default=0.1.");
    opts->Register("max-volume-variance", &max_volume_variance, "The variation in "
                   "volume will vary form -max-volume-variance to max-volume-variance randomly."
                   "default=0.03.")
    opts->Register("max-snr",&max_snr,"Specify a upperbound Signal to Noise Ration. We will scale the noise according "
                   "to the original signal and SNR. Normally, it's a non-zero number between -30 and 30"
                   "default=-5.");
    opts->Register("min-snr",&min_snr,"Specify a lowerbound Signal to Noise Ration. We will scale the noise according "
                   "to the original signal and SNR. Normally, it's a non-zero number between -30 and 30"
                   "default=10");
    opts->Register("do-volume-variation", &do_volume_variation, "If ture, we will "
                   "conduct variations in volume.");
    opts->Register("do-speed-perturbation", &do_speed_perturbation, "If ture, we will "
                   "conduct variations in speed.");
    opts->Register("do-time-shift", &do_time_shift, "If ture, we will "
                   "conduct time shift. That means randomly select the start point and "
                   "get the successive 'expected_chunk_length' data. Otherwise, we get "
                   "the data from the head.");
    opts->Register("do-add-noise", &do_add_noise, "If ture, we will "
                   "conduct add additive noise to source chunk.");
    opts->Register("do-reverberation", &do_reverberation, "If ture, we will "
                   "conduct reverberation operation.");
  }
};

class FvectorPerturb {
 public:
  PerturbXvector(FvectorPerturbOptions opts) { opts_ = opts; }
  void ApplyPerturbation(const MatrixBase<BaseFloat> &input_chunk,
                         Matrix<BaseFloat> *perturbed_chunk);
  void DoVolumeVariation(MatrixBase<BaseFloat> &chunk);
  void DoSpeedPerturbation(MatrixBase<BaseFloat> &chunk);
  void DoTimeShift(MatrixBase<BaseFloat> &chunk);
  void DoAddNoise(MatrixBase<BaseFloat> &chunk);

 private:
  FvectorPerturbOptions opts_;
};

} // end of namespace kaldi
#endif // KALDI_FVECTOR_PERTURB_H_
