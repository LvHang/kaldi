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
  bool volume_perturbation;
  bool speed_perturbation;
  bool time_shift;
  bool add_noise;

  XvectorPerturbOptions(): sample_frequency(8000),
                           expected_chunk_length(100),
                           max_speed_perturb_rate(0.1),
                           max_volume_variance(0.03),
                           max_snr(-5),
                           min_snr(20),
                           volume_perturbaton(true),
                           speed_perturbation(true),
                           time_shift(true),
                           add_noise(true) { }
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
    opts->Register("max-snr",&max_snr,"Specify a upperbound Signal to Noise Ratio. We will scale the noise according "
                   "to the original signal and SNR. Normally, it's a non-zero number between -30 and 30"
                   "default=-5.");
    opts->Register("min-snr",&min_snr,"Specify a lowerbound Signal to Noise Ratio. We will scale the noise according "
                   "to the original signal and SNR. Normally, it's a non-zero number between -30 and 30"
                   "default=10");
    opts->Register("volume-perturbation", &volume_perturbation, "If ture, we will "
                   "conduct variations in volume.");
    opts->Register("speed-perturbation", &speed_perturbation, "If ture, we will "
                   "conduct variations in speed.");
    opts->Register("time-shift", &time_shift, "If ture, we will "
                   "conduct time shift. That means randomly select the start point and "
                   "get the successive 'expected_chunk_length' data. Otherwise, we get "
                   "the data from the head.");
    opts->Register("add-noise", &add_noise, "If ture, we will "
                   "conduct add additive noise to source chunk.");
  }
};

/* This class is used to do 4 kinds of perturbation operation to fvector.
 * The input always is a Matrix which contains four lines(S1, S2, N1, N2)[S1=S2]
 * Then we will call different perturbation methods.
 * For the details about the four kinds of perturbation operation, please see
 * the document in fvector-perturb.cc.
 */
class FvectorPerturb {
 public:
  PerturbXvector(FvectorPerturbOptions opts) { opts_ = opts; }
  void ApplyPerturbation(const MatrixBase<BaseFloat>& input_chunk,
                         Matrix<BaseFloat>* perturbed_chunk);

  // Randomly Generate 4 scale number and scale each line respectively
  void VolumePerturbation(MatrixBase<BaseFloat>* chunk);

  // Use ArbitraryResample. For each line, randomly generate a speed factor. 
  // Then do time axis strench. As speed factor is different, so we deal with
  // each vector sepeartely. The dim of output_vector is bigger than
  // expected_chunk_length(ms)
  void SpeedPerturbation(MatrixBase<BaseFloat>& input_vector,
                         BaseFloat samp_freq,
                         BaseFloat speed_factor,
                         VectorBase<BaseFloat>* output_vector);

  // Randomly choose a expect_chunk_length(ms) vector.
  void TimeShift(VectorBase<BaseFloat>& input_vector,
                 VectorBase<BaseFloat>* output_vector);

  // The input is a matrix with four lines after three kinds of perturbation.
  // It is (S1, S2, N1, N2). Each line is expected_chunk_length(ms)(e.g. 800 dims)
  // add N1 to S1, add N2 to S2 with random snr.
  // After that, only the first two lines is meaningful, which represents two 
  // perturbed signals from the same source wavform signal.
  void AddNoise(MatrixBase<BaseFloat>* chunk);

 private:
  FvectorPerturbOptions opts_;
};

} // end of namespace kaldi
#endif // KALDI_FVECTOR_PERTURB_H_
