#include "fvector/fvector-perturb.h"

namespace kaldi {
namespace nnet3 {

void FvectorPerturb::ApplyPerturbation(const MatrixBase<BaseFloat>& input_chunk,
                                       Matrix<BaseFloat>* perturbed_chunk) {
  Matrix<BaseFloat> tmp_matrix(input_chunk);
  if (opts_.do_volume_variation) {
    DoVolumeVariation(tmp_matrix);
  }
  if (opts_.do_speed_perturbation) {
    DoSpeedPerturbation(tmp_matrix);
  }
  if (opts_.do_time_shift) {
    DoTimeshift(tmp_matrix);
  }
  if (opts_.do_add_noise) {
    DoAddNoise(tmp_matrix);
  }
  perturb_chunk->Swap(&tmp_matrix);
}

void FvectorPerturb::DoVolumeVariation(MatrixBase<BaseFloat>& chunk) {
  //1. Randomly generate 4 number from (1-max-volume-variance, 1+max-volume-variance)
  //2. scale each line respectively.
}

void FvectorPerturb::DoSpeedPerturbation(MatrixBase<BaseFloat>& chunk) {
  //1. generate 4 speed perturb factor randomly
  //(1) a=min{expected_chunk_length/original_length, max-speed-perturb-rate}
  //(2) the range of factor is (1-a, 1+a)
  //2. Use ArbitraryResample deal with each line
  // Reference Pegah's Time-stretch
}

void FvectorPerturb::DoTimeShift(MatrixBase<BaseFloat>& chunk) {
  //1. generate 4 start point randomly whose range is
  // [0, row.NumCols()- expected_chunk_length * sample_frequency)
  //2. get the successive expected_chunk_length * sample_frequency data.
}

void FvectorPerturb::DoAddNoise(MatrixBase<BaseFloat>& chunk) {
  //Now, the dim of each line is expected_chunk_length * sample_frequency
  //e.g 100ms * 8000Hz = 800.
  //1. generate 2 SNR from (min-snr, max-snr)
  //2. add N1(line3) to S1(line1) with snr1
  //   add N2(line4) to S2(line2) with snr2
  //Reference my perivous code for fvector or wav-reverberation.cc 
}


} // end of namespace nnet3
} // end of namespace kaldi
