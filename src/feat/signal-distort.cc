// featbin/signal-distort.cc

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



#include "feat/signal-distort.h"

namespace kaldi {

void ComputeAndApplyRandDistortion(const MatrixBase<BaseFloat> &input_egs,
                                   Matrix<BaseFloat> *perturb_egs) {
  // Generate impluse response |H(w)| using nonzero random sequence and smooth them 
  // using moving-average window with small window size.
  // For simplicity, assume zero-phase response and H(w) = |H(w)|.
  // num_fft_samp = 512
  int32 num_fft_samp = 512;
  Vector<BaseFloat> im_response(num_fft_samp);

}

void TimeStretch(const MatrixBase<BaseFloat> &input_egs, 
                 BaseFloat max_time_stretch,
                 Matrix<BaseFloat> *perturb_egs) {
  Matrix<BaseFloat> in_mat(input_egs), 
    out_mat(perturb_egs->NumRows(), perturb_egs->NumCols());
  int32 input_dim = input_egs.NumCols(), 
    dim = perturb_egs->NumCols();
  Vector<BaseFloat> samp_points_secs(dim);
  BaseFloat samp_freq = 2000, 
    max_stretch = max_time_stretch;
  // we stretch the middle part of the example and the input should be expanded
  // by extra frame to be larger than the output length => s * (m+n)/2 < m.
  // y((m - n + 2 * t)/2) = x(s * (m - n + 2 * t)/2) for t = 0,..,n 
  // where m = dim(x) and n = dim(y).
  KALDI_ASSERT(input_dim > dim * ((1.0 + max_stretch) / (1.0 - max_stretch)));
  // Generate random stretch value between -max_stretch, max_stretch.
  int32 max_stretch_int = static_cast<int32>(max_stretch * 1000);
  BaseFloat stretch = static_cast<BaseFloat>(RandInt(-max_stretch_int, max_stretch_int) / 1000.0); 
  if (abs(stretch) > 0) {
    int32 num_zeros = 4; // Number of zeros of the sinc function that the window extends out to.
    BaseFloat filter_cutoff_hz = samp_freq * 0.475; // lowpass frequency that's lower than 95% of 
                                                    // the Nyquist.
    for (int32 i = 0; i < dim; i++) 
      samp_points_secs(i) = static_cast<BaseFloat>(((1.0 + stretch) * 
        (0.5 * (input_dim - dim) + i))/ samp_freq);

    ArbitraryResample time_resample(input_dim, samp_freq,
                                    filter_cutoff_hz, 
                                    samp_points_secs,
                                    num_zeros);
    time_resample.Resample(in_mat, &out_mat);
  } else {
    int32 offset = static_cast<BaseFloat>(0.5 * (input_egs.NumCols() - perturb_egs->NumCols()));
    out_mat.CopyFromMat(input_egs.Range(0, input_egs.NumRows(), offset, perturb_egs->NumCols()));
  }
  perturb_egs->CopyFromMat(out_mat);
}

// This function add the noise to the orginial signal. We should not normalize 
// the signal level of the orginial signal. According to SNR, we rescale the noise
// and add it. So that the perturbed signal is created. 
void PerturbXvectorSignal::ApplyAdditiveNoise(const MatrixBase<BaseFloat> &input_eg,
                                              const Matrix<BaseFloat> &noise_eg,
                                              const int32 &SNR,
                                              Matrix<BaseFloat> *perturb_eg) {
  // In the version, we ask the noise_cols == input_cols.
  int32 input_rows = input_eg.NumRows(), input_cols = input_eg.NumCols();  
  KALDI_ASSERT(noise_eg.NumCols() == input_cols);

  // According to the rows of noise_eg, form the noise_mat
  // repeat the noise_eg blocks to have a new block which is longer than input_eg
 
  // As the noise_eg is very huge and the input_eg is small normally,
  // so we'd better not reload the "noise_eg" matrix
  // select the noise range

  Matrix<BaseFloat> selected_noise_mat;
  selected_noise_mat.Resize(input_rows, input_cols);
  
  int32 noise_rows = noise_eg.NumRows();
  int32 start_row_ind = RandInt(0, noise_rows - input_rows);
  
  if (noise_eg.NumRows() < input_rows) {
    int32 indices[input_rows];
    for (int32 i=0; i < input_rows; ++i) {
      indices[i] = (start_row_ind + i) % noise_eg.NumRows();
    }
    selected_noise_mat.CopyRows(noise_eg, indices);
  } else {
    selected_noise_mat.AddMat(1.0, noise_eg.Range(start_row_ind, input_rows,
                                                  0, input_cols));
  }

  // compute the energy of noise and input
  Matrix<BaseFloat> input_energy_mat(input_rows, input_cols);
  input_energy_mat.AddMatMatElements(1.0, input_eg, input_eg, 0.0);
  double input_energy = input_energy_mat.Sum();
  Matrix<BaseFloat> noise_energy_mat(input_rows, input_cols);
  noise_energy_mat.AddMatMatElements(1.0, selected_noise_mat, selected_noise_mat, 0.0);
  double noise_energy = noise_energy_mat.Sum();

  // In Energy domain, SNR=20log10(S/N). 
  // 10^(SNR/20) = input_energy / (scale^2 * noise_energy)
  double scale = input_energy / noise_energy / (pow(10,SNR/20));
  scale = sqrt(scale);
  
  // Add noise mat to input_eg mat
  perturb_eg->Resize(input_rows, input_cols);
  perturb_eg->CopyFromMat(input_eg);
  perturb_eg->AddMat(scale, selected_noise_mat);
}

void PerturbXvectorSignal::ApplyDistortion(const MatrixBase<BaseFloat> &input_egs,
                                           Matrix<BaseFloat> *perturb_egs) {
  if (!opts_.add_noise_rspecifier.empty()) { // deal with the add_noise ark situdation
    // count the number of noise examples and record the key
    std::vector<std::string> list_noise_egs;
    list_noise_egs.clear();
    kaldi::nnet3::SequentialNnetExampleReader noise_seq_reader(opts_.add_noise_rspecifier);
    for (; !noise_seq_reader.Done(); noise_seq_reader.Next()) {
      std::string key = noise_seq_reader.Key();
      list_noise_egs.push_back(key);
    }
    noise_seq_reader.Close();
    
    // random choose a noise_eg and use it.
    int32 num_noise_egs = list_noise_egs.size();
    int32 index_noise_egs = RandInt(0, num_noise_egs - 1);
    std::string key_noise_egs = list_noise_egs[index_noise_egs];

    kaldi::nnet3::RandomAccessNnetExampleReader noise_random_reader(opts_.add_noise_rspecifier);
    const kaldi::nnet3::NnetExample &noise_eg = noise_random_reader.Value(key_noise_egs);
    const kaldi::nnet3::NnetIo &noise_eg_io = noise_eg.io[0];
    Matrix<BaseFloat> noise_eg_mat;
    noise_eg_io.features.CopyToMat(&noise_eg_mat);
    int32 SNR = opts_.snr;

    // conduct ApplyAdditiveNoise
    ApplyAdditiveNoise(input_egs, noise_eg_mat, SNR, perturb_egs);

    // conduct others
    // TODO
  } else { // deal with the opts_.noise_egs situation
    // TODO
  }
}
// add-end
} // end of namespace kaldi
