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
                                              const Matrix<BaseFloat> &noise_mat,
                                              Matrix<BaseFloat> *perturbed_eg) {
  // In the version, we ask the noise_cols == input_cols.
  int32 input_rows = input_eg.NumRows(), input_cols = input_eg.NumCols();  
  KALDI_ASSERT(noise_mat.NumCols() == input_cols);

  // As the noise_mat is very huge and the input_eg is small normally,
  // so we'd better not reload the "noise_mat" matrix
  // select the noise range

  Matrix<BaseFloat> selected_noise_mat;
  selected_noise_mat.Resize(input_rows, input_cols);
  
  int32 noise_rows = noise_mat.NumRows();
  int32 start_row_ind = RandInt(0, noise_rows - input_rows);
  
  if (noise_mat.NumRows() < input_rows) {
    int32 indices[input_rows];
    for (int32 i=0; i < input_rows; ++i) {
      indices[i] = (start_row_ind + i) % noise_mat.NumRows();
    }
    selected_noise_mat.CopyRows(noise_mat, indices);
  } else {
    selected_noise_mat.AddMat(1.0, noise_mat.Range(start_row_ind, input_rows,
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
  double scale = input_energy / noise_energy / (pow(10,opts_.snr/20));
  scale = sqrt(scale);
  
  // Add noise mat to input_eg mat
  perturbed_eg->Resize(input_rows, input_cols);
  perturbed_eg->CopyFromMat(input_eg);
  perturbed_eg->AddMat(scale, selected_noise_mat);
}

void PerturbXvectorSignal::ApplyDistortion(const MatrixBase<BaseFloat> &input_egs,
                                           Matrix<BaseFloat> *perturb_egs) {
  if (!opts_.add_noise.empty()) {
    // choose a noise from the noise.scp/ark
    // 1) we need to record the keys of noise_egs
    std::vector<std::string> noise_list;
    SequentialBaseFloatMatrixReader noise_seq_reader(opts_.add_noise);
    for (; !noise_seq_reader.Done(); noise_seq_reader.Next()) {
      std::string key = noise_seq_reader.Key();
      noise_list.push_back(key);
    }
    noise_seq_reader.Close();

    // 2) we random choose an noise example
    int32 num_noises = noise_list.size();
    int32 noise_index = RandInt(0, num_noises - 1);
    std::string noise_name = noise_list[noise_index];
    RandomAccessBaseFloatMatrixReader noise_random_reader(opts_.add_noise);
    Matrix<BaseFloat> noise_mat = noise_random_reader.Value(noise_name);

    // 3) conduct ApplyAdditiveNoise
    ApplyAdditiveNoise(input_egs, noise_mat, perturb_egs);
    // conduct others
    // TODO
  } 
}

// This function calls ApplyDistortion to apply different type of perturbations.
void PerturbExample(XvectorPerturbOptions opts,
                    const Matrix<BaseFloat> &input_egs,
                    Matrix<BaseFloat> *perturbed_egs) {
  // new a PerturbXvectorSignal object and call ApplyDistortion
  PerturbXvectorSignal perturb_egs(opts);
  perturb_egs.ApplyDistortion(input_egs, perturbed_egs);
}

} // end of namespace kaldi
