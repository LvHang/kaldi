// xvectorbin/nnet3-xvector-signal-perturb-egs.cc

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/signal-distort.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-example-utils.h" 
namespace kaldi {
namespace nnet3 {

// This function applies different type of perturbation to input_egs.
// random distortion of inputs, random shifts, adding additive noise,
// random time stretch and random negations are different type of 
// distortions used in this function.
void ApplyPerturbation(XvectorPerturbOptions opts,
                       const Matrix<BaseFloat> &input_egs,
                       Matrix<BaseFloat> *noise_egs,
                       Matrix<BaseFloat> *perturb_egs) {

  PerturbXvectorSignal perturb_xvector(opts);
  
  Matrix<BaseFloat> shifted_egs(input_egs);
  // Generate random shift samples to shift egs. 
  if (opts.max_shift != 0.0) {
    int32 max_shift_int = static_cast<int32>(opts.max_shift * opts.frame_dim);
    // shift input_egs using random shift. 
    int32 eg_dim = input_egs.NumCols() - opts.frame_dim,
      shift = RandInt(0, max_shift_int);
    shifted_egs.CopyFromMat(input_egs.Range(0, input_egs.NumRows(), shift, eg_dim));
  }
  
  Matrix<BaseFloat> rand_distort_shifted_egs(shifted_egs);
  if (opts.rand_distort) {
    // randomly generate an zero-phase FIR filter with no zeros.
    // In future, we can select trucated part of room impluse response
    // and convolve it with input_egs.
    ////perturb_xvector.ComputeAndApplyRandDistortion(shifted_egs,
    ////                              &rand_distort_shifted_egs);
  }

  if (noise_egs) { 
    // select random block of noise egs and add to input_egs
    // number of additive noises should be larger than number of input-egs.
    KALDI_ASSERT(noise_egs->NumRows() >= input_egs.NumRows());
    if (noise_egs->NumRows() < input_egs.NumRows()) {
      // repeat the noise_egs_mat blocks to have same length block
      // and randomly perturb the rows.
    } else {
      // Select random submatrix out of noise_egs and add it to perturb_egs.
      // we should shuffle noise_egs before passing them to this binary.
      int32 start_row_ind = RandInt(0, noise_egs->NumRows() - input_egs.NumRows()),
        start_col_ind = RandInt(0, noise_egs->NumCols() - input_egs.NumCols()); 
      rand_distort_shifted_egs.AddMat(1.0, noise_egs->Range(start_row_ind, input_egs.NumRows(),
                                      start_col_ind, input_egs.NumCols()));
    }
  }
  // Perturb speed of signal egs
  Matrix<BaseFloat> warped_distorted_shifted_egs(rand_distort_shifted_egs);
  ////if (opts.max_time_stretch != 0.0) 
  ////  perturb_xvector.TimeStretch(rand_distort_shifted_egs, 
  ////                              &warped_distorted_shifted_egs);
   
  // If nagation is true, the sample values are randomly negated
  // with some probability.
  ////if (opts.negation) {
   
  ////}
}

// add
// This function add the noise to the orginial signal. We should not normalize 
// the signal level of the orginial signal. According to SNR, we rescale the noise
// and add it. So that the perturbed signal is created. 
void ApplyAddAdditiveNoise(const int32 &SNR,
                           const Matrix<BaseFloat> &input_eg,
                           const Matrix<BaseFloat> &noise_eg,
                           Matrix<BaseFloat> *perturb_eg) {
  // In the version, we ask the noise_cols == input_cols.
  int32 input_rows = input_eg.NumRows(), input_cols = input_eg.NumCols();  
  KALDI_ASSERT(noise_eg.NumCols() == input_cols);

  // According to the rows of noise_eg, form the noise_mat
  // repeat the noise_eg blocks to have a new block which is longer than input_eg
  Matrix<BaseFloat> noise_mat;
  if (noise_eg.NumRows() < input_rows) {
    int32 repeat_times = (input_rows / noise_eg.NumRows()) + 1;
    noise_mat.Resize(noise_eg.NumRows() * repeat_times, noise_eg.NumCols());
    for (int32 i = 0; i < repeat_times; ++i) {
      noise_mat.Range(i*noise_eg.NumRows(), noise_eg.NumRows(), 
                      0, noise_eg.NumCols()).CopyFromMat(noise_eg);
    }
  } else {
    noise_mat.Resize(noise_eg.NumRows(), noise_eg.NumCols());
    noise_mat.CopyFromMat(noise_eg);
  }

  // select the noise range
  int32 noise_rows = noise_mat.NumRows();
  int32 start_row_ind = RandInt(0, noise_rows - input_rows);
  Matrix<BaseFloat> selected_noise_mat(input_rows, input_cols);
  selected_noise_mat.AddMat(1.0, noise_mat.Range(start_row_ind, input_rows,
                                                 0, input_cols));
  // compute the energy of noise and input
  Matrix<BaseFloat> input_energy_mat(input_rows, input_cols);
  input_energy_mat.AddMatMatElements(1.0, input_eg, input_eg, 1.0);
  double input_energy = input_energy_mat.Sum();
  Matrix<BaseFloat> noise_energy_mat(input_rows, input_cols);
  noise_energy_mat.AddMatMatElements(1.0, selected_noise_mat, selected_noise_mat, 1.0);
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
// add-end

} // end of namespace nnet3
} // end of namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::nnet3;
    typedef kaldi::int32 int32;   
    typedef kaldi::int64 int64;


    const char *usage =
        "Corrupts  the examples supplied via input pipe with different type of distortions\n"
        "such as additive noise, negation, random time shifts or random distortion.\n"
        "Usage: nnet3-xvector-signal-perturb-egs [options...] <egs-especifier> <egs-wspecifier>\n"
        "e.g.\n"
        "nnet3-xvector-signal-perturb-egs --noise-egs=noise.egs\n"
        "--max-shift=0.2 --max-speed-perturb=0.1 --negation=true\n"
        "ark:input.egs akr:distorted.egs\n";
    ParseOptions po(usage);

    XvectorPerturbOptions perturb_opts;
    perturb_opts.Register(&po);

    // add
    std::string add_noise_rspecifier;
    po.Register("add-noise", &add_noise_rspecifier, "specify a file contains some noise egs");
    int32 snr;
    po.Register("SNR",&snr,"specify a Signal to Noise Ration.We will scale the noise according \
                to the original signal and SNR. Normally, it's a non-zero number between -30 and 30");
    // add-end

    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    
    std::string examples_rspecifier = po.GetArg(1),
      examples_wspecifier = po.GetArg(2);

    SequentialNnetExampleReader example_reader(examples_rspecifier);
     
    NnetExampleWriter example_writer(examples_wspecifier);

    // add
    // count the number of noise examples and record the key
    std::vector<std::string> list_noise_egs;
    SequentialNnetExampleReader noise_seq_reader(add_noise_rspecifier);
    for (; !noise_seq_reader.Done(); noise_seq_reader.Next()) {
      std::string key = noise_seq_reader.Key();
      list_noise_egs.push_back(key);
    }
    noise_seq_reader.Close();
    int32 num_noise_egs = list_noise_egs.size();
    // initial a RandomAccessTableReader for noise egs
    RandomAccessNnetExampleReader noise_random_reader(add_noise_rspecifier);
    // add-end

    int64 num_read = 0, num_written = 0;

    Matrix<BaseFloat> *noise_mat = NULL;
    // read additive noise egs if it is specified.
    if (!perturb_opts.noise_egs.empty()) {
      SequentialNnetExampleReader noise_reader(perturb_opts.noise_egs);
      const NnetExample &noise_egs = noise_reader.Value();
      const NnetIo &noise_io = noise_egs.io[0];
      noise_io.features.CopyToMat(noise_mat);
       
    }

    for (; !example_reader.Done(); example_reader.Next(), num_read++) {
      std::string key = example_reader.Key();
      const NnetExample &input_eg = example_reader.Value();
      const NnetIo &input_eg_io = input_eg.io[0];
      NnetExample *perturb_eg = new NnetExample();
      Matrix<BaseFloat> perturb_eg_mat, 
        input_eg_mat;
      input_eg_io.features.CopyToMat(&input_eg_mat);
      
      // add
      if (!add_noise_rspecifier.empty()) {
        // random choose a noise example
        int32 index_noise_egs = RandInt(0, num_noise_egs - 1);
        std::string key_noise_egs = list_noise_egs[index_noise_egs];
        const NnetExample &noise_eg = noise_random_reader.Value(key_noise_egs);
        const NnetIo &noise_eg_io = noise_eg.io[0];
        
        Matrix<BaseFloat> noise_eg_mat;
        noise_eg_io.features.CopyToMat(&noise_eg_mat);

        // deal with add noise
        ApplyAddAdditiveNoise(snr, input_eg_mat, noise_eg_mat, &perturb_eg_mat);
      } else {
        ApplyPerturbation(perturb_opts, input_eg_mat, noise_mat, &perturb_eg_mat);
      }
      // add-end
      perturb_eg->io.resize(1.0);
      perturb_eg->io[0].features.SwapFullMatrix(&perturb_eg_mat);
      example_writer.Write(key, *perturb_eg);
      num_written++;
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
