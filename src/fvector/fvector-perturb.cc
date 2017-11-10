#include "fvector/fvector-perturb.h"

namespace kaldi {

void FvectorPerturb::ApplyPerturbation(const MatrixBase<BaseFloat>& input_chunk,
                                       Matrix<BaseFloat>* perturbed_chunk) {
  // The original_dim_matrix is a matrix whose dimension is same with input_chunk.
  // Assume the sample_frequency=8kHz, the original length is 120ms.
  // It will be a (4, 960) matrix.
  Matrix<BaseFloat> original_dim_matrix(input_chunk);
  if (opts_.volume_perturbation) {
    VolumePerturbation(&original_dim_matrix);
  }
  // The expected_dim_matrix is a matrix (input_chunk.NumRows(), expected-chunk-length
  // * sample_frequency / 1000). E.g. it is a (4, 800) matrix.
  Matrix<BaseFloat> expected_dim_matrix(input_chunk.NumRows(),
      opts_.expected_chunk_length * opts_.sample_frequency / 1000);
  if (opts_.speed_perturbation) {
    //1. generate speed perturb factor randomly(Noice: the expected_length is
    //always smaller than original_length) for each line.
    //(1) a=min{original_length/expected_length -1, max-speed-perturb-rate}
    //(2) the range of factor is (1-a, 1+a)
    BaseFloat boundary = std::min((input_chunk.NumCols() / opts_.sample_frequency) / opts_.expected_chunk_length - 1,
                                  opts_.max_speed_perturb_rate);
    for (MatrixIndexT i = 0; i < original_dim_matrix.NumRows(); ++i) {
      //caculate the speed factor
      BaseFloat factor =static_cast<BaseFloat> (RandInt(
          (int)((1-boundary)*100),(int)((1+boundary)*100)) * 1.0 / 100.0);
      
      Vector<BaseFloat> speed_input_vector(original_dim_matrix.Row(i));
      
      MatrixIndexT speed_output_dim = static_cast<MatrixIndexT>(original_dim_matrix.NumCols() / factor);
      KALDI_ASSERT(speed_output_dim >= opts_.expected_chunk_length * opts_.sample_frequency / 1000);
      Vector<BaseFloat> speed_output_vector(speed_output_dim);

      SpeedPerturbation(speed_input_vector, opts_.sample_frequency, factor, &speed_output_vector);
      
      Vector<BaseFloat> time_shifted_vector(expected_dim_matrix.NumCols());
      if (opts_.time_shift) {
        TimeShift(speed_output_vector, &time_shifted_vector);
      } else {
        time_shifted_vector.CopyFromVec(speed_output_vector.Range(0, expected_dim_matrix.NumCols()));
      }
      expected_dim_matrix.CopyRowFromVec(time_shifted_vector, i); 
    }
  } else { //no speed_perturbation
    if (opts_.time_shift) {
      for (MatrixIndexT i = 0; i < original_dim_matrix.NumRows(); ++i) {
        Vector<BaseFloat> input_vector(original_dim_matrix.Row(i));
        Vector<BaseFloat> time_shifted_vector(expected_dim_matrix.NumCols());
        TimeShift(input_vector, &time_shifted_vector);
        expected_dim_matrix.CopyRowFromVec(time_shifted_vector, i);
      }  
    } else {
      expected_dim_matrix.CopyFromMat(original_dim_matrix.Range(0, expected_dim_matrix.NumRows(),
                                                                0, expected_dim_matrix.NumCols()));
    }
  }
  // Now we operate the "expected_dim_matrix"
  if (opts_.add_noise) {
    AddNoise(&expected_dim_matrix);
  }
  perturbed_chunk->Resize(2, expected_dim_matrix.NumCols());
  MatrixIndexT indices[2] = {0, 1};
  perturbed_chunk->CopyRows(expected_dim_matrix, indices);
}

void FvectorPerturb::VolumePerturbation(MatrixBase<BaseFloat>* chunk) {
  //1. Randomly generate 4 number from (1-max-volume-variance, 1+max-volume-variance)
  std::vector<BaseFloat> volume_factors;
  for (MatrixIndexT i = 0; i < chunk->NumRows(); ++i) {
    BaseFloat factor = static_cast<BaseFloat>(
        RandInt((int)((1-opts_.max_volume_variance)*100),
                (int)((1+opts_.max_volume_variance)*100)) / 100.0);
    volume_factors.push_back(factor);
  }
  //2. scale each line respectively.
  for (MatrixIndexT i = 0; i < chunk->NumRows(); ++i) {
    chunk->Row(i).Scale(volume_factors[i]);
  }
}

// we stretch the signal from the beginning to end.
// y(t) = x(s*t) for t = 0,...,n. If s>0, the output will be shorter than
// input. It represents speeding up. Vice versa.
// Use ArbitraryResample deal with each line.
//
// In ArbitraryResample, according to num_zeros and filter_cutoff, it generates
// the "filter_with". And then each output_sample(t) corresponds to few input_samples
// from (t-filter_with) to (t+filter_with), which is stored in "first_index_".
// And "weights_" will be adjust by a Hanning window in function FilterFunc.
// In brief, you can think each output sample is the weighted sum of few input_samples.
void FvectorPerturb::SpeedPerturbation(VectorBase<BaseFloat>& input_vector,
                                       BaseFloat samp_freq,
                                       BaseFloat speed_factor,
                                       VectorBase<BaseFloat>* output_vector) {
  if (speed_factor == 1.0) {
    output_vector->CopyFromVec(input_vector);
  } else {
    Vector<BaseFloat> in_vec(input_vector),
                      out_vec(output_vector->Dim());
    int32 input_dim = in_vec.Dim(),
          output_dim = out_vec.Dim();
    Vector<BaseFloat> samp_points_secs(output_dim);
    int32 num_zeros = 4; // Number of zeros of the sinc function that the window extends out to.
    // lowpass frequency that's lower than 95% of the Nyquist.
    BaseFloat filter_cutoff_hz = samp_freq * 0.475; 
    for (int32 i = 0; i < output_dim; i++) {
      samp_points_secs(i) = static_cast<BaseFloat>(speed_factor * i / samp_freq);
    }
    ArbitraryResample time_resample(input_dim, samp_freq,
                                    filter_cutoff_hz, 
                                    samp_points_secs,
                                    num_zeros);
    time_resample.Resample(in_vec, &out_vec);
    output_vector->CopyFromVec(out_vec);
  }
}

void FvectorPerturb::TimeShift(VectorBase<BaseFloat>& input_vector,
                               VectorBase<BaseFloat>* output_vector) {
  //1. generate start point randomly whose range is
  // [0, row.NumCols()- expected_chunk_length * sample_frequency)
  int32 start_point = static_cast<int32>(RandInt(0, input_vector.Dim() - output_vector->Dim()));
  //2. get the successive expected_chunk_length * sample_frequency data.
  output_vector->CopyFromVec(input_vector.Range(start_point, output_vector->Dim()));
}

void FvectorPerturb::AddNoise(MatrixBase<BaseFloat>* chunk) {
  //Now, the dim of each line is expected_chunk_length * sample_frequency
  //e.g 100ms * 8000Hz = 800.
  //1. generate 2 SNR from (min-snr, max-snr)
  //2. add N1(line3) to S1(line1) with snr1
  //   add N2(line4) to S2(line2) with snr2
  for (MatrixIndexT i = 0; i < 2; i++) {
    Vector<BaseFloat> source(chunk->Row(i));
    Vector<BaseFloat> noise(chunk->Row(i+2));
    BaseFloat source_energy = VecVec(source, source);
    BaseFloat noise_energy = VecVec(noise, noise);
    // The smaller the value, the greater the snr
    int32 snr = RandInt(opts_.max_snr, opts_.min_snr);
    BaseFloat scale_factor = sqrt(source_energy/ noise_energy / (pow(10, snr/20)));
    chunk->Row(i).AddVec(scale_factor, noise);
  }
}


} // end of namespace kaldi
