// fvector/nnet3-fvector-perturb-signal.cc

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
#include "feat/wave-reader.h"
#include "feat/signal.h"

namespace kaldi {

struct AdditiveNoiseRange{
  BaseFloat wav_t_start;
  BaseFloat wav_t_end;
  std::string noise_uttid;
  BaseFloat noise_t_start;
  BaseFloat noise_t_end;
  BaseFloat snr;

  AdditiveNoiseRange(BaseFloat wav_t_start, BaseFloat wav_t_end, std::string noise_uttid,
                     BaseFloat noise_t_start, BaseFloat noise_t_end, BaseFloat snr):
    wav_t_start(wav_t_start), wav_t_end(wav_t_end), noise_uttid(noise_uttid),
    noise_t_start(noise_t_start), noise_t_end(noise_t_end), snr(snr) { }
};

void GenerateController(std::vector<std::string> &segments, 
                        std::vector<AdditiveNoiseRange> *controller) {
  BaseFloat wav_t_start;
  BaseFloat wav_t_end;
  std::string noise_uttid;
  BaseFloat noise_t_start;
  BaseFloat noise_t_end;
  BaseFloat snr;
  for(int i=0; i < segments.size(); ++i) {
    std::vector<std::string> split_string;
    SplitStringToVector(segments[i], ":", true, &split_string);
    KALDI_ASSERT(split_string.size() == 6);
    ConvertStringToReal(split_string[0], &wav_t_start);
    ConvertStringToReal(split_string[1], &wav_t_end);
    noise_uttid = split_string[2];
    ConvertStringToReal(split_string[3], &noise_t_start);
    ConvertStringToReal(split_string[4], &noise_t_end);
    ConvertStringToReal(split_string[5], &snr);
  
    controller->push_back(AdditiveNoiseRange(wav_t_start, wav_t_end, noise_uttid,
                                             noise_t_start, noise_t_end, snr));
  }
}

void ApplyNoise(std::string &noise_scp, const std::vector<AdditiveNoiseRange> &controller,
                const VectorBase<BaseFloat> &input_wav, const int &samp_freq_input,
                VectorBase<BaseFloat> *perturbed_wav) {
  // about noise list
  RandomAccessTableReader<WaveHolder> noise_reader(noise_scp);

  // add noise

  for (int i=0; i < controller.size(); ++i) {
    const WaveData &noise_wav = noise_reader.Value(controller[i].noise_uttid);
    BaseFloat samp_freq_noise = noise_wav.SampFreq();
    KALDI_ASSERT(samp_freq_input == samp_freq_noise);
      
    const Matrix<BaseFloat> &noise_matrix = noise_wav.Data();
    int32 num_samp_noise = noise_matrix.NumCols();
    Vector<BaseFloat> noise(num_samp_noise);
    noise.CopyRowFromMat(noise_matrix, 0);

    int32 input_start_point = samp_freq_input * controller[i].wav_t_start;
    int32 input_end_point = samp_freq_input * controller[i].wav_t_end - 1;
    int32 noise_start_point = samp_freq_noise * controller[i].noise_t_start;
    int32 noise_end_point = samp_freq_noise * controller[i].noise_t_end - 1;
    BaseFloat snr = controller[i].snr;
    // This part is used to deal with the precise problem.
    // e.g. If the wav_t_start = 259.49, the sample frequency is 8000. In theroy,
    // the wav_start_point is 2075920, however, it will be 2075919 in practise.
    int32 input_length = input_end_point - input_start_point + 1;
    int32 noise_length = noise_end_point - noise_start_point + 1;
    if (input_length != noise_length) {
      int32 delta = (input_length > noise_length?(input_length - noise_length)
                                                :(noise_length-input_length));
      if (delta < 0.01*samp_freq_input) {
        if (input_length > noise_length) {
          input_end_point = input_end_point - delta;
        } else {
          noise_end_point = noise_end_point - delta;
        }
      } else {
        KALDI_ERR << "There is a problem about input length does not match noise length"
                  << " where the noise-id is: " << controller[i].noise_uttid
                  << ", the input length is: " << input_length
                  << ", the noise length is: " << noise_length << std::endl; 
      }
    }

    // End sample must be less than total number
    if ((input_end_point > input_wav.Dim()-1) || (noise_end_point > noise.Dim()-1)) {
      int32 over_boundary = ((input_end_point - input_wav.Dim() + 1) > (noise_end_point - noise.Dim() + 1) ?
                             (input_end_point - input_wav.Dim() + 1) : (noise_end_point - noise.Dim() + 1));
      input_end_point = input_end_point - over_boundary;
      noise_end_point = noise_end_point - over_boundary;
    }
    // The input vector and noise vector contain the whole content of utt seperately.
    // According to the AdditiveNoiseRange, we stepwise add the additive noise to input.
    // To save the space, we use Subvector, because it returns the pointer.
    SubVector<BaseFloat> input_part(input_wav, input_start_point,
                                    input_end_point - input_start_point + 1);
    SubVector<BaseFloat> noise_part(noise, noise_start_point,
                                    noise_end_point - noise_start_point + 1);
    Vector<BaseFloat> selected_noise(input_part.Dim());

    // When encounter the situation where noise_part_length is shorter than input_part_length,
    // We pad recursively until the selected_noise_length equal to input_part_length.
    // Otherwise, selected_noise = noise_part
    if (noise_part.Dim() < input_part.Dim()) {
      int32 the_rest_length = selected_noise.Dim();
      while (the_rest_length > noise_part.Dim()) {
        selected_noise.Range(selected_noise.Dim()-the_rest_length,
                             noise_part.Dim()).CopyFromVec(noise_part);
        the_rest_length = the_rest_length - noise_part.Dim();
      }
      selected_noise.Range(selected_noise.Dim()-the_rest_length, the_rest_length).CopyFromVec(
          noise_part.Range(0, the_rest_length));
    } else {
      selected_noise.CopyFromVec(noise_part);
    }
      
    BaseFloat input_energy = VecVec(input_part, input_part);
    BaseFloat noise_energy = VecVec(selected_noise, selected_noise);
    BaseFloat scale_factor = sqrt(input_energy/ noise_energy/ (pow(10, snr/20)) );
    perturbed_wav->Range(input_start_point, input_part.Dim()).AddVec(scale_factor, selected_noise);
  }
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Perturb the wave files supplied via the specified noise-range\n"
        "Usage:  nnet3-fvector-perturb-signal [options...] <wav-in-rxfilename> "
        "<wav-out-wxfilename>\n"
        "e.g.\n"
        "nnet3-fvector-perturb-signal --noise=scp:noise.scp --noise-range="
        "wav1-perturbed-1 0.0:1.0:noise1:3.5:4.5:-8,... --input-channel=0 "
        "input.wav perturbed_input.wav\n";

    ParseOptions po(usage);
    
    std::string noise;
    std::string noise_range;
    int32 input_channel = 0;

    po.Register("noise",&noise,
                "There is a list of optional noise. It need to match the --noise-range.");
    po.Register("noise-range",&noise_range,
                "Provide a range file. We use the content in this file to control "
                "the process of adding noise. For each line, the format is <utt_id-perturb-i> "
                "<wav_t_start_1>:<wav_t_end_1>:<noise_utt_id_1>:<noise_t_start_1>:<noise_t_end_1>:<snr_1>,...,"
                "<wav_t_start_N>:<wav_t_end_N>:<noise_utt_id_N>:<noise_t_start_N>:<noise_t_end_N>:<snr_N>");
    po.Register("input-channel",&input_channel,
                "Specifies the channel to be used in input file");
    
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string input_wave_file = po.GetArg(1);
    std::string output_wave_file = po.GetArg(2);

    // Generate the Noise Controller list
    std::vector<AdditiveNoiseRange> controller;
    if (!noise_range.empty()) {
      //int index = noise_range.find_first_of(" ");
      //std::string perturbed_utt_id = noise_range.substr(0, index);
      //std::string noise_range_content = noise_range.substr(index+1);
      std::vector<std::string> segments;
      SplitStringToVector(noise_range, ",", true, &segments);
      GenerateController(segments, &controller);
    }

    bool binary = true;
    WaveData input_wave;
    {
      WaveHolder waveholder;
      Input ki(input_wave_file, &binary);
      waveholder.Read(ki.Stream());
      input_wave = waveholder.Value();
    }

    // about input wav
    const Matrix<BaseFloat> &input_matrix = input_wave.Data();
    BaseFloat samp_freq_input = input_wave.SampFreq();
    int32 num_samp_input = input_matrix.NumCols(),  // #samples in the input
          num_input_channel = input_matrix.NumRows();  // #channels in the input
    KALDI_VLOG(1) << "Sampling frequency of input: " << samp_freq_input
                  << "the number of samples: " << num_samp_input
                  << "the number of channels: " << num_input_channel;
    KALDI_ASSERT(input_channel < num_input_channel);
    Vector<BaseFloat> input(num_samp_input);
    input.CopyRowFromMat(input_matrix, input_channel);

    // new output vector and add noise
    Vector<BaseFloat> output(input);
    ApplyNoise(noise, controller, input, samp_freq_input, &output);

    Matrix<BaseFloat> out_matrix(1, num_samp_input);
    out_matrix.CopyRowsFromVec(output);

    WaveData out_wave(samp_freq_input, out_matrix);
    Output ko(output_wave_file, binary, false);
    WaveHolder::Write(ko.Stream(), true, out_wave);

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

