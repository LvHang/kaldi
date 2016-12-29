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

struct NoiseController{
  BaseFloat wav_t_start_;
  BaseFloat wav_t_end_;
  std::string noise_uttid_;
  BaseFloat noise_t_start_;
  BaseFloat noise_t_end_;
  BaseFloat snr_;

  NoiseController(BaseFloat wav_t_start, BaseFloat wav_t_end, std::string noise_uttid,
                  BaseFloat noise_t_start, BaseFloat noise_t_end, BaseFloat snr):
    wav_t_start_(wav_t_start), wav_t_end_(wav_t_end), noise_uttid_(noise_uttid),
    noise_t_start_(noise_t_start), noise_t_end_(noise_t_end), snr_(snr) { }
};

void GenerateController(std::vector<std::string> &segments, 
                        std::vector<NoiseController> *controller) {
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
  
    controller->push_back(NoiseController(wav_t_start, wav_t_end, noise_uttid,
                                            noise_t_start, noise_t_end, snr));
  }
}

void ApplyNoise(std::string &noise_scp, const std::vector<NoiseController> &controller,
                const VectorBase<BaseFloat> &input_wav, VectorBase<BaseFloat> *perturbed_wav) {
  // about noise list
  RandomAccessTableReader<WaveHolder> noise_reader(noise_scp);
  int samp_freq_input = input_wav.Dim();

  // add noise

  for (int i=0; i < controller.size(); ++i) {
    const WaveData &noise_wav = noise_reader.Value(controller[i].noise_uttid_);
    BaseFloat samp_freq_noise = noise_wav.SampFreq();
    KALDI_ASSERT(samp_freq_input == samp_freq_noise);
      
    const Matrix<BaseFloat> &noise_matrix = noise_wav.Data();
    int32 num_samp_noise = noise_matrix.NumCols();
    Vector<BaseFloat> noise(num_samp_noise);
    noise.CopyRowFromMat(noise_matrix, 0);

    int32 input_start_point = samp_freq_input * controller[i].wav_t_start_;
    int32 input_end_point = samp_freq_input * controller[i].wav_t_end_ - 1;
    int32 noise_start_point = samp_freq_noise * controller[i].noise_t_start_;
    int32 noise_end_point = samp_freq_noise * controller[i].noise_t_end_ - 1;
    BaseFloat snr = controller[i].snr_;

    SubVector<BaseFloat> input_part(input_wav, input_start_point,
                                    input_end_point - input_start_point + 1);
    SubVector<BaseFloat> noise_part(noise, noise_start_point,
                                    noise_end_point - noise_start_point + 1);
    Vector<BaseFloat> selected_noise(input_part.Dim());
    if (noise_part.Dim() < input_part.Dim()) {
      int32 the_rest = selected_noise.Dim();
      while (the_rest > noise_part.Dim()) {
        selected_noise.Range(selected_noise.Dim()-the_rest,
                             noise_part.Dim()).CopyFromVec(noise_part);
        the_rest = the_rest - noise_part.Dim();
      }
      selected_noise.Range(selected_noise.Dim()-the_rest, the_rest).CopyFromVec(
          noise_part.Range(0, the_rest));
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
        "\"head -n 5 a.noiserange | tail -n 1\" --input-channel=0 input.wav "
        "perturbed_input.wav\n";

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
    std::vector<NoiseController> controller;
    if (noise_range != "") {
      int index = noise_range.find_first_of(" ");
      std::string perturbed_utt_id = noise_range.substr(0, index);
      std::string noise_range_content = noise_range.substr(index+1);
      std::vector<std::string> segments;
      SplitStringToVector(noise_range_content, ",", true, &segments);
      GenerateController(segments, &controller);
    }

    WaveData input_wave;
    {
      WaveHolder waveholder;
      Input ki(input_wave_file);
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
    ApplyNoise(noise, controller, input, &output);

    Matrix<BaseFloat> out_matrix(1, num_samp_input);
    out_matrix.CopyRowsFromVec(output);

    WaveData out_wave(samp_freq_input, out_matrix);
    Output ko(output_wave_file, false);
    out_wave.Write(ko.Stream());

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

