#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    const char *usage =
        "Get the data chunks.\n"
        "Usage:  fvector-chunk [options...] <wav-rspecifier> <noise-rspecifier>"
        "<utt2dur-rxfilename> <feats-wspecifier>\n";

    // construct all the global objects
    ParseOptions po(usage);
    int32 chunk_size = 120;
    int32 channel = -1;
    int32 shift_time = 60;
    BaseFloat min_duration = 0.0;
    po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, "
                "0 -> left, 1 -> right)");
    po.Register("chunk_size", &chunk_size, "The expected length of the chunk.");
    po.Register("shift-time", &shift_time, "Time shift, which decide the overlap "
                "of two adjacent chunks in the same utterance.");
    po.Register("min-duration", &min_duration, "Minimum duration of segments "
                "to process (in seconds).");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string wav_rspecifier = po.GetArg(1);
    std::string noise_rspecifier = po.GetArg(2);
    std::string utt2dur_rxfilename = po.GetArg(3);
    std::string output_wspecifier = po.GetArg(4);


    SequentialTableReader<WaveHolder> reader(wav_rspecifier);
    RandomAccessTableReader<WaveHolder> noise_reader(noise_rspecifier);
    Input ki(utt2dur_rxfilename);
    BaseFloatMatrixWriter kaldi_writer;  // typedef to TableWriter<something>.

    //Read the utt2dur file
    //the vector--utt2dur is used to randomly select the noise chunk.
    std::vector<std::pair<std::string, double>> utt2dur 
    std::string line;
    while (std::getline(ki.Stream(), line)) {
      num_lines++;
      std::vector<std::string> split_line;
      // Split the line by space or tab and check the number of fields in each
      // line. There must be 2 fields--segment utt_id and duration
      SplitStringToVector(line, " \t\r", true, &split_line);
      if (split_line.size() != 2) {
        KALDI_WARN << "Invalid line in segments file: " << line;
        continue;
      }
      std::string utt = split_line[0],
        duration_str = split_line[1];

      double duration;
      if (!ConvertStringToReal(duration_str, &duration)) {
        KALDI_WARN << "Invalid line in utt2dur file: " << line;
        continue;
      }
      utt2dur.push_back(std::pair(utt, duration))
    }

    //random number in [0, utt2dur_len)
    utt2dur_len = utt2dur.size();

    // Start to chunk the data, compose 1 source chunk and 2 noise chunks into
    // a matrix.
    int32 num_utts = 0, num_success = 0;
    for (; !reader.Done(); reader.Next()) {
      num_utts++;
      std::string utt = reader.Key();
      const WaveData &wave_data = reader.Value();
      if (wave_data.Duration() < min_duration) {
        KALDI_WARN << "File: " << utt << " is too short ("
                   << wave_data.Duration() << " sec): producing no output.";
        continue;
      }
      int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
      {  // This block works out the channel (0=left, 1=right...)
        KALDI_ASSERT(num_chan > 0);  // should have been caught in
        // reading code if no channels.
        if (channel == -1) {
          this_chan = 0;
          if (num_chan != 1)
            KALDI_WARN << "Channel not specified but you have data with "
                       << num_chan  << " channels; defaulting to zero";
        } else {
          if (this_chan >= num_chan) {
            KALDI_WARN << "File with id " << utt << " has "
                       << num_chan << " channels but you specified channel "
                       << channel << ", producing no output.";
            continue;
          }
        }
      }

      SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
      int32 index = 0;
      int32 num_chunk = (waveform.Dim() / wav_data.SampFreq() - chunk_size ) / time_shift + 1;
      try {
        for (index = 0; index < num_chunk; ++index) {
          Matrix<BaseFloat> features;
          int32 source_start = wav_data.SampFreq() * (index * time_shift)
          //1. Generate 2 random number form [0, utt2dur_len)
          //2. From vector utt2dur, get the 2 pairs
          //3. Generate 2 random "start point" number from [0, utt2dur[x][1])
          //4. According to the utt2dur[x][0]--utt_id and startpoint form RandomAccessTable
          //   read noise chunk.
          //5. The features matrix has 3 lines: source, nosie1, noise2. 
          ostringstream utt_id_new;
          utt_id_new << utt << '_' << index;
          kaldi_writer.Write(utt_id_new.str(), features);
        }
      } catch (...) {
        KALDI_WARN << "Failed to compute features for utterance "
                   << utt;
        continue;
      }
      
      if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
      KALDI_VLOG(2) << "Processed features for key " << utt;
      num_success++;
    }
    KALDI_LOG << " Done " << num_success << " out of " << num_utts
              << " utterances.";
    return (num_success != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

