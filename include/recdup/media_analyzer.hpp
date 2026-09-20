#pragma once

#include "recdup/types.hpp"

#include <cstddef>
#include <functional>
#include <string>

namespace recdup {

enum class AnalysisMode { Auto, Video, Audio, Both };

struct AnalyzerOptions {
  AnalysisMode mode = AnalysisMode::Auto;
  double bucket_seconds = 1.0;
  double audio_hop_seconds = 0.5;
  bool extract_perceptual_video = true;
  bool repair_timestamp_discontinuities = true;
  double timestamp_jump_threshold_seconds = 10.0;
  double timestamp_backwards_tolerance_seconds = 1.0;
  std::function<void(double processed_seconds, double total_seconds)>
      progress_callback;
};

class MediaAnalyzer {
 public:
  explicit MediaAnalyzer(AnalyzerOptions options = {});
  MediaFeatures analyze(const std::string& input_path,
                        const std::string& source_id) const;

 private:
  AnalyzerOptions options_;
};

}  // namespace recdup
