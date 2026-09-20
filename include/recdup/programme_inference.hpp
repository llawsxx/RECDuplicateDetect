#pragma once

#include "recdup/types.hpp"

#include <cstddef>
#include <vector>

namespace recdup {

struct ProgrammeInferenceOptions {
  double minimum_programme_seconds = 120.0;
  double maximum_short_repeat_seconds = 180.0;
  double ad_block_gap_seconds = 20.0;
  double minimum_ad_break_seconds = 10.0;
  std::size_t minimum_ad_occurrences = 2;
  double feature_alignment_tolerance_seconds = 3.0;
  double minimum_audio_similarity = 0.985;
  double minimum_video_similarity = 0.985;
  double audio_video_confirmation_margin = 0.02;
  double maximum_marker_seconds = 30.0;
  std::size_t minimum_marker_occurrences = 3;
};

ProgrammeInference inferProgrammeTimeline(
    const MediaFeatures& media,
    const std::vector<MatchSpan>& matches,
    const ProgrammeInferenceOptions& options = {});

}  // namespace recdup
