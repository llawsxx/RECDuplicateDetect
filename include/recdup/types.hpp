#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace recdup {

enum class FeatureKind : std::uint8_t {
  VideoPerceptual = 1,
  AudioSpectrum = 2
};

inline const char* featureKindName(FeatureKind kind) {
  switch (kind) {
    case FeatureKind::VideoPerceptual: return "video_perceptual";
    case FeatureKind::AudioSpectrum: return "audio_spectrum";
  }
  return "unknown";
}

struct FeatureBucket {
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  std::int64_t start_byte = -1;
  std::int64_t end_byte = -1;
  std::vector<float> video;
  std::vector<float> audio;
  bool has_video = false;
  bool has_audio = false;
};

struct TimestampDiscontinuity {
  std::string stream;
  std::string direction;
  double previous_raw_seconds = 0.0;
  double current_raw_seconds = 0.0;
  double corrected_time_seconds = 0.0;
};

struct MediaFeatures {
  std::string input_path;
  std::string source_id;
  std::string source_type = "unknown";
  double duration_seconds = 0.0;
  std::int64_t file_size = -1;
  std::size_t recoverable_decode_errors = 0;
  std::size_t dropped_packets = 0;
  std::vector<TimestampDiscontinuity> timestamp_discontinuities;
  std::vector<FeatureBucket> buckets;
};

struct FeatureVector {
  std::string source_id;
  std::string source_name;
  std::string source_path;
  std::string source_type = "unknown";
  std::string extractor_id;
  FeatureKind kind = FeatureKind::VideoPerceptual;
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  std::int64_t start_byte = -1;
  std::int64_t end_byte = -1;
  std::vector<float> values;
};

struct MatchSpan {
  FeatureKind kind = FeatureKind::VideoPerceptual;
  std::string extractor_id;
  float similarity = 0.0F;
  std::size_t anchor_count = 0;
  std::string reference_id;
  std::string reference_name;
  std::string reference_path;
  std::string reference_type = "unknown";
  double query_start_seconds = 0.0;
  double query_end_seconds = 0.0;
  std::int64_t query_start_byte = -1;
  std::int64_t query_end_byte = -1;
  double reference_start_seconds = 0.0;
  double reference_end_seconds = 0.0;
  std::int64_t reference_start_byte = -1;
  std::int64_t reference_end_byte = -1;
};

struct RepeatOccurrence {
  std::string source_id;
  std::string source_name;
  std::string source_path;
  std::string source_type = "unknown";
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  std::int64_t start_byte = -1;
  std::int64_t end_byte = -1;
};

struct ContentFamily {
  std::string id;
  std::string classification;
  std::string known_content_type = "unknown";
  double confidence = 0.0;
  double typical_duration_seconds = 0.0;
  bool has_audio = false;
  bool has_video = false;
  bool audio_video_confirmed = false;
  std::vector<RepeatOccurrence> occurrences;
};

struct TimelineEvidence {
  double repeated_content_coverage = 0.0;
  double audio_video_coverage = 0.0;
  std::size_t repeat_family_count = 0;
  std::string break_out_family_id;
  std::string break_in_family_id;
  bool preceded_by_break = false;
  bool followed_by_break = false;
};

struct TimelineSegment {
  std::string label;
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  std::int64_t start_byte = -1;
  std::int64_t end_byte = -1;
  double confidence = 0.0;
  double boundary_uncertainty_seconds = 0.0;
  TimelineEvidence evidence;
};

struct ProgrammeInference {
  std::vector<ContentFamily> content_families;
  std::vector<TimelineSegment> timeline;
  std::vector<TimelineSegment> programme_guesses;
};

}  // namespace recdup
