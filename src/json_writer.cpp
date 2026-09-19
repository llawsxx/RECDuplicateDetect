#include "recdup/json_writer.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

namespace recdup {
namespace {

std::string quote(const std::string& input) {
  std::ostringstream output;
  output << '"';
  for (unsigned char ch : input) {
    switch (ch) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (ch < 0x20) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned>(ch) << std::dec;
        } else {
          output << static_cast<char>(ch);
        }
    }
  }
  output << '"';
  return output.str();
}

void number(std::ostringstream& output, double value) {
  if (!std::isfinite(value)) output << "null";
  else output << std::fixed << std::setprecision(3) << value;
}

void bytes(std::ostringstream& output, std::int64_t value) {
  if (value < 0) output << "null";
  else output << value;
}

void timelineSegment(std::ostringstream& output,
                     const TimelineSegment& segment,
                     const std::string& indent) {
  output << indent << "{\n"
         << indent << "  \"label\": " << quote(segment.label) << ",\n"
         << indent << "  \"start_byte\": ";
  bytes(output, segment.start_byte);
  output << ",\n" << indent << "  \"end_byte\": ";
  bytes(output, segment.end_byte);
  output << ",\n" << indent << "  \"start_time_seconds\": ";
  number(output, segment.start_seconds);
  output << ",\n" << indent << "  \"end_time_seconds\": ";
  number(output, segment.end_seconds);
  output << ",\n" << indent << "  \"confidence\": ";
  number(output, segment.confidence);
  output << ",\n" << indent << "  \"boundary_uncertainty_seconds\": ";
  number(output, segment.boundary_uncertainty_seconds);
  output << ",\n" << indent << "  \"evidence\": {\n"
         << indent << "    \"repeated_content_coverage\": ";
  number(output, segment.evidence.repeated_content_coverage);
  output << ",\n" << indent << "    \"audio_video_coverage\": ";
  number(output, segment.evidence.audio_video_coverage);
  output << ",\n" << indent << "    \"repeat_family_count\": "
         << segment.evidence.repeat_family_count << ",\n"
         << indent << "    \"break_out_family_id\": ";
  if (segment.evidence.break_out_family_id.empty()) output << "null";
  else output << quote(segment.evidence.break_out_family_id);
  output << ",\n" << indent << "    \"break_in_family_id\": ";
  if (segment.evidence.break_in_family_id.empty()) output << "null";
  else output << quote(segment.evidence.break_in_family_id);
  output << ",\n"
         << indent << "    \"preceded_by_break\": "
         << (segment.evidence.preceded_by_break ? "true" : "false") << ",\n"
         << indent << "    \"followed_by_break\": "
         << (segment.evidence.followed_by_break ? "true" : "false") << "\n"
         << indent << "  }\n"
         << indent << '}';
}

}  // namespace

std::string makeResultJson(const MediaFeatures& media,
                           const std::vector<MatchSpan>& matches,
                           std::size_t database_vectors, bool stored,
                           std::size_t database_vectors_after,
                           const ProgrammeInference* inference) {
  std::ostringstream output;
  output << "{\n"
         << "  \"schema_version\": 3,\n"
         << "  \"input\": {\n"
         << "    \"source_id\": " << quote(media.source_id) << ",\n"
         << "    \"content_type\": " << quote(media.source_type) << ",\n"
         << "    \"path\": " << quote(media.input_path) << ",\n"
         << "    \"file_size\": ";
  bytes(output, media.file_size);
  output << ",\n    \"duration_seconds\": ";
  number(output, media.duration_seconds);
  output << ",\n    \"recoverable_decode_errors\": "
         << media.recoverable_decode_errors
         << ",\n    \"dropped_packets\": " << media.dropped_packets
         << ",\n    \"timestamp_discontinuities\": [";
  if (media.timestamp_discontinuities.empty()) {
    output << "]\n";
  } else {
    output << '\n';
    for (std::size_t i = 0; i < media.timestamp_discontinuities.size(); ++i) {
      const auto& event = media.timestamp_discontinuities[i];
      output << "      {\"stream\": " << quote(event.stream)
             << ", \"direction\": " << quote(event.direction)
             << ", \"previous_raw_seconds\": ";
      number(output, event.previous_raw_seconds);
      output << ", \"current_raw_seconds\": ";
      number(output, event.current_raw_seconds);
      output << ", \"corrected_time_seconds\": ";
      number(output, event.corrected_time_seconds);
      output << '}';
      if (i + 1 != media.timestamp_discontinuities.size()) output << ',';
      output << '\n';
    }
    output << "    ]\n";
  }
  output << "  },\n"
         << "  \"database_vector_count\": " << database_vectors << ",\n"
         << "  \"stored\": " << (stored ? "true" : "false");
  if (stored)
    output << ",\n  \"database_vector_count_after\": "
           << database_vectors_after;
  output << ",\n"
         << "  \"matches\": [";
  if (!matches.empty()) output << '\n';
  for (std::size_t i = 0; i < matches.size(); ++i) {
    const auto& match = matches[i];
    output << "    {\n"
           << "      \"feature\": " << quote(featureKindName(match.kind)) << ",\n"
           << "      \"extractor_id\": " << quote(match.extractor_id) << ",\n"
           << "      \"anchor_count\": " << match.anchor_count << ",\n"
           << "      \"similarity\": ";
    number(output, match.similarity);
    output << ",\n      \"query\": {\n"
           << "        \"start_byte\": ";
    bytes(output, match.query_start_byte);
    output << ",\n        \"end_byte\": ";
    bytes(output, match.query_end_byte);
    output << ",\n        \"start_time_seconds\": ";
    number(output, match.query_start_seconds);
    output << ",\n        \"end_time_seconds\": ";
    number(output, match.query_end_seconds);
    output << "\n      },\n"
           << "      \"reference\": {\n"
           << "        \"source_id\": " << quote(match.reference_id) << ",\n"
           << "        \"name\": " << quote(match.reference_name) << ",\n"
           << "        \"path\": " << quote(match.reference_path) << ",\n"
           << "        \"content_type\": " << quote(match.reference_type)
           << ",\n"
           << "        \"start_byte\": ";
    bytes(output, match.reference_start_byte);
    output << ",\n        \"end_byte\": ";
    bytes(output, match.reference_end_byte);
    output << ",\n        \"start_time_seconds\": ";
    number(output, match.reference_start_seconds);
    output << ",\n        \"end_time_seconds\": ";
    number(output, match.reference_end_seconds);
    output << "\n      }\n    }";
    if (i + 1 != matches.size()) output << ',';
    output << '\n';
  }
  output << "  ]";
  if (inference) {
    output << ",\n  \"content_families\": [";
    if (!inference->content_families.empty()) output << '\n';
    for (std::size_t i = 0; i < inference->content_families.size(); ++i) {
      const auto& family = inference->content_families[i];
      output << "    {\n"
             << "      \"id\": " << quote(family.id) << ",\n"
             << "      \"classification\": "
             << quote(family.classification) << ",\n"
             << "      \"known_content_type\": "
             << quote(family.known_content_type) << ",\n"
             << "      \"confidence\": ";
      number(output, family.confidence);
      output << ",\n      \"typical_duration_seconds\": ";
      number(output, family.typical_duration_seconds);
      output << ",\n      \"has_audio\": "
             << (family.has_audio ? "true" : "false") << ",\n"
             << "      \"has_video\": "
             << (family.has_video ? "true" : "false") << ",\n"
             << "      \"audio_video_confirmed\": "
             << (family.audio_video_confirmed ? "true" : "false") << ",\n"
             << "      \"occurrences\": [";
      if (!family.occurrences.empty()) output << '\n';
      for (std::size_t j = 0; j < family.occurrences.size(); ++j) {
        const auto& occurrence = family.occurrences[j];
        output << "        {\n"
               << "          \"source_id\": " << quote(occurrence.source_id)
               << ",\n"
               << "          \"name\": " << quote(occurrence.source_name)
               << ",\n"
               << "          \"path\": " << quote(occurrence.source_path)
               << ",\n"
               << "          \"content_type\": "
               << quote(occurrence.source_type) << ",\n"
               << "          \"start_byte\": ";
        bytes(output, occurrence.start_byte);
        output << ",\n          \"end_byte\": ";
        bytes(output, occurrence.end_byte);
        output << ",\n          \"start_time_seconds\": ";
        number(output, occurrence.start_seconds);
        output << ",\n          \"end_time_seconds\": ";
        number(output, occurrence.end_seconds);
        output << "\n        }";
        if (j + 1 != family.occurrences.size()) output << ',';
        output << '\n';
      }
      output << "      ]\n    }";
      if (i + 1 != inference->content_families.size()) output << ',';
      output << '\n';
    }
    output << "  ],\n  \"timeline\": [";
    if (!inference->timeline.empty()) output << '\n';
    for (std::size_t i = 0; i < inference->timeline.size(); ++i) {
      timelineSegment(output, inference->timeline[i], "    ");
      if (i + 1 != inference->timeline.size()) output << ',';
      output << '\n';
    }
    output << "  ],\n  \"programme_guesses\": [";
    if (!inference->programme_guesses.empty()) output << '\n';
    for (std::size_t i = 0; i < inference->programme_guesses.size(); ++i) {
      timelineSegment(output, inference->programme_guesses[i], "    ");
      if (i + 1 != inference->programme_guesses.size()) output << ',';
      output << '\n';
    }
    output << "  ],\n  \"programme_recaps\": [";
    if (!inference->programme_recaps.empty()) output << '\n';
    for (std::size_t i = 0; i < inference->programme_recaps.size(); ++i) {
      timelineSegment(output, inference->programme_recaps[i], "    ");
      if (i + 1 != inference->programme_recaps.size()) output << ',';
      output << '\n';
    }
    output << "  ]";
  }
  output << "\n}\n";
  return output.str();
}

}  // namespace recdup
