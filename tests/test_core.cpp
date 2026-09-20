#include "recdup/detector.hpp"
#include "recdup/json_writer.hpp"
#include "recdup/programme_inference.hpp"
#include "recdup/timestamp_normalizer.hpp"
#include "recdup/vector_database.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void expect(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

recdup::MediaFeatures syntheticMedia() {
  recdup::MediaFeatures media;
  media.input_path = "synthetic.ts";
  media.source_id = "source-a";
  media.source_type = "advertisement";
  media.duration_seconds = 20.0;
  media.file_size = 20000;
  for (int second = 0; second < 20; ++second) {
    recdup::FeatureBucket bucket;
    bucket.start_seconds = second;
    bucket.end_seconds = second + 1;
    bucket.start_byte = second * 1000;
    bucket.end_byte = second * 1000 + 999;
    bucket.video = {static_cast<float>((second % 10) + 1),
                    static_cast<float>((second * 3 % 10) + 1)};
    bucket.audio = {static_cast<float>((second % 10) + 2),
                    static_cast<float>((second * 7 % 10) + 1)};
    bucket.has_video = true;
    bucket.has_audio = true;
    media.buckets.push_back(std::move(bucket));
  }
  return media;
}

void testBaseVectorsAndAlignment() {
  auto vectors = recdup::buildBaseVectors(syntheticMedia(), "test");
  expect(vectors.size() == 40, "unexpected base vector count");
  recdup::SearchOptions search;
  search.minimum_similarity = 0.999F;
  search.minimum_duration_seconds = 5.0;
  search.top_k = 32;
  auto matches = recdup::detectDuplicates(vectors, nullptr, search);
  expect(!matches.empty(), "expected aligned internal duplicate");
  bool found = false;
  for (const auto& match : matches) {
    if (match.query_start_seconds == 0.0 &&
        match.reference_start_seconds == 10.0 &&
        match.query_end_seconds >= 10.0 && match.anchor_count >= 10) {
      found = true;
    }
  }
  expect(found, "expected ten-second offset-aligned run");

  auto overlapping_media = syntheticMedia();
  recdup::FeatureBucket first_audio;
  first_audio.start_seconds = 0.0;
  first_audio.end_seconds = 1.0;
  first_audio.start_byte = 0;
  first_audio.end_byte = 999;
  first_audio.audio = {1.0F, 2.0F};
  first_audio.has_audio = true;
  overlapping_media.audio_buckets.push_back(first_audio);
  auto second_audio = first_audio;
  second_audio.start_seconds = 0.5;
  second_audio.end_seconds = 1.5;
  second_audio.start_byte = 500;
  second_audio.end_byte = 1499;
  overlapping_media.audio_buckets.push_back(second_audio);
  const auto overlapping_vectors =
      recdup::buildBaseVectors(overlapping_media, "test");
  expect(overlapping_vectors.size() == 22,
         "separate audio windows did not replace bucket audio");
  const auto& overlap_vector = overlapping_vectors.back();
  expect(overlap_vector.kind == recdup::FeatureKind::AudioSpectrum &&
             overlap_vector.extractor_id == "spectrum-audio-v2:d128" &&
             overlap_vector.start_seconds == 0.5 &&
             overlap_vector.end_seconds == 1.5,
         "overlapping audio vector metadata is wrong");
}

void testDatabaseRoundTrip() {
  auto vectors = recdup::buildBaseVectors(syntheticMedia(), "test");
  auto database = recdup::VectorDatabase::create();
  database.replaceSource("source-a", vectors);
  const auto path = std::filesystem::temp_directory_path() / "recdup-v4-test.recdb";
  database.save(path.string());
  auto loaded = recdup::VectorDatabase::load(path.string());
  expect(loaded.size() == vectors.size(), "database record count changed");
  expect(loaded.sourceCount() == 1, "database source count changed");
  expect(loaded.extractorCount() == 2, "database extractor count changed");
  auto query = vectors.front();
  query.source_id = "query";
  auto hits = loaded.search(query, 1, 0.98F);
  expect(hits.size() == 1, "database failed to find exact vector");
  expect(hits.front().similarity > 0.99F, "quantized similarity is wrong");
  expect(hits.front().source_type == "advertisement",
         "database source type changed");
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

std::vector<recdup::FeatureVector> sourceVectors(const std::string& id,
                                                 const std::string& type) {
  auto media = syntheticMedia();
  media.source_id = id;
  media.source_type = type;
  media.input_path = id + ".ts";
  return recdup::buildBaseVectors(media, id);
}

void testRecordingRetentionLimit() {
  auto database = recdup::VectorDatabase::create();
  database.setMaximumRecordings(2);
  database.replaceSource("ad", sourceVectors("ad", "advertisement"));
  database.replaceSource("rec-1", sourceVectors("rec-1", "recording"));
  database.replaceSource("rec-2", sourceVectors("rec-2", "recording"));
  const auto first_eviction = database.replaceSource(
      "rec-3", sourceVectors("rec-3", "recording"));
  expect(first_eviction.size() == 1 && first_eviction.front() == "rec-1",
         "oldest recording was not evicted");
  expect(database.recordingCount() == 2 && database.sourceCount() == 3,
         "recording limit removed catalogue material");
  expect(database.containsSource("ad") && !database.containsSource("rec-1"),
         "recording eviction selected the wrong source");

  database.replaceSource("rec-2", sourceVectors("rec-2", "recording"));
  const auto refreshed_eviction = database.replaceSource(
      "rec-4", sourceVectors("rec-4", "recording"));
  expect(refreshed_eviction.size() == 1 &&
             refreshed_eviction.front() == "rec-3",
         "replacing a source did not refresh its retention order");

  const auto path =
      std::filesystem::temp_directory_path() / "recdup-v4-retention.recdb";
  database.save(path.string());
  auto loaded = recdup::VectorDatabase::load(path.string());
  expect(loaded.maximumRecordings() == 2 && loaded.recordingCount() == 2,
         "recording limit did not survive database reload");
  const auto persisted_eviction = loaded.replaceSource(
      "rec-5", sourceVectors("rec-5", "recording"));
  expect(persisted_eviction.size() == 1 &&
             persisted_eviction.front() == "rec-2" &&
             loaded.containsSource("rec-4") && loaded.containsSource("rec-5"),
         "retention order did not survive database reload");
  std::error_code ignored;
  std::filesystem::remove(path, ignored);
}

void testLargeApproximateIndex() {
  std::vector<recdup::FeatureVector> vectors;
  vectors.reserve(21000);
  for (int i = 0; i < 21000; ++i) {
    recdup::FeatureVector vector;
    vector.source_id = "large";
    vector.source_name = "large";
    vector.source_path = "large.ts";
    vector.extractor_id = "test-large:d16";
    vector.kind = recdup::FeatureKind::VideoPerceptual;
    vector.start_seconds = i;
    vector.end_seconds = i + 1;
    vector.values.resize(16);
    for (std::size_t d = 0; d < vector.values.size(); ++d) {
      std::uint32_t bits = static_cast<std::uint32_t>(i + 1) * 2654435761U;
      bits ^= static_cast<std::uint32_t>(d + 11) * 2246822519U;
      bits ^= bits >> 13U;
      vector.values[d] = static_cast<float>(static_cast<int>(bits % 2001U) - 1000);
    }
    vectors.push_back(std::move(vector));
  }
  auto query = vectors[12345];
  auto database = recdup::VectorDatabase::create();
  database.replaceSource("large", std::move(vectors));
  auto hits = database.search(query, 5, 0.999F);
  expect(!hits.empty(), "approximate index missed an exact vector");
  bool exact_time = false;
  for (const auto& hit : hits)
    if (hit.start_seconds == 12345.0) exact_time = true;
  expect(exact_time, "approximate index did not return exact vector");
}

void testJsonEscaping() {
  auto media = syntheticMedia();
  media.input_path = "a\\\"b.ts";
  const auto json = recdup::makeResultJson(media, {}, 0, false);
  expect(json.find("a\\\\\\\"b.ts") != std::string::npos,
         "JSON path is not escaped");
  expect(json.find("\"schema_version\": 3") != std::string::npos,
         "JSON schema version is wrong");
  expect(json.find("\"recoverable_decode_errors\": 0") !=
             std::string::npos &&
             json.find("\"dropped_packets\": 0") != std::string::npos,
         "JSON decode recovery counters are missing");
}

void testTimestampNormalization() {
  std::vector<recdup::TimestampDiscontinuity> events;
  double frontier = 0.0;
  recdup::TimestampNormalizer timestamps("video", true, 10.0, 1.0);
  expect(timestamps.map(0.0, true, 1.0, frontier, events) == 0.0,
         "timestamp origin changed");
  expect(timestamps.map(1.0, true, 1.0, frontier, events) == 1.0,
         "continuous timestamp changed");
  expect(timestamps.map(100.0, true, 1.0, frontier, events) == 2.0,
         "forward timestamp jump was not repaired");
  expect(timestamps.map(101.0, true, 1.0, frontier, events) == 3.0,
         "timestamp offset did not persist");
  expect(timestamps.map(10.0, true, 1.0, frontier, events) == 4.0,
         "backward timestamp jump was not repaired");
  expect(events.size() == 2 && events[0].direction == "forward" &&
             events[1].direction == "backward",
         "timestamp discontinuities were not reported");

  std::vector<recdup::TimestampDiscontinuity> audio_events;
  double shared = 0.0;
  recdup::TimestampNormalizer audio("audio", true, 10.0, 1.0);
  audio.map(0.0, true, 1.0, shared, audio_events);
  shared = 30.0;  // The video stream continued while audio was absent.
  expect(audio.map(30.0, true, 1.0, shared, audio_events) == 30.0,
         "a real single-stream gap was incorrectly compressed");
  expect(audio_events.empty(), "a real single-stream gap was reported as a jump");
}

recdup::MatchSpan repeatMatch(recdup::FeatureKind kind, double query_start,
                              double reference_start, double length,
                              float similarity) {
  recdup::MatchSpan match;
  match.kind = kind;
  match.extractor_id = kind == recdup::FeatureKind::AudioSpectrum
                           ? "spectrum-audio-v2:d128"
                           : "perceptual-video-v2:d128";
  match.similarity = similarity;
  match.anchor_count = static_cast<std::size_t>(length);
  match.reference_id = "timeline-source";
  match.reference_path = "timeline.ts";
  match.query_start_seconds = query_start;
  match.query_end_seconds = query_start + length;
  match.reference_start_seconds = reference_start;
  match.reference_end_seconds = reference_start + length;
  return match;
}

recdup::MatchSpan catalogueAdvertisement(double query_start, double length) {
  auto match = repeatMatch(recdup::FeatureKind::AudioSpectrum, query_start,
                           0.0, length, 0.95F);
  match.reference_id = "ad-001";
  match.reference_name = "known advertisement";
  match.reference_path = "ad-001.wav";
  match.reference_type = "advertisement";
  return match;
}

recdup::MatchSpan catalogueMarker(double query_start, double length,
                                  const std::string& role) {
  auto match = repeatMatch(recdup::FeatureKind::AudioSpectrum, query_start,
                           0.0, length, 0.95F);
  match.reference_id = role + "-catalogue";
  match.reference_name = role;
  match.reference_path = role + ".wav";
  match.reference_type = role;
  return match;
}

recdup::MatchSpan externalRepeat(double query_start, double length,
                                 const std::string& reference_id,
                                 double reference_start = 0.0) {
  auto match = repeatMatch(recdup::FeatureKind::AudioSpectrum, query_start,
                           reference_start, length, 0.998F);
  match.reference_id = reference_id;
  match.reference_path = reference_id + ".wav";
  return match;
}

recdup::MediaFeatures timelineMedia(int duration_seconds) {
  recdup::MediaFeatures media;
  media.source_id = "timeline-source";
  media.input_path = "timeline.ts";
  media.duration_seconds = duration_seconds;
  for (int second = 0; second < duration_seconds; ++second) {
    recdup::FeatureBucket bucket;
    bucket.start_seconds = second;
    bucket.end_seconds = second + 1;
    bucket.start_byte = second * 1000;
    bucket.end_byte = second * 1000 + 999;
    media.buckets.push_back(std::move(bucket));
  }
  return media;
}

void testProgrammeInference() {
  auto media = timelineMedia(600);

  std::vector<recdup::MatchSpan> matches{
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 100, 300, 30, 0.998F),
      repeatMatch(recdup::FeatureKind::VideoPerceptual, 100, 300, 30, 0.998F),
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 140, 340, 30, 0.997F),
      repeatMatch(recdup::FeatureKind::VideoPerceptual, 140, 340, 30, 0.997F),
      repeatMatch(recdup::FeatureKind::VideoPerceptual, 200, 400, 20, 0.999F),
      catalogueAdvertisement(500, 15)};
  const auto inference = recdup::inferProgrammeTimeline(media, matches);
  expect(inference.content_families.size() == 4,
         "expected short, visual reuse, and catalogue families");
  bool found_break = false;
  for (const auto& segment : inference.timeline)
    if (segment.label == "ad_break" && segment.start_seconds == 100.0 &&
        segment.end_seconds == 170.0)
      found_break = true;
  expect(found_break, "expected adjacent short repeats to form an ad break");
  bool found_known_ad = false;
  for (const auto& segment : inference.timeline)
    if (segment.label == "ad_break" && segment.start_seconds == 500.0 &&
        segment.end_seconds == 515.0)
      found_known_ad = true;
  expect(found_known_ad, "known advertisement must form an ad break alone");

  recdup::ProgrammeInferenceOptions three_occurrences;
  three_occurrences.minimum_ad_occurrences = 3;
  const auto stricter_occurrence_result =
      recdup::inferProgrammeTimeline(media, matches, three_occurrences);
  bool stricter_found_automatic = false;
  bool stricter_found_catalogued = false;
  for (const auto& segment : stricter_occurrence_result.timeline) {
    if (segment.label != "ad_break") continue;
    stricter_found_automatic |= segment.start_seconds == 100.0 &&
                                segment.end_seconds == 170.0;
    stricter_found_catalogued |= segment.start_seconds == 500.0 &&
                                segment.end_seconds == 515.0;
  }
  expect(!stricter_found_automatic,
         "ad occurrence minimum did not filter twice-seen unknown content");
  expect(stricter_found_catalogued,
         "ad occurrence minimum filtered a catalogued advertisement");

  const std::vector<recdup::MatchSpan> frequent_repeats{
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 100, 300, 10, 0.998F),
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 100, 500, 10, 0.998F),
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 110, 310, 10, 0.998F),
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 110, 510, 10, 0.998F)};
  const auto frequent_result = recdup::inferProgrammeTimeline(
      media, frequent_repeats, three_occurrences);
  bool frequent_found_break = false;
  for (const auto& segment : frequent_result.timeline)
    if (segment.label == "ad_break" && segment.start_seconds == 100.0 &&
        segment.end_seconds == 120.0)
      frequent_found_break = true;
  expect(frequent_found_break,
         "three-appearance families did not provide advertisement evidence");

  const std::vector<recdup::MatchSpan> one_frequent_family{
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 100, 300, 10, 0.998F),
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 100, 500, 10, 0.998F)};
  const auto conservative_single_result = recdup::inferProgrammeTimeline(
      media, one_frequent_family, three_occurrences);
  expect(std::none_of(conservative_single_result.timeline.begin(),
                      conservative_single_result.timeline.end(),
                      [](const recdup::TimelineSegment& segment) {
                        return segment.label == "ad_break";
                      }),
         "the default family minimum accepted a single unknown family");
  auto allow_single_family = three_occurrences;
  allow_single_family.minimum_ad_families = 1;
  const auto single_family_result = recdup::inferProgrammeTimeline(
      media, one_frequent_family, allow_single_family);
  const auto single_family_breaks = static_cast<std::size_t>(std::count_if(
      single_family_result.timeline.begin(), single_family_result.timeline.end(),
      [](const recdup::TimelineSegment& segment) {
        return segment.label == "ad_break";
      }));
  expect(single_family_breaks == 3,
         "configured family minimum did not accept one frequent family");

  bool found_programme = false;
  for (const auto& segment : inference.programme_guesses)
    if (segment.start_seconds == 170.0 && segment.end_seconds == 300.0)
      found_programme = true;
  expect(found_programme, "visual-only reuse must not split programme time");
  const auto json = recdup::makeResultJson(media, matches, 0, false, 0,
                                            &inference);
  expect(json.find("\"programme_guesses\"") != std::string::npos,
         "programme inference missing from JSON");

  const std::vector<recdup::MatchSpan> borderline_audio{
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 250, 450, 10, 0.92F)};
  expect(recdup::inferProgrammeTimeline(media, borderline_audio)
             .content_families.empty(),
         "default audio evidence threshold accepted a weak match");
  recdup::ProgrammeInferenceOptions relaxed_audio;
  relaxed_audio.minimum_audio_similarity = 0.91;
  const auto isolated_repeat =
      recdup::inferProgrammeTimeline(media, borderline_audio, relaxed_audio);
  expect(isolated_repeat.content_families.size() == 1,
         "configured audio evidence threshold was not applied");
  expect(isolated_repeat.timeline.size() == 1 &&
             isolated_repeat.timeline.front().label == "programme" &&
             isolated_repeat.timeline.front().start_seconds == 0.0 &&
             isolated_repeat.timeline.front().end_seconds == 600.0,
         "an isolated short repeat split the programme timeline");

  const std::vector<recdup::MatchSpan> paired_borderline{
      repeatMatch(recdup::FeatureKind::AudioSpectrum, 250, 450, 10, 0.915F),
      repeatMatch(recdup::FeatureKind::VideoPerceptual, 250, 450, 10, 0.97F)};
  expect(recdup::inferProgrammeTimeline(media, paired_borderline)
             .content_families.size() == 1,
         "default confirmation margin did not confirm aligned evidence");
  recdup::ProgrammeInferenceOptions tighter_confirmation;
  tighter_confirmation.audio_video_confirmation_margin = 0.01;
  expect(recdup::inferProgrammeTimeline(media, paired_borderline,
                                        tighter_confirmation)
             .content_families.empty(),
         "configured confirmation margin was not applied");
}

void testBreakMarkers() {
  const auto media = timelineMedia(1000);
  const std::vector<recdup::MatchSpan> catalogue_markers{
      catalogueMarker(100, 8, "break_out"),
      catalogueMarker(170, 8, "break_in")};
  const auto explicit_result =
      recdup::inferProgrammeTimeline(media, catalogue_markers);
  bool found_explicit_break = false;
  for (const auto& segment : explicit_result.timeline) {
    if (segment.label == "ad_break" && segment.start_seconds == 108.0 &&
        segment.end_seconds == 170.0 &&
        !segment.evidence.break_out_family_id.empty() &&
        !segment.evidence.break_in_family_id.empty())
      found_explicit_break = true;
  }
  expect(found_explicit_break,
         "catalogued break markers did not infer an advertisement break");
  recdup::ProgrammeInferenceOptions short_repeat_limit;
  short_repeat_limit.maximum_short_repeat_seconds = 5.0;
  const auto explicit_with_short_limit = recdup::inferProgrammeTimeline(
      media, catalogue_markers, short_repeat_limit);
  bool explicit_survived_short_limit = false;
  for (const auto& segment : explicit_with_short_limit.timeline)
    if (segment.label == "ad_break" && segment.start_seconds == 108.0 &&
        segment.end_seconds == 170.0)
      explicit_survived_short_limit = true;
  expect(explicit_survived_short_limit,
         "short-repeat limit overrode catalogued marker roles");
  const auto marker_json = recdup::makeResultJson(
      media, catalogue_markers, 0, false, 0, &explicit_result);
  expect(marker_json.find("\"break_out_family_id\": \"repeat-") !=
                 std::string::npos &&
             marker_json.find("\"break_in_family_id\": \"repeat-") !=
                 std::string::npos,
         "break marker evidence is missing from JSON");

  std::vector<recdup::MatchSpan> automatic_markers;
  const std::vector<double> starts{100.0, 400.0, 700.0};
  for (std::size_t i = 0; i < starts.size(); ++i) {
    const double start = starts[i];
    automatic_markers.push_back(externalRepeat(start, 8, "shared-out"));
    automatic_markers.push_back(externalRepeat(
        start + 10, 15, "varying-ad-a-" + std::to_string(i)));
    automatic_markers.push_back(externalRepeat(
        start + 27, 15, "varying-ad-b-" + std::to_string(i)));
    automatic_markers.push_back(externalRepeat(start + 44, 8, "shared-in"));
  }
  const auto automatic_result =
      recdup::inferProgrammeTimeline(media, automatic_markers);
  std::size_t break_out_families = 0;
  std::size_t break_in_families = 0;
  for (const auto& family : automatic_result.content_families) {
    if (family.classification == "break_out") ++break_out_families;
    if (family.classification == "break_in") ++break_in_families;
  }
  expect(break_out_families == 1 && break_in_families == 1,
         "stable advertisement-edge markers were not recognized");
  std::size_t marked_breaks = 0;
  for (const auto& segment : automatic_result.timeline) {
    if (segment.label == "ad_break" &&
        !segment.evidence.break_out_family_id.empty() &&
        !segment.evidence.break_in_family_id.empty())
      ++marked_breaks;
  }
  expect(marked_breaks == 3,
         "automatic break markers did not bound every advertisement break");

  std::vector<recdup::MatchSpan> marker_only_evidence;
  for (std::size_t i = 0; i < starts.size(); ++i) {
    const double start = starts[i];
    marker_only_evidence.push_back(externalRepeat(start, 8, "only-out"));
    marker_only_evidence.push_back(externalRepeat(
        start + 10, 15, "single-item-" + std::to_string(i)));
    marker_only_evidence.push_back(
        externalRepeat(start + 27, 8, "only-in"));
  }
  const auto marker_only_result =
      recdup::inferProgrammeTimeline(media, marker_only_evidence);
  std::size_t inferred_markers = 0;
  std::size_t inferred_breaks = 0;
  for (const auto& family : marker_only_result.content_families)
    if (family.classification == "break_out" ||
        family.classification == "break_in")
      ++inferred_markers;
  for (const auto& segment : marker_only_result.timeline)
    if (segment.label == "ad_break") ++inferred_breaks;
  expect(inferred_markers == 2,
         "automatic marker-only test did not recognize its edge markers");
  expect(inferred_breaks == 0,
         "automatic markers created an advertisement break without "
         "independent evidence");

  auto bridged_evidence = automatic_markers;
  bridged_evidence.push_back(externalRepeat(800, 10, "bridge-a"));
  bridged_evidence.push_back(externalRepeat(830, 8, "shared-out"));
  bridged_evidence.push_back(externalRepeat(858, 10, "bridge-b"));
  const auto bridged_result =
      recdup::inferProgrammeTimeline(media, bridged_evidence);
  bool bridge_created_break = false;
  for (const auto& segment : bridged_result.timeline)
    if (segment.label == "ad_break" && segment.end_seconds > 800.0 &&
        segment.start_seconds < 868.0)
      bridge_created_break = true;
  expect(!bridge_created_break,
         "an automatic marker bridged independent repeat families");

  auto interleaved_markers = automatic_markers;
  interleaved_markers.push_back(catalogueMarker(800, 8, "break_out"));
  interleaved_markers.push_back(externalRepeat(850, 8, "shared-out"));
  interleaved_markers.push_back(catalogueMarker(900, 8, "break_in"));
  const auto interleaved_result =
      recdup::inferProgrammeTimeline(media, interleaved_markers);
  bool found_interleaved_explicit_pair = false;
  for (const auto& segment : interleaved_result.timeline)
    if (segment.label == "ad_break" && segment.start_seconds == 808.0 &&
        segment.end_seconds == 900.0)
      found_interleaved_explicit_pair = true;
  expect(found_interleaved_explicit_pair,
         "an automatic marker blocked a catalogued marker pair");
}

void testMinimumAdBreakDuration() {
  const auto media = timelineMedia(500);
  const std::vector<recdup::MatchSpan> matches{
      externalRepeat(100, 4, "short-ad-a"),
      externalRepeat(104, 4, "short-ad-b"),
      catalogueAdvertisement(200, 5),
      catalogueMarker(300, 5, "break_out"),
      catalogueMarker(314, 5, "break_in")};

  const auto default_result = recdup::inferProgrammeTimeline(media, matches);
  bool found_short_automatic = false;
  bool found_short_catalogued_ad = false;
  bool found_short_catalogued_markers = false;
  for (const auto& segment : default_result.timeline) {
    if (segment.label != "ad_break") continue;
    found_short_automatic |= segment.start_seconds == 100.0 &&
                             segment.end_seconds == 108.0;
    found_short_catalogued_ad |= segment.start_seconds == 200.0 &&
                                 segment.end_seconds == 205.0;
    found_short_catalogued_markers |= segment.start_seconds == 305.0 &&
                                      segment.end_seconds == 314.0;
  }
  expect(!found_short_automatic,
         "an automatic break shorter than the minimum was retained");
  expect(found_short_catalogued_ad,
         "the minimum removed a catalogued advertisement");
  expect(found_short_catalogued_markers,
         "the minimum removed a break bounded by catalogued markers");

  recdup::ProgrammeInferenceOptions relaxed;
  relaxed.minimum_ad_break_seconds = 8.0;
  const auto relaxed_result =
      recdup::inferProgrammeTimeline(media, matches, relaxed);
  for (const auto& segment : relaxed_result.timeline)
    if (segment.label == "ad_break" && segment.start_seconds == 100.0 &&
        segment.end_seconds == 108.0)
      found_short_automatic = true;
  expect(found_short_automatic,
         "configured minimum advertisement-break duration was not applied");
}

}  // namespace

int main() {
  try {
    testBaseVectorsAndAlignment();
    testDatabaseRoundTrip();
    testRecordingRetentionLimit();
    testLargeApproximateIndex();
    testJsonEscaping();
    testTimestampNormalization();
    testProgrammeInference();
    testBreakMarkers();
    testMinimumAdBreakDuration();
    std::cout << "All recdup core tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Test failure: " << error.what() << '\n';
    return 1;
  }
}
