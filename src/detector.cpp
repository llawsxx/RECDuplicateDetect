#include "recdup/detector.hpp"

#include "recdup/vector_database.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>

namespace recdup {
namespace {

void appendVector(std::vector<FeatureVector>& output,
                  const MediaFeatures& media,
                  const std::string& source_name,
                  const FeatureBucket& bucket,
                  FeatureKind kind,
                  const std::string& extractor_id,
                  const std::vector<float>& values) {
  if (values.empty()) return;
  double norm = 0.0;
  for (float value : values) norm += static_cast<double>(value) * value;
  if (norm <= 1.0e-12) return;
  FeatureVector vector;
  vector.source_id = media.source_id;
  vector.source_name = source_name;
  vector.source_path = media.input_path;
  vector.source_type = media.source_type;
  vector.extractor_id = extractor_id;
  vector.kind = kind;
  vector.start_seconds = bucket.start_seconds;
  vector.end_seconds = bucket.end_seconds;
  vector.start_byte = bucket.start_byte;
  vector.end_byte = bucket.end_byte;
  vector.values = values;
  output.push_back(std::move(vector));
}

struct Anchor {
  const FeatureVector* query = nullptr;
  SearchHit reference;
};

struct GroupKey {
  std::string reference_id;
  std::string extractor_id;
  FeatureKind kind = FeatureKind::VideoPerceptual;
  std::int64_t offset_bin = 0;

  bool operator<(const GroupKey& other) const {
    return std::tie(reference_id, extractor_id, kind, offset_bin) <
           std::tie(other.reference_id, other.extractor_id, other.kind,
                    other.offset_bin);
  }
};

MatchSpan makeRun(const std::vector<Anchor>& anchors,
                  std::size_t begin, std::size_t end) {
  const auto& first = anchors[begin];
  MatchSpan result;
  result.kind = first.query->kind;
  result.extractor_id = first.query->extractor_id;
  result.reference_id = first.reference.source_id;
  result.reference_name = first.reference.source_name;
  result.reference_path = first.reference.source_path;
  result.reference_type = first.reference.source_type;
  result.query_start_seconds = first.query->start_seconds;
  result.query_end_seconds = first.query->end_seconds;
  result.query_start_byte = first.query->start_byte;
  result.query_end_byte = first.query->end_byte;
  result.reference_start_seconds = first.reference.start_seconds;
  result.reference_end_seconds = first.reference.end_seconds;
  result.reference_start_byte = first.reference.start_byte;
  result.reference_end_byte = first.reference.end_byte;
  double score_sum = 0.0;
  for (std::size_t i = begin; i < end; ++i) {
    const auto& anchor = anchors[i];
    result.query_start_seconds =
        std::min(result.query_start_seconds, anchor.query->start_seconds);
    result.query_end_seconds =
        std::max(result.query_end_seconds, anchor.query->end_seconds);
    result.reference_start_seconds =
        std::min(result.reference_start_seconds, anchor.reference.start_seconds);
    result.reference_end_seconds =
        std::max(result.reference_end_seconds, anchor.reference.end_seconds);
    if (result.query_start_byte < 0 ||
        (anchor.query->start_byte >= 0 &&
         anchor.query->start_byte < result.query_start_byte))
      result.query_start_byte = anchor.query->start_byte;
    result.query_end_byte = std::max(result.query_end_byte, anchor.query->end_byte);
    if (result.reference_start_byte < 0 ||
        (anchor.reference.start_byte >= 0 &&
         anchor.reference.start_byte < result.reference_start_byte))
      result.reference_start_byte = anchor.reference.start_byte;
    result.reference_end_byte =
        std::max(result.reference_end_byte, anchor.reference.end_byte);
    score_sum += anchor.reference.similarity;
  }
  result.anchor_count = end - begin;
  result.similarity = static_cast<float>(score_sum / result.anchor_count);
  return result;
}

std::vector<MatchSpan> alignAnchors(std::vector<Anchor> anchors,
                                    const SearchOptions& options) {
  std::map<GroupKey, std::vector<Anchor>> groups;
  for (auto& anchor : anchors) {
    const double offset = anchor.reference.start_seconds -
                          anchor.query->start_seconds;
    const auto bin = static_cast<std::int64_t>(
        std::llround(offset / options.offset_bin_seconds));
    GroupKey key{anchor.reference.source_id, anchor.query->extractor_id,
                 anchor.query->kind, bin};
    groups[std::move(key)].push_back(std::move(anchor));
  }

  std::vector<MatchSpan> matches;
  for (auto& entry : groups) {
    auto& group = entry.second;
    std::sort(group.begin(), group.end(), [](const Anchor& a, const Anchor& b) {
      return std::tie(a.query->start_seconds, a.reference.start_seconds,
                      a.reference.similarity) <
             std::tie(b.query->start_seconds, b.reference.start_seconds,
                      b.reference.similarity);
    });

    // Keep one reference anchor per query sample in an offset bucket.
    std::vector<Anchor> unique;
    for (auto& anchor : group) {
      if (!unique.empty() &&
          std::abs(unique.back().query->start_seconds -
                   anchor.query->start_seconds) < 1.0e-6) {
        if (anchor.reference.similarity > unique.back().reference.similarity)
          unique.back() = std::move(anchor);
      } else {
        unique.push_back(std::move(anchor));
      }
    }

    std::size_t run_begin = 0;
    auto finish = [&](std::size_t run_end) {
      if (run_end <= run_begin) return;
      auto match = makeRun(unique, run_begin, run_end);
      const double duration = std::min(
          match.query_end_seconds - match.query_start_seconds,
          match.reference_end_seconds - match.reference_start_seconds);
      if (match.anchor_count >= options.minimum_anchors &&
          duration >= options.minimum_duration_seconds)
        matches.push_back(std::move(match));
    };

    for (std::size_t i = 1; i <= unique.size(); ++i) {
      bool continuous = i < unique.size();
      if (continuous) {
        const double query_gap = unique[i].query->start_seconds -
                                 unique[i - 1].query->start_seconds;
        const double reference_gap = unique[i].reference.start_seconds -
                                     unique[i - 1].reference.start_seconds;
        continuous = query_gap > 0.0 && reference_gap > 0.0 &&
                     query_gap <= options.maximum_anchor_gap_seconds &&
                     reference_gap <= options.maximum_anchor_gap_seconds &&
                     std::abs(query_gap - reference_gap) <=
                         options.offset_bin_seconds * 1.5;
      }
      if (!continuous) {
        finish(i);
        run_begin = i;
      }
    }
  }

  std::sort(matches.begin(), matches.end(), [](const MatchSpan& a,
                                                const MatchSpan& b) {
    return std::tie(a.query_start_seconds, a.reference_id,
                    a.reference_start_seconds, a.kind) <
           std::tie(b.query_start_seconds, b.reference_id,
                    b.reference_start_seconds, b.kind);
  });
  return matches;
}

}  // namespace

float cosineSimilarity(const std::vector<float>& a,
                       const std::vector<float>& b) {
  if (a.size() != b.size() || a.empty()) return -1.0F;
  double dot = 0.0;
  double aa = 0.0;
  double bb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    aa += static_cast<double>(a[i]) * a[i];
    bb += static_cast<double>(b[i]) * b[i];
  }
  if (aa <= 1.0e-18 || bb <= 1.0e-18) return -1.0F;
  return static_cast<float>(dot / std::sqrt(aa * bb));
}

std::vector<FeatureVector> buildBaseVectors(const MediaFeatures& media,
                                            const std::string& source_name) {
  std::vector<FeatureVector> result;
  result.reserve(media.buckets.size() * 2 + media.audio_buckets.size());
  const bool has_separate_audio_windows = !media.audio_buckets.empty();
  for (const auto& bucket : media.buckets) {
    if (bucket.has_video)
      appendVector(result, media, source_name, bucket,
                   FeatureKind::VideoPerceptual,
                   "perceptual-video-v2:d128", bucket.video);
    if (bucket.has_audio && !has_separate_audio_windows)
      appendVector(result, media, source_name, bucket,
                   FeatureKind::AudioSpectrum,
                   "spectrum-audio-v2:d128", bucket.audio);
  }
  for (const auto& bucket : media.audio_buckets) {
    if (bucket.has_audio)
      appendVector(result, media, source_name, bucket,
                   FeatureKind::AudioSpectrum,
                   "spectrum-audio-v2:d128", bucket.audio);
  }
  return result;
}

std::vector<MatchSpan> detectDuplicates(
    const std::vector<FeatureVector>& query, const VectorDatabase* database,
    const SearchOptions& options) {
  if (options.offset_bin_seconds <= 0.0 ||
      options.maximum_anchor_gap_seconds <= 0.0 ||
      options.minimum_duration_seconds <= 0.0)
    throw std::invalid_argument("alignment durations must be positive");

  std::vector<Anchor> anchors;
  if (database) {
    for (const auto& vector : query) {
      for (auto& hit : database->search(
               vector, options.top_k, options.minimum_similarity)) {
        if (hit.source_id == vector.source_id) continue;
        anchors.push_back({&vector, std::move(hit)});
      }
    }
  }

  if (options.find_internal_duplicates && !query.empty()) {
    VectorDatabase self = VectorDatabase::create();
    self.replaceSource(query.front().source_id, query);
    for (const auto& vector : query) {
      const auto requested = std::max<std::size_t>(64, options.top_k + 16);
      for (auto& hit : self.search(
               vector, requested, options.minimum_similarity)) {
        const double offset = hit.start_seconds - vector.start_seconds;
        if (offset + 1.0e-6 < options.minimum_duration_seconds) continue;
        anchors.push_back({&vector, std::move(hit)});
      }
    }
  }
  return alignAnchors(std::move(anchors), options);
}

}  // namespace recdup
