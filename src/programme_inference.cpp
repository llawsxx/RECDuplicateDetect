#include "recdup/programme_inference.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace recdup {
namespace {

double duration(const MatchSpan& match) {
  return std::max(0.0, std::min(
      match.query_end_seconds - match.query_start_seconds,
      match.reference_end_seconds - match.reference_start_seconds));
}

double overlap(double a0, double a1, double b0, double b1) {
  return std::max(0.0, std::min(a1, b1) - std::max(a0, b0));
}

bool sameAlignment(const MatchSpan& a, const MatchSpan& b, double tolerance) {
  if (a.reference_id != b.reference_id) return false;
  if (std::abs(a.query_start_seconds - b.query_start_seconds) > tolerance ||
      std::abs(a.reference_start_seconds - b.reference_start_seconds) > tolerance)
    return false;
  const double query_shorter = std::min(
      a.query_end_seconds - a.query_start_seconds,
      b.query_end_seconds - b.query_start_seconds);
  const double reference_shorter = std::min(
      a.reference_end_seconds - a.reference_start_seconds,
      b.reference_end_seconds - b.reference_start_seconds);
  return query_shorter > 0.0 && reference_shorter > 0.0 &&
         overlap(a.query_start_seconds, a.query_end_seconds,
                 b.query_start_seconds, b.query_end_seconds) >=
             query_shorter * 0.70 &&
         overlap(a.reference_start_seconds, a.reference_end_seconds,
                 b.reference_start_seconds, b.reference_end_seconds) >=
             reference_shorter * 0.70;
}

class DisjointSet {
 public:
  std::size_t add() {
    const std::size_t index = parent_.size();
    parent_.push_back(index);
    rank_.push_back(0);
    return index;
  }

  std::size_t find(std::size_t value) {
    if (parent_[value] != value) parent_[value] = find(parent_[value]);
    return parent_[value];
  }

  void join(std::size_t a, std::size_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (rank_[a] < rank_[b]) std::swap(a, b);
    parent_[b] = a;
    if (rank_[a] == rank_[b]) ++rank_[a];
  }

 private:
  std::vector<std::size_t> parent_;
  std::vector<unsigned> rank_;
};

struct SelectedMatch {
  const MatchSpan* match = nullptr;
  bool confirmed = false;
  std::size_t query_node = 0;
  std::size_t reference_node = 0;
};

bool equivalentOccurrence(const RepeatOccurrence& a, const RepeatOccurrence& b,
                          double tolerance) {
  if (a.source_id != b.source_id) return false;
  const double shorter = std::min(a.end_seconds - a.start_seconds,
                                  b.end_seconds - b.start_seconds);
  if (shorter <= 0.0) return false;
  const double shared = overlap(a.start_seconds, a.end_seconds,
                                b.start_seconds, b.end_seconds);
  return shared >= shorter * 0.70 &&
         (std::abs(a.start_seconds - b.start_seconds) <= tolerance ||
          std::abs(a.end_seconds - b.end_seconds) <= tolerance);
}

void mergeOccurrence(RepeatOccurrence& target, const RepeatOccurrence& source) {
  target.start_seconds = std::min(target.start_seconds, source.start_seconds);
  target.end_seconds = std::max(target.end_seconds, source.end_seconds);
  if (target.start_byte < 0 ||
      (source.start_byte >= 0 && source.start_byte < target.start_byte))
    target.start_byte = source.start_byte;
  target.end_byte = std::max(target.end_byte, source.end_byte);
  if (target.source_name.empty()) target.source_name = source.source_name;
  if (target.source_path.empty()) target.source_path = source.source_path;
  if (target.source_type == "unknown") target.source_type = source.source_type;
}

double medianDuration(const std::vector<RepeatOccurrence>& occurrences) {
  std::vector<double> values;
  values.reserve(occurrences.size());
  for (const auto& occurrence : occurrences)
    values.push_back(occurrence.end_seconds - occurrence.start_seconds);
  std::sort(values.begin(), values.end());
  if (values.empty()) return 0.0;
  const std::size_t middle = values.size() / 2;
  return values.size() % 2 == 0
             ? (values[middle - 1] + values[middle]) * 0.5
             : values[middle];
}

std::int64_t byteAt(const MediaFeatures& media, double seconds, bool end) {
  if (media.buckets.empty()) return -1;
  const double position = std::max(0.0, seconds - (end ? 1.0e-6 : 0.0));
  const auto index = std::min<std::size_t>(
      media.buckets.size() - 1, static_cast<std::size_t>(std::floor(position)));
  return end ? media.buckets[index].end_byte : media.buckets[index].start_byte;
}

double unionCoverage(std::vector<std::pair<double, double>> intervals,
                     double start, double end) {
  if (end <= start || intervals.empty()) return 0.0;
  for (auto& interval : intervals) {
    interval.first = std::max(start, interval.first);
    interval.second = std::min(end, interval.second);
  }
  intervals.erase(std::remove_if(intervals.begin(), intervals.end(),
                                 [](const auto& value) {
                                   return value.second <= value.first;
                                 }),
                  intervals.end());
  std::sort(intervals.begin(), intervals.end());
  double covered = 0.0;
  double current_start = 0.0;
  double current_end = 0.0;
  bool active = false;
  for (const auto& interval : intervals) {
    if (!active || interval.first > current_end) {
      if (active) covered += current_end - current_start;
      current_start = interval.first;
      current_end = interval.second;
      active = true;
    } else {
      current_end = std::max(current_end, interval.second);
    }
  }
  if (active) covered += current_end - current_start;
  return std::clamp(covered / (end - start), 0.0, 1.0);
}

struct Candidate {
  double start = 0.0;
  double end = 0.0;
  std::size_t family = 0;
  double confidence = 0.0;
  bool confirmed = false;
  bool known_advertisement = false;
};

TimelineEvidence evidenceFor(
    double start, double end, const std::string& source_id,
    const std::vector<ContentFamily>& families) {
  TimelineEvidence result;
  std::vector<std::pair<double, double>> repeats;
  std::vector<std::pair<double, double>> confirmed;
  std::set<std::size_t> used_families;
  for (std::size_t i = 0; i < families.size(); ++i) {
    for (const auto& occurrence : families[i].occurrences) {
      if (occurrence.source_id != source_id ||
          overlap(start, end, occurrence.start_seconds,
                  occurrence.end_seconds) <= 0.0)
        continue;
      repeats.emplace_back(occurrence.start_seconds, occurrence.end_seconds);
      if (families[i].audio_video_confirmed)
        confirmed.emplace_back(occurrence.start_seconds, occurrence.end_seconds);
      used_families.insert(i);
    }
  }
  result.repeated_content_coverage = unionCoverage(std::move(repeats), start, end);
  result.audio_video_coverage = unionCoverage(std::move(confirmed), start, end);
  result.repeat_family_count = used_families.size();
  return result;
}

TimelineSegment makeSegment(const MediaFeatures& media, std::string label,
                            double start, double end, double confidence,
                            double uncertainty,
                            const std::vector<ContentFamily>& families) {
  TimelineSegment result;
  result.label = std::move(label);
  result.start_seconds = start;
  result.end_seconds = end;
  result.start_byte = byteAt(media, start, false);
  result.end_byte = byteAt(media, end, true);
  result.confidence = std::clamp(confidence, 0.0, 1.0);
  result.boundary_uncertainty_seconds = uncertainty;
  result.evidence = evidenceFor(start, end, media.source_id, families);
  return result;
}

}  // namespace

ProgrammeInference inferProgrammeTimeline(
    const MediaFeatures& media, const std::vector<MatchSpan>& matches,
    const ProgrammeInferenceOptions& options) {
  if (options.minimum_programme_seconds <= 0.0 ||
      options.maximum_short_repeat_seconds <= 0.0 ||
      options.ad_block_gap_seconds < 0.0 ||
      !std::isfinite(options.minimum_ad_break_seconds) ||
      options.minimum_ad_break_seconds < 0.0 ||
      options.minimum_ad_occurrences < 2 ||
      options.feature_alignment_tolerance_seconds < 0.0 ||
      !std::isfinite(options.minimum_audio_similarity) ||
      options.minimum_audio_similarity < 0.0 ||
      options.minimum_audio_similarity > 1.0 ||
      !std::isfinite(options.minimum_video_similarity) ||
      options.minimum_video_similarity < 0.0 ||
      options.minimum_video_similarity > 1.0 ||
      !std::isfinite(options.audio_video_confirmation_margin) ||
      options.audio_video_confirmation_margin < 0.0 ||
      options.audio_video_confirmation_margin > 1.0 ||
      !std::isfinite(options.maximum_marker_seconds) ||
      options.maximum_marker_seconds <= 0.0 ||
      options.minimum_marker_occurrences < 2)
    throw std::invalid_argument("invalid programme inference options");

  std::vector<bool> confirmed(matches.size(), false);
  for (std::size_t i = 0; i < matches.size(); ++i) {
    if (matches[i].kind != FeatureKind::AudioSpectrum ||
        matches[i].similarity < options.minimum_audio_similarity -
                                    options.audio_video_confirmation_margin)
      continue;
    for (std::size_t j = 0; j < matches.size(); ++j) {
      if (matches[j].kind != FeatureKind::VideoPerceptual ||
          matches[j].similarity < options.minimum_video_similarity -
                                      options.audio_video_confirmation_margin)
        continue;
      if (sameAlignment(matches[i], matches[j],
                        options.feature_alignment_tolerance_seconds)) {
        confirmed[i] = true;
        confirmed[j] = true;
      }
    }
  }

  DisjointSet sets;
  std::vector<RepeatOccurrence> nodes;
  std::vector<SelectedMatch> selected;
  for (std::size_t i = 0; i < matches.size(); ++i) {
    const auto& match = matches[i];
    const bool strong_audio = match.kind == FeatureKind::AudioSpectrum &&
                              match.similarity >= options.minimum_audio_similarity;
    const bool strong_video = match.kind == FeatureKind::VideoPerceptual &&
                              match.similarity >= options.minimum_video_similarity;
    const bool known_catalogue_item =
        match.reference_type == "advertisement" ||
        match.reference_type == "programme" ||
        match.reference_type == "break_out" ||
        match.reference_type == "break_in";
    if (!known_catalogue_item && !confirmed[i] && !strong_audio && !strong_video)
      continue;
    if (duration(match) <= 0.0) continue;

    RepeatOccurrence query;
    query.source_id = media.source_id;
    query.source_path = media.input_path;
    query.source_type = media.source_type;
    query.start_seconds = match.query_start_seconds;
    query.end_seconds = match.query_end_seconds;
    query.start_byte = match.query_start_byte;
    query.end_byte = match.query_end_byte;

    RepeatOccurrence reference;
    reference.source_id = match.reference_id;
    reference.source_name = match.reference_name;
    reference.source_path = match.reference_path;
    reference.source_type = match.reference_type;
    reference.start_seconds = match.reference_start_seconds;
    reference.end_seconds = match.reference_end_seconds;
    reference.start_byte = match.reference_start_byte;
    reference.end_byte = match.reference_end_byte;

    const std::size_t query_node = sets.add();
    nodes.push_back(std::move(query));
    const std::size_t reference_node = sets.add();
    nodes.push_back(std::move(reference));
    sets.join(query_node, reference_node);
    selected.push_back({&match, confirmed[i], query_node, reference_node});
  }

  std::map<std::string, std::vector<std::size_t>> nodes_by_source;
  for (std::size_t i = 0; i < nodes.size(); ++i)
    nodes_by_source[nodes[i].source_id].push_back(i);
  for (auto& entry : nodes_by_source) {
    auto& indices = entry.second;
    std::sort(indices.begin(), indices.end(), [&](std::size_t a, std::size_t b) {
      return nodes[a].start_seconds < nodes[b].start_seconds;
    });
    for (std::size_t i = 0; i < indices.size(); ++i) {
      for (std::size_t j = i + 1; j < indices.size(); ++j) {
        if (nodes[indices[j]].start_seconds - nodes[indices[i]].start_seconds >
            options.feature_alignment_tolerance_seconds)
          break;
        if (equivalentOccurrence(nodes[indices[i]], nodes[indices[j]],
                                 options.feature_alignment_tolerance_seconds))
          sets.join(indices[i], indices[j]);
      }
    }
  }

  struct FamilyBuilder {
    std::vector<RepeatOccurrence> occurrences;
    bool has_audio = false;
    bool has_video = false;
    bool confirmed = false;
    double score_sum = 0.0;
    std::size_t score_count = 0;
  };
  std::map<std::size_t, FamilyBuilder> builders;
  for (std::size_t i = 0; i < nodes.size(); ++i)
    builders[sets.find(i)].occurrences.push_back(nodes[i]);
  for (const auto& item : selected) {
    auto& builder = builders[sets.find(item.query_node)];
    builder.has_audio |= item.match->kind == FeatureKind::AudioSpectrum;
    builder.has_video |= item.match->kind == FeatureKind::VideoPerceptual;
    builder.confirmed |= item.confirmed;
    builder.score_sum += item.match->similarity;
    ++builder.score_count;
  }

  ProgrammeInference result;
  for (auto& entry : builders) {
    auto& builder = entry.second;
    std::sort(builder.occurrences.begin(), builder.occurrences.end(),
              [](const RepeatOccurrence& a, const RepeatOccurrence& b) {
                return std::tie(a.source_id, a.start_seconds, a.end_seconds) <
                       std::tie(b.source_id, b.start_seconds, b.end_seconds);
              });
    std::vector<RepeatOccurrence> occurrences;
    for (const auto& occurrence : builder.occurrences) {
      if (!occurrences.empty() &&
          equivalentOccurrence(occurrences.back(), occurrence,
                               options.feature_alignment_tolerance_seconds))
        mergeOccurrence(occurrences.back(), occurrence);
      else
        occurrences.push_back(occurrence);
    }
    if (occurrences.size() < 2) continue;

    ContentFamily family;
    family.occurrences = std::move(occurrences);
    family.typical_duration_seconds = medianDuration(family.occurrences);
    family.has_audio = builder.has_audio;
    family.has_video = builder.has_video;
    family.audio_video_confirmed = builder.confirmed;
    bool known_advertisement = false;
    bool known_programme = false;
    bool known_break_out = false;
    bool known_break_in = false;
    for (const auto& occurrence : family.occurrences) {
      known_advertisement |= occurrence.source_type == "advertisement";
      known_programme |= occurrence.source_type == "programme";
      known_break_out |= occurrence.source_type == "break_out";
      known_break_in |= occurrence.source_type == "break_in";
    }
    if (known_advertisement)
      family.known_content_type = "advertisement";
    else if (known_programme)
      family.known_content_type = "programme";
    else if (known_break_out && !known_break_in)
      family.known_content_type = "break_out";
    else if (known_break_in && !known_break_out)
      family.known_content_type = "break_in";
    if (known_advertisement)
      family.classification = "advertisement";
    else if (known_programme)
      family.classification = "programme_repeat";
    else if (known_break_out && !known_break_in)
      family.classification = "break_out";
    else if (known_break_in && !known_break_out)
      family.classification = "break_in";
    else if (family.typical_duration_seconds >
             options.maximum_short_repeat_seconds)
      family.classification = "programme_repeat";
    else if (!family.has_audio)
      family.classification = "visual_reuse";
    else
      family.classification = "short_repeat";
    const double average_score = builder.score_count == 0
                                     ? 0.0
                                     : builder.score_sum / builder.score_count;
    const double base = known_advertisement ? 0.94
                        : known_programme ? 0.90
                        : known_break_out || known_break_in ? 0.92
                        : family.audio_video_confirmed ? 0.88
                        : family.has_audio ? 0.76
                                           : 0.48;
    family.confidence = std::clamp(
        base + 0.15 * std::max(0.0, average_score - 0.90) / 0.10 +
            std::min(0.07, (family.occurrences.size() - 2) * 0.02),
        0.0, 0.99);
    result.content_families.push_back(std::move(family));
  }

  std::sort(result.content_families.begin(), result.content_families.end(),
            [&](const ContentFamily& a, const ContentFamily& b) {
              auto first = [&](const ContentFamily& family) {
                double value = media.duration_seconds + 1.0;
                for (const auto& occurrence : family.occurrences)
                  if (occurrence.source_id == media.source_id)
                    value = std::min(value, occurrence.start_seconds);
                return value;
              };
              return std::make_tuple(first(a), a.classification,
                                     a.typical_duration_seconds) <
                     std::make_tuple(first(b), b.classification,
                                     b.typical_duration_seconds);
            });
  for (std::size_t i = 0; i < result.content_families.size(); ++i)
    result.content_families[i].id = "repeat-" + std::to_string(i + 1);

  std::vector<Candidate> candidates;
  for (std::size_t i = 0; i < result.content_families.size(); ++i) {
    const auto& family = result.content_families[i];
    const bool known_advertisement =
        family.known_content_type == "advertisement";
    if ((!known_advertisement && family.classification != "short_repeat") ||
        family.confidence < 0.75 ||
        (!known_advertisement &&
         family.occurrences.size() < options.minimum_ad_occurrences) ||
        (!known_advertisement &&
         family.typical_duration_seconds > options.maximum_short_repeat_seconds))
      continue;
    for (const auto& occurrence : family.occurrences) {
      if (occurrence.source_id != media.source_id) continue;
      candidates.push_back({std::clamp(occurrence.start_seconds, 0.0,
                                       media.duration_seconds),
                            std::clamp(occurrence.end_seconds, 0.0,
                                       media.duration_seconds),
                            i, family.confidence,
                            family.audio_video_confirmed,
                            known_advertisement});
    }
  }
  std::sort(candidates.begin(), candidates.end(), [](const Candidate& a,
                                                       const Candidate& b) {
    return std::tie(a.start, a.end, a.family) <
           std::tie(b.start, b.end, b.family);
  });

  struct Block {
    double start = 0.0;
    double end = 0.0;
    std::vector<Candidate> items;
  };
  auto buildBlocks = [&](const std::vector<Candidate>& source) {
    std::vector<Block> result;
    for (const auto& candidate : source) {
      if (candidate.end <= candidate.start) continue;
      if (result.empty() ||
          candidate.start >
              result.back().end + options.ad_block_gap_seconds) {
        result.push_back({candidate.start, candidate.end, {candidate}});
      } else {
        result.back().end = std::max(result.back().end, candidate.end);
        result.back().items.push_back(candidate);
      }
    }
    return result;
  };
  const auto blocks = buildBlocks(candidates);

  const double maximum_block =
      std::max(360.0, options.maximum_short_repeat_seconds * 2.0);

  struct FamilyExtent {
    std::size_t family = 0;
    double start = 0.0;
    double end = 0.0;
  };
  auto familyExtents = [](const Block& block) {
    std::map<std::size_t, FamilyExtent> by_family;
    for (const auto& item : block.items) {
      auto inserted = by_family.emplace(
          item.family, FamilyExtent{item.family, item.start, item.end});
      if (!inserted.second) {
        inserted.first->second.start =
            std::min(inserted.first->second.start, item.start);
        inserted.first->second.end =
            std::max(inserted.first->second.end, item.end);
      }
    }
    std::vector<FamilyExtent> result;
    result.reserve(by_family.size());
    for (const auto& entry : by_family) result.push_back(entry.second);
    return result;
  };
  auto isAdEvidence = [&](const Block& block) {
    std::set<std::size_t> families;
    bool known_advertisement = false;
    for (const auto& item : block.items) {
      families.insert(item.family);
      known_advertisement |= item.known_advertisement;
    }
    return known_advertisement || families.size() >= 2;
  };

  // A repeated short family is inferred as a transition marker only when it
  // consistently occupies the same edge of several ad-like blocks and its
  // inward neighbor changes. The changing-neighbor requirement prevents a
  // fixed first or last advertisement from being promoted to a marker.
  struct EdgeStats {
    std::size_t block_count = 0;
    std::set<std::size_t> inward_neighbors;
  };
  std::vector<EdgeStats> leading(result.content_families.size());
  std::vector<EdgeStats> trailing(result.content_families.size());
  for (const auto& block : blocks) {
    if (!isAdEvidence(block) || block.end - block.start > maximum_block) continue;
    auto by_start = familyExtents(block);
    if (by_start.size() < 3) continue;
    std::sort(by_start.begin(), by_start.end(), [](const FamilyExtent& a,
                                                   const FamilyExtent& b) {
      return std::tie(a.start, a.end, a.family) <
             std::tie(b.start, b.end, b.family);
    });
    if (std::abs(by_start[0].start - by_start[1].start) > 1.0e-6) {
      ++leading[by_start[0].family].block_count;
      leading[by_start[0].family].inward_neighbors.insert(
          by_start[1].family);
    }

    auto by_end = by_start;
    std::sort(by_end.begin(), by_end.end(), [](const FamilyExtent& a,
                                               const FamilyExtent& b) {
      return std::tie(a.end, a.start, a.family) <
             std::tie(b.end, b.start, b.family);
    });
    const std::size_t last = by_end.size() - 1;
    if (std::abs(by_end[last].end - by_end[last - 1].end) > 1.0e-6) {
      ++trailing[by_end[last].family].block_count;
      trailing[by_end[last].family].inward_neighbors.insert(
          by_end[last - 1].family);
    }
  }
  for (std::size_t i = 0; i < result.content_families.size(); ++i) {
    auto& family = result.content_families[i];
    if (family.classification != "short_repeat" ||
        family.typical_duration_seconds > options.maximum_marker_seconds)
      continue;
    const auto current_occurrences = static_cast<std::size_t>(std::count_if(
        family.occurrences.begin(), family.occurrences.end(),
        [&](const RepeatOccurrence& occurrence) {
          return occurrence.source_id == media.source_id;
        }));
    if (current_occurrences < options.minimum_marker_occurrences) continue;
    const bool is_leading =
        leading[i].block_count >= options.minimum_marker_occurrences &&
        leading[i].inward_neighbors.size() >= 2;
    const bool is_trailing =
        trailing[i].block_count >= options.minimum_marker_occurrences &&
        trailing[i].inward_neighbors.size() >= 2;
    if (is_leading != is_trailing)
      family.classification = is_leading ? "break_out" : "break_in";
  }

  struct MarkerOccurrence {
    double start = 0.0;
    double end = 0.0;
    std::size_t family = 0;
    bool break_out = false;
    bool catalogue = false;
    double confidence = 0.0;
  };
  std::vector<MarkerOccurrence> markers;
  for (std::size_t i = 0; i < result.content_families.size(); ++i) {
    const auto& family = result.content_families[i];
    const bool break_out = family.classification == "break_out";
    const bool break_in = family.classification == "break_in";
    if (!break_out && !break_in) continue;
    for (const auto& occurrence : family.occurrences) {
      if (occurrence.source_id != media.source_id) continue;
      markers.push_back({occurrence.start_seconds, occurrence.end_seconds, i,
                         break_out,
                         family.known_content_type == family.classification,
                         family.confidence});
    }
  }
  std::sort(markers.begin(), markers.end(), [](const MarkerOccurrence& a,
                                                const MarkerOccurrence& b) {
    return std::tie(a.start, a.end, a.family) <
           std::tie(b.start, b.end, b.family);
  });

  // Automatic markers may only refine an ad block. Rebuild the evidence
  // blocks without markers so a marker cannot bridge otherwise independent
  // repeated families and create a break by itself.
  std::vector<Candidate> independent_candidates;
  independent_candidates.reserve(candidates.size());
  for (const auto& candidate : candidates) {
    const auto& classification =
        result.content_families[candidate.family].classification;
    if (classification != "break_out" && classification != "break_in")
      independent_candidates.push_back(candidate);
  }
  const auto evidence_blocks = buildBlocks(independent_candidates);

  struct BreakProposal {
    double start = 0.0;
    double end = 0.0;
    double confidence = 0.0;
    bool minimum_duration_exempt = false;
    std::string break_out_family_id;
    std::string break_in_family_id;
  };
  std::vector<BreakProposal> proposals;
  for (const auto& block : evidence_blocks) {
    if (!isAdEvidence(block) || block.end - block.start > maximum_block) continue;
    std::set<std::size_t> non_marker_families;
    double start = media.duration_seconds;
    double end = 0.0;
    double confidence = 0.0;
    bool confirmed_block = false;
    bool known_advertisement = false;
    bool has_non_marker = false;
    for (const auto& item : block.items) {
      const auto& classification =
          result.content_families[item.family].classification;
      if (classification == "break_out" || classification == "break_in")
        continue;
      has_non_marker = true;
      non_marker_families.insert(item.family);
      start = std::min(start, item.start);
      end = std::max(end, item.end);
      confidence = std::max(confidence, item.confidence);
      confirmed_block |= item.confirmed;
      known_advertisement |= item.known_advertisement;
    }
    if (!has_non_marker || end <= start ||
        (!known_advertisement && non_marker_families.size() < 2))
      continue;

    const MarkerOccurrence* preceding = nullptr;
    const MarkerOccurrence* following = nullptr;
    for (const auto& marker : markers) {
      if (marker.break_out && marker.start <= start &&
          marker.end <= start + options.feature_alignment_tolerance_seconds &&
          start - marker.end <= options.ad_block_gap_seconds &&
          (!preceding || marker.end > preceding->end))
        preceding = &marker;
      if (!marker.break_out && marker.end >= end &&
          marker.start >= end - options.feature_alignment_tolerance_seconds &&
          marker.start - end <= options.ad_block_gap_seconds &&
          (!following || marker.start < following->start))
        following = &marker;
    }
    BreakProposal proposal;
    proposal.start = preceding ? preceding->end : start;
    proposal.end = following ? following->start : end;
    proposal.confidence = confidence + 0.04 +
                          (confirmed_block ? 0.03 : 0.0) +
                          (known_advertisement ? 0.05 : 0.0);
    proposal.minimum_duration_exempt = known_advertisement;
    if (preceding)
      proposal.break_out_family_id =
          result.content_families[preceding->family].id;
    if (following)
      proposal.break_in_family_id =
          result.content_families[following->family].id;
    if (proposal.end > proposal.start) proposals.push_back(std::move(proposal));
  }

  // A break_out followed by a break_in is sufficient to infer a break even
  // when none of the intervening advertisements repeated or were catalogued,
  // but only when both markers were explicitly catalogued. Automatically
  // inferred markers may refine an independently established break, never
  // create one by themselves.
  std::vector<MarkerOccurrence> catalogue_markers;
  for (const auto& marker : markers)
    if (marker.catalogue) catalogue_markers.push_back(marker);
  for (std::size_t i = 0; i < catalogue_markers.size(); ++i) {
    if (!catalogue_markers[i].break_out) continue;
    for (std::size_t j = i + 1; j < catalogue_markers.size(); ++j) {
      if (catalogue_markers[j].start < catalogue_markers[i].end) continue;
      if (catalogue_markers[j].break_out) break;
      const double length =
          catalogue_markers[j].start - catalogue_markers[i].end;
      if (length <= 0.0 || length > maximum_block) break;
      BreakProposal proposal;
      proposal.start = catalogue_markers[i].end;
      proposal.end = catalogue_markers[j].start;
      proposal.confidence =
          std::min(catalogue_markers[i].confidence,
                   catalogue_markers[j].confidence) + 0.06;
      proposal.minimum_duration_exempt = true;
      proposal.break_out_family_id =
          result.content_families[catalogue_markers[i].family].id;
      proposal.break_in_family_id =
          result.content_families[catalogue_markers[j].family].id;
      proposals.push_back(std::move(proposal));
      break;
    }
  }

  std::sort(proposals.begin(), proposals.end(), [](const BreakProposal& a,
                                                    const BreakProposal& b) {
    return std::tie(a.start, a.end) < std::tie(b.start, b.end);
  });
  std::vector<BreakProposal> merged_proposals;
  for (const auto& proposal : proposals) {
    if (merged_proposals.empty() ||
        proposal.start > merged_proposals.back().end +
                             options.feature_alignment_tolerance_seconds) {
      merged_proposals.push_back(proposal);
      continue;
    }
    auto& merged = merged_proposals.back();
    merged.end = std::max(merged.end, proposal.end);
    merged.confidence = std::max(merged.confidence, proposal.confidence);
    merged.minimum_duration_exempt |= proposal.minimum_duration_exempt;
    if (merged.break_out_family_id.empty())
      merged.break_out_family_id = proposal.break_out_family_id;
    if (!proposal.break_in_family_id.empty())
      merged.break_in_family_id = proposal.break_in_family_id;
  }

  std::vector<TimelineSegment> interruptions;
  interruptions.reserve(merged_proposals.size());
  for (const auto& proposal : merged_proposals) {
    if (!proposal.minimum_duration_exempt &&
        proposal.end - proposal.start < options.minimum_ad_break_seconds)
      continue;
    auto segment = makeSegment(media, "ad_break", proposal.start, proposal.end,
                               proposal.confidence, 2.0,
                               result.content_families);
    segment.evidence.break_out_family_id = proposal.break_out_family_id;
    segment.evidence.break_in_family_id = proposal.break_in_family_id;
    interruptions.push_back(std::move(segment));
  }

  double cursor = 0.0;
  for (std::size_t i = 0; i <= interruptions.size(); ++i) {
    const double boundary = i < interruptions.size()
                                ? interruptions[i].start_seconds
                                : media.duration_seconds;
    if (boundary > cursor + 1.0e-6) {
      const double length = boundary - cursor;
      const bool preceded = i > 0;
      const bool followed = i < interruptions.size();
      const bool programme = length >= options.minimum_programme_seconds;
      double confidence = programme ? 0.55 : 0.35;
      if (programme && preceded) confidence += 0.15;
      if (programme && followed) confidence += 0.15;
      if (programme && length >= 600.0) confidence += 0.08;
      auto segment = makeSegment(media, programme ? "programme" : "unknown",
                                 cursor, boundary, confidence,
                                 preceded || followed ? 2.0 : 10.0,
                                 result.content_families);
      segment.evidence.preceded_by_break = preceded;
      segment.evidence.followed_by_break = followed;
      result.timeline.push_back(segment);
      if (programme) result.programme_guesses.push_back(std::move(segment));
    }
    if (i < interruptions.size()) {
      result.timeline.push_back(interruptions[i]);
      cursor = std::max(cursor, interruptions[i].end_seconds);
    }
  }
  return result;
}

}  // namespace recdup
