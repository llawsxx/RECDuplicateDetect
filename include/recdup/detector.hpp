#pragma once

#include "recdup/types.hpp"

#include <cstddef>
#include <vector>

namespace recdup {

std::vector<FeatureVector> buildBaseVectors(const MediaFeatures& media,
                                            const std::string& source_name);

struct SearchOptions {
  float minimum_similarity = 0.985F;
  std::size_t top_k = 12;
  bool find_internal_duplicates = true;
  double minimum_duration_seconds = 5.0;
  double offset_bin_seconds = 1.0;
  double maximum_anchor_gap_seconds = 2.5;
  std::size_t minimum_anchors = 3;
};

class VectorDatabase;

std::vector<MatchSpan> detectDuplicates(
    const std::vector<FeatureVector>& query,
    const VectorDatabase* database,
    const SearchOptions& options);

float cosineSimilarity(const std::vector<float>& a,
                       const std::vector<float>& b);

}  // namespace recdup
