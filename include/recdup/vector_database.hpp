#pragma once

#include "recdup/types.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace recdup {

struct SearchHit {
  float similarity = 0.0F;
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
};

class VectorDatabase {
 public:
  static VectorDatabase load(const std::string& path);
  static VectorDatabase create();

  std::vector<std::string> replaceSource(
      const std::string& source_id, std::vector<FeatureVector> vectors);
  std::vector<std::string> setMaximumRecordings(std::size_t maximum);
  void save(const std::string& path) const;

  std::vector<SearchHit> search(const FeatureVector& query,
                                std::size_t top_k,
                                float minimum_similarity) const;

  std::size_t size() const;
  std::size_t sourceCount() const;
  std::size_t recordingCount() const;
  std::size_t extractorCount() const;
  std::size_t maximumRecordings() const;
  bool containsSource(const std::string& source_id) const;

 private:
  struct Source {
    std::string id;
    std::string name;
    std::string path;
    std::string type;
    std::uint64_t sequence = 0;
  };
  struct Extractor {
    std::string id;
    FeatureKind kind = FeatureKind::VideoPerceptual;
    std::uint32_t dimension = 0;
  };
  struct StoredVector {
    std::uint32_t source_index = 0;
    std::uint32_t extractor_index = 0;
    double start_seconds = 0.0;
    double end_seconds = 0.0;
    std::int64_t start_byte = -1;
    std::int64_t end_byte = -1;
    float quantized_norm = 0.0F;
    std::vector<std::int8_t> values;
  };
  struct IndexEntry {
    std::uint64_t key = 0;
    std::uint32_t record_index = 0;
  };

  void ensureIndex() const;
  std::vector<std::string> enforceMaximumRecordings();
  void compactSources();
  std::uint32_t ensureSource(const FeatureVector& vector);
  std::uint32_t ensureExtractor(const FeatureVector& vector);

  std::vector<Source> sources_;
  std::vector<Extractor> extractors_;
  std::vector<StoredVector> records_;
  std::size_t maximum_recordings_ = 0;
  std::uint64_t next_source_sequence_ = 1;
  mutable bool index_dirty_ = true;
  mutable std::vector<IndexEntry> index_;
};

}  // namespace recdup
