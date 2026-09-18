#include "recdup/vector_database.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace recdup {
namespace {

constexpr std::array<char, 8> kMagic{{'R', 'C', 'D', 'D', 'B', '0', '4', '\0'}};
constexpr std::uint32_t kVersion = 4;
constexpr std::uint32_t kEndianMarker = 0x01020304U;
constexpr std::uint32_t kMaximumString = 16U * 1024U * 1024U;
constexpr std::uint32_t kMaximumDimension = 64U * 1024U;
constexpr std::uint64_t kMaximumRecords = 100000000ULL;
constexpr unsigned kHashBits = 20;
constexpr unsigned kHashTables = 8;
constexpr unsigned kProjectionSamples = 8;

template <typename T>
void writeValue(std::ostream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!output) throw std::runtime_error("failed to write vector database");
}

template <typename T>
T readValue(std::istream& input) {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(value));
  if (!input) throw std::runtime_error("truncated vector database");
  return value;
}

void writeString(std::ostream& output, const std::string& value) {
  if (value.size() > kMaximumString)
    throw std::runtime_error("database string is too large");
  writeValue(output, static_cast<std::uint32_t>(value.size()));
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
  if (!output) throw std::runtime_error("failed to write database string");
}

std::string readString(std::istream& input) {
  const auto size = readValue<std::uint32_t>(input);
  if (size > kMaximumString)
    throw std::runtime_error("invalid database string length");
  std::string value(size, '\0');
  input.read(value.data(), static_cast<std::streamsize>(size));
  if (!input) throw std::runtime_error("truncated database string");
  return value;
}

std::uint64_t mix(std::uint64_t value) {
  value ^= value >> 30U;
  value *= 0xbf58476d1ce4e5b9ULL;
  value ^= value >> 27U;
  value *= 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

template <typename ValueAt>
std::uint32_t signature(std::size_t dimension, unsigned table,
                        ValueAt value_at) {
  std::uint32_t result = 0;
  for (unsigned bit = 0; bit < kHashBits; ++bit) {
    double projection = 0.0;
    for (unsigned sample = 0; sample < kProjectionSamples; ++sample) {
      const auto random = mix((static_cast<std::uint64_t>(table) << 56U) ^
                              (static_cast<std::uint64_t>(bit) << 32U) ^
                              sample);
      const std::size_t dimension_index = static_cast<std::size_t>(random % dimension);
      projection += (random & (1ULL << 20U) ? 1.0 : -1.0) *
                    value_at(dimension_index);
    }
    if (projection >= 0.0) result |= (1U << bit);
  }
  return result;
}

std::uint64_t indexKey(std::uint32_t extractor, unsigned table,
                       std::uint32_t hash) {
  if (extractor >= (1U << 24U))
    throw std::runtime_error("too many extractors in database");
  return (static_cast<std::uint64_t>(extractor) << 40U) |
         (static_cast<std::uint64_t>(table) << 32U) | hash;
}

bool validKind(std::uint8_t value) {
  return value == static_cast<std::uint8_t>(FeatureKind::VideoPerceptual) ||
         value == static_cast<std::uint8_t>(FeatureKind::AudioSpectrum);
}

void replaceFile(const std::filesystem::path& from,
                 const std::filesystem::path& to) {
#ifdef _WIN32
  if (!MoveFileExW(from.c_str(), to.c_str(),
                   MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
    throw std::runtime_error("failed to replace database file (Windows error " +
                             std::to_string(GetLastError()) + ")");
  }
#else
  std::filesystem::rename(from, to);
#endif
}

std::vector<float> normalized(std::vector<float> values) {
  double norm = 0.0;
  for (float value : values) norm += static_cast<double>(value) * value;
  if (norm <= 1.0e-20) throw std::runtime_error("cannot store a zero vector");
  const float scale = static_cast<float>(1.0 / std::sqrt(norm));
  for (float& value : values) value *= scale;
  return values;
}

}  // namespace

VectorDatabase VectorDatabase::create() { return {}; }

std::size_t VectorDatabase::size() const { return records_.size(); }
std::size_t VectorDatabase::sourceCount() const {
  std::vector<bool> used(sources_.size(), false);
  for (const auto& record : records_) used[record.source_index] = true;
  return static_cast<std::size_t>(std::count(used.begin(), used.end(), true));
}
std::size_t VectorDatabase::recordingCount() const {
  std::vector<bool> used(sources_.size(), false);
  for (const auto& record : records_) used[record.source_index] = true;
  std::size_t result = 0;
  for (std::size_t i = 0; i < sources_.size(); ++i)
    if (used[i] && sources_[i].type == "recording") ++result;
  return result;
}
std::size_t VectorDatabase::extractorCount() const { return extractors_.size(); }
std::size_t VectorDatabase::maximumRecordings() const {
  return maximum_recordings_;
}
bool VectorDatabase::containsSource(const std::string& source_id) const {
  std::vector<bool> used(sources_.size(), false);
  for (const auto& record : records_) used[record.source_index] = true;
  for (std::size_t i = 0; i < sources_.size(); ++i)
    if (used[i] && sources_[i].id == source_id) return true;
  return false;
}

VectorDatabase VectorDatabase::load(const std::string& path) {
  std::ifstream input(std::filesystem::u8path(path), std::ios::binary);
  if (!input) throw std::runtime_error("cannot open database: " + path);
  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != kMagic)
    throw std::runtime_error(
        "unsupported database format; recreate it with this version: " + path);
  if (readValue<std::uint32_t>(input) != kVersion)
    throw std::runtime_error("unsupported database version");
  if (readValue<std::uint32_t>(input) != kEndianMarker)
    throw std::runtime_error("database byte order is not supported");
  const auto source_count = readValue<std::uint32_t>(input);
  const auto extractor_count = readValue<std::uint32_t>(input);
  const auto record_count = readValue<std::uint64_t>(input);
  const auto maximum_recordings = readValue<std::uint64_t>(input);
  if (record_count > kMaximumRecords)
    throw std::runtime_error("unreasonable database record count");
  if (maximum_recordings > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("database recording limit is too large");

  VectorDatabase database;
  database.maximum_recordings_ = static_cast<std::size_t>(maximum_recordings);
  database.sources_.reserve(source_count);
  for (std::uint32_t i = 0; i < source_count; ++i) {
    Source source;
    source.id = readString(input);
    source.name = readString(input);
    source.path = readString(input);
    source.type = readString(input);
    source.sequence = readValue<std::uint64_t>(input);
    if (source.sequence == std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("database source sequence is exhausted");
    database.next_source_sequence_ =
        std::max(database.next_source_sequence_, source.sequence + 1);
    database.sources_.push_back(std::move(source));
  }
  database.extractors_.reserve(extractor_count);
  for (std::uint32_t i = 0; i < extractor_count; ++i) {
    Extractor extractor;
    extractor.id = readString(input);
    const auto raw_kind = readValue<std::uint8_t>(input);
    if (!validKind(raw_kind)) throw std::runtime_error("invalid database feature kind");
    extractor.kind = static_cast<FeatureKind>(raw_kind);
    extractor.dimension = readValue<std::uint32_t>(input);
    if (extractor.dimension == 0 || extractor.dimension > kMaximumDimension)
      throw std::runtime_error("invalid database vector dimension");
    database.extractors_.push_back(std::move(extractor));
  }
  database.records_.reserve(static_cast<std::size_t>(record_count));
  for (std::uint64_t i = 0; i < record_count; ++i) {
    StoredVector record;
    record.source_index = readValue<std::uint32_t>(input);
    record.extractor_index = readValue<std::uint32_t>(input);
    if (record.source_index >= database.sources_.size() ||
        record.extractor_index >= database.extractors_.size())
      throw std::runtime_error("invalid database metadata reference");
    record.start_seconds = readValue<double>(input);
    record.end_seconds = readValue<double>(input);
    record.start_byte = readValue<std::int64_t>(input);
    record.end_byte = readValue<std::int64_t>(input);
    record.quantized_norm = readValue<float>(input);
    const auto dimension = database.extractors_[record.extractor_index].dimension;
    record.values.resize(dimension);
    input.read(reinterpret_cast<char*>(record.values.data()),
               static_cast<std::streamsize>(dimension));
    if (!input || record.quantized_norm <= 0.0F)
      throw std::runtime_error("truncated or invalid vector payload");
    database.records_.push_back(std::move(record));
  }
  database.index_dirty_ = true;
  database.enforceMaximumRecordings();
  return database;
}

std::uint32_t VectorDatabase::ensureSource(const FeatureVector& vector) {
  for (std::size_t i = 0; i < sources_.size(); ++i) {
    if (sources_[i].id == vector.source_id) {
      sources_[i].name = vector.source_name;
      sources_[i].path = vector.source_path;
      sources_[i].type = vector.source_type;
      return static_cast<std::uint32_t>(i);
    }
  }
  sources_.push_back({vector.source_id, vector.source_name, vector.source_path,
                      vector.source_type, 0});
  return static_cast<std::uint32_t>(sources_.size() - 1);
}

std::uint32_t VectorDatabase::ensureExtractor(const FeatureVector& vector) {
  if (vector.extractor_id.empty()) throw std::runtime_error("extractor ID is empty");
  for (std::size_t i = 0; i < extractors_.size(); ++i) {
    if (extractors_[i].id != vector.extractor_id) continue;
    if (extractors_[i].kind != vector.kind ||
        extractors_[i].dimension != vector.values.size())
      throw std::runtime_error("extractor ID conflicts with existing vector schema: " +
                               vector.extractor_id);
    return static_cast<std::uint32_t>(i);
  }
  if (vector.values.empty() || vector.values.size() > kMaximumDimension)
    throw std::runtime_error("invalid vector dimension");
  extractors_.push_back({vector.extractor_id, vector.kind,
                         static_cast<std::uint32_t>(vector.values.size())});
  return static_cast<std::uint32_t>(extractors_.size() - 1);
}

std::vector<std::string> VectorDatabase::replaceSource(
    const std::string& source_id, std::vector<FeatureVector> vectors) {
  if (source_id.empty()) throw std::runtime_error("source ID is empty");
  for (const auto& vector : vectors)
    if (vector.source_id != source_id)
      throw std::runtime_error("source replacement contains mixed source IDs");
  std::uint32_t source_index = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t i = 0; i < sources_.size(); ++i)
    if (sources_[i].id == source_id) source_index = static_cast<std::uint32_t>(i);
  if (source_index != std::numeric_limits<std::uint32_t>::max()) {
    records_.erase(std::remove_if(records_.begin(), records_.end(),
                                  [&](const StoredVector& record) {
                                    return record.source_index == source_index;
                                  }),
                   records_.end());
  }

  if (!vectors.empty()) {
    source_index = ensureSource(vectors.front());
    if (next_source_sequence_ == std::numeric_limits<std::uint64_t>::max())
      throw std::runtime_error("database source sequence is exhausted");
    sources_[source_index].sequence = next_source_sequence_++;
  }
  for (auto& vector : vectors) {
    const auto extractor = ensureExtractor(vector);
    const auto normalized_values = normalized(std::move(vector.values));
    StoredVector stored;
    stored.source_index = source_index;
    stored.extractor_index = extractor;
    stored.start_seconds = vector.start_seconds;
    stored.end_seconds = vector.end_seconds;
    stored.start_byte = vector.start_byte;
    stored.end_byte = vector.end_byte;
    stored.values.resize(normalized_values.size());
    double norm = 0.0;
    for (std::size_t i = 0; i < normalized_values.size(); ++i) {
      const auto quantized = static_cast<int>(
          std::lround(std::max(-1.0F, std::min(1.0F, normalized_values[i])) * 127.0F));
      stored.values[i] = static_cast<std::int8_t>(quantized);
      norm += static_cast<double>(quantized) * quantized;
    }
    stored.quantized_norm = static_cast<float>(std::sqrt(norm));
    if (stored.quantized_norm <= 0.0F)
      throw std::runtime_error("vector became zero during quantization");
    records_.push_back(std::move(stored));
  }
  index_.clear();
  index_dirty_ = true;
  compactSources();
  return enforceMaximumRecordings();
}

std::vector<std::string> VectorDatabase::setMaximumRecordings(
    std::size_t maximum) {
  maximum_recordings_ = maximum;
  return enforceMaximumRecordings();
}

void VectorDatabase::compactSources() {
  std::vector<bool> used(sources_.size(), false);
  for (const auto& record : records_) used[record.source_index] = true;
  std::vector<std::uint32_t> remap(
      sources_.size(), std::numeric_limits<std::uint32_t>::max());
  std::vector<Source> compacted;
  compacted.reserve(sourceCount());
  for (std::size_t i = 0; i < sources_.size(); ++i) {
    if (!used[i]) continue;
    remap[i] = static_cast<std::uint32_t>(compacted.size());
    compacted.push_back(std::move(sources_[i]));
  }
  for (auto& record : records_) record.source_index = remap[record.source_index];
  sources_ = std::move(compacted);
}

std::vector<std::string> VectorDatabase::enforceMaximumRecordings() {
  if (maximum_recordings_ == 0) return {};
  std::vector<std::size_t> recordings;
  for (std::size_t i = 0; i < sources_.size(); ++i)
    if (sources_[i].type == "recording") recordings.push_back(i);
  if (recordings.size() <= maximum_recordings_) return {};
  std::sort(recordings.begin(), recordings.end(), [&](std::size_t a,
                                                       std::size_t b) {
    return std::tie(sources_[a].sequence, sources_[a].id) <
           std::tie(sources_[b].sequence, sources_[b].id);
  });
  const std::size_t remove_count = recordings.size() - maximum_recordings_;
  std::vector<bool> remove(sources_.size(), false);
  std::vector<std::string> removed;
  removed.reserve(remove_count);
  for (std::size_t i = 0; i < remove_count; ++i) {
    remove[recordings[i]] = true;
    removed.push_back(sources_[recordings[i]].id);
  }
  records_.erase(std::remove_if(records_.begin(), records_.end(),
                                [&](const StoredVector& record) {
                                  return remove[record.source_index];
                                }),
                 records_.end());
  compactSources();
  index_.clear();
  index_dirty_ = true;
  return removed;
}

void VectorDatabase::save(const std::string& path) const {
  const std::filesystem::path destination = std::filesystem::u8path(path);
  if (destination.has_parent_path())
    std::filesystem::create_directories(destination.parent_path());
  auto temporary = destination;
  temporary += L".tmp";
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
      throw std::runtime_error("cannot create database: " + temporary.u8string());
    output.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    writeValue(output, kVersion);
    writeValue(output, kEndianMarker);
    writeValue(output, static_cast<std::uint32_t>(sources_.size()));
    writeValue(output, static_cast<std::uint32_t>(extractors_.size()));
    writeValue(output, static_cast<std::uint64_t>(records_.size()));
    writeValue(output, static_cast<std::uint64_t>(maximum_recordings_));
    for (const auto& source : sources_) {
      writeString(output, source.id);
      writeString(output, source.name);
      writeString(output, source.path);
      writeString(output, source.type);
      writeValue(output, source.sequence);
    }
    for (const auto& extractor : extractors_) {
      writeString(output, extractor.id);
      writeValue(output, static_cast<std::uint8_t>(extractor.kind));
      writeValue(output, extractor.dimension);
    }
    for (const auto& record : records_) {
      writeValue(output, record.source_index);
      writeValue(output, record.extractor_index);
      writeValue(output, record.start_seconds);
      writeValue(output, record.end_seconds);
      writeValue(output, record.start_byte);
      writeValue(output, record.end_byte);
      writeValue(output, record.quantized_norm);
      output.write(reinterpret_cast<const char*>(record.values.data()),
                   static_cast<std::streamsize>(record.values.size()));
      if (!output) throw std::runtime_error("failed to write vector payload");
    }
    output.flush();
    if (!output) throw std::runtime_error("failed to flush database");
  }
  try {
    replaceFile(temporary, destination);
  } catch (...) {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    throw;
  }
}

void VectorDatabase::ensureIndex() const {
  if (!index_dirty_) return;
  index_.clear();
  if (records_.size() > std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("database is too large for the in-memory index");
  index_.reserve(records_.size() * kHashTables);
  for (std::size_t i = 0; i < records_.size(); ++i) {
    const auto& record = records_[i];
    for (unsigned table = 0; table < kHashTables; ++table) {
      const auto hash = signature(record.values.size(), table, [&](std::size_t d) {
        return static_cast<double>(record.values[d]);
      });
      index_.push_back({indexKey(record.extractor_index, table, hash),
                        static_cast<std::uint32_t>(i)});
    }
  }
  std::sort(index_.begin(), index_.end(), [](const IndexEntry& a,
                                              const IndexEntry& b) {
    return a.key < b.key || (a.key == b.key && a.record_index < b.record_index);
  });
  index_dirty_ = false;
}

std::vector<SearchHit> VectorDatabase::search(const FeatureVector& query,
                                              std::size_t top_k,
                                              float minimum_similarity) const {
  if (query.values.empty() || top_k == 0) return {};
  std::uint32_t extractor_index = std::numeric_limits<std::uint32_t>::max();
  for (std::size_t i = 0; i < extractors_.size(); ++i) {
    const auto& extractor = extractors_[i];
    if (extractor.id == query.extractor_id && extractor.kind == query.kind &&
        extractor.dimension == query.values.size()) {
      extractor_index = static_cast<std::uint32_t>(i);
      break;
    }
  }
  if (extractor_index == std::numeric_limits<std::uint32_t>::max()) return {};

  const auto query_values = normalized(query.values);
  ensureIndex();
  std::vector<std::uint32_t> candidates;
  auto add_bucket = [&](std::uint64_t key) {
    const auto begin = std::lower_bound(
        index_.begin(), index_.end(), key,
        [](const IndexEntry& entry, std::uint64_t value) { return entry.key < value; });
    for (auto cursor = begin; cursor != index_.end() && cursor->key == key; ++cursor)
      candidates.push_back(cursor->record_index);
  };

  if (records_.size() <= 20000) {
    for (std::size_t i = 0; i < records_.size(); ++i)
      if (records_[i].extractor_index == extractor_index)
        candidates.push_back(static_cast<std::uint32_t>(i));
  } else {
    std::array<std::uint32_t, kHashTables> bases{};
    for (unsigned table = 0; table < kHashTables; ++table) {
      bases[table] = signature(query_values.size(), table, [&](std::size_t d) {
        return static_cast<double>(query_values[d]);
      });
      add_bucket(indexKey(extractor_index, table, bases[table]));
      for (unsigned bit = 0; bit < kHashBits; ++bit)
        add_bucket(indexKey(extractor_index, table,
                           bases[table] ^ (1U << bit)));
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()),
                     candidates.end());
    if (candidates.size() < std::max<std::size_t>(64, top_k * 8)) {
      for (unsigned table = 0; table < kHashTables; ++table) {
        for (unsigned a = 0; a < kHashBits; ++a)
          for (unsigned b = a + 1; b < kHashBits; ++b)
            add_bucket(indexKey(extractor_index, table,
                                bases[table] ^ (1U << a) ^ (1U << b)));
      }
    }
  }
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());

  struct Scored { std::uint32_t index; float score; };
  std::vector<Scored> scored;
  for (std::uint32_t index : candidates) {
    const auto& record = records_[index];
    if (record.extractor_index != extractor_index) continue;
    double dot = 0.0;
    for (std::size_t d = 0; d < query_values.size(); ++d)
      dot += query_values[d] * record.values[d];
    const float score = static_cast<float>(dot / record.quantized_norm);
    if (score >= minimum_similarity) scored.push_back({index, score});
  }
  std::sort(scored.begin(), scored.end(), [](const Scored& a, const Scored& b) {
    return a.score > b.score;
  });
  if (scored.size() > top_k) scored.resize(top_k);

  std::vector<SearchHit> result;
  result.reserve(scored.size());
  for (const auto& item : scored) {
    const auto& record = records_[item.index];
    const auto& source = sources_[record.source_index];
    const auto& extractor = extractors_[record.extractor_index];
    SearchHit hit;
    hit.similarity = item.score;
    hit.source_id = source.id;
    hit.source_name = source.name;
    hit.source_path = source.path;
    hit.source_type = source.type;
    hit.extractor_id = extractor.id;
    hit.kind = extractor.kind;
    hit.start_seconds = record.start_seconds;
    hit.end_seconds = record.end_seconds;
    hit.start_byte = record.start_byte;
    hit.end_byte = record.end_byte;
    result.push_back(std::move(hit));
  }
  return result;
}

}  // namespace recdup
