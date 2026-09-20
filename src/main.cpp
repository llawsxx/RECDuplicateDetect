#include "recdup/detector.hpp"
#include "recdup/json_writer.hpp"
#include "recdup/media_analyzer.hpp"
#include "recdup/programme_inference.hpp"
#include "recdup/vector_database.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#endif

namespace {

#ifdef _WIN32
std::string wideToUtf8(const wchar_t* value) {
  if (!value) return {};
  const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                                       nullptr, 0, nullptr, nullptr);
  if (size <= 0) throw std::runtime_error("failed to decode Windows command line");
  std::string result(static_cast<std::size_t>(size), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                          result.data(), size, nullptr, nullptr) <= 0)
    throw std::runtime_error("failed to decode Windows command line");
  result.resize(static_cast<std::size_t>(size - 1));
  return result;
}

std::vector<std::string> unicodeArguments() {
  int count = 0;
  wchar_t** values = CommandLineToArgvW(GetCommandLineW(), &count);
  if (!values) throw std::runtime_error("failed to read Windows command line");
  std::vector<std::string> result;
  result.reserve(static_cast<std::size_t>(count));
  try {
    for (int i = 0; i < count; ++i) result.push_back(wideToUtf8(values[i]));
  } catch (...) {
    LocalFree(values);
    throw;
  }
  LocalFree(values);
  return result;
}
#endif

struct Arguments {
  std::string command;
  std::unordered_map<std::string, std::string> values;
  std::unordered_set<std::string> flags;
};

void usage(std::ostream& output) {
  output <<
      "recdup - duplicate broadcast segment detector\n\n"
      "Usage:\n"
      "  recdup init         --db FILE\n"
      "  recdup info         --db FILE\n"
      "  recdup configure    --db FILE --max-recordings N\n"
      "  recdup ingest       --db FILE --input MEDIA [options]\n"
      "  recdup scan         --input MEDIA [--db FILE] [options]\n\n"
      "Database options:\n"
      "  --max-recordings N     Keep newest N recordings; 0 = unlimited\n\n"
      "Media options:\n"
      "  --name TEXT             Human-readable source name\n"
      "  --id TEXT               Stable source ID\n"
      "  --content-type TYPE     advertisement|programme|break_out|break_in|recording|unknown\n"
      "  --mode auto|video|audio|both (default: auto)\n"
      "  --audio-hop SEC         Audio feature step, <= 1 (default: 0.5)\n"
      "  --timestamp-jump SEC    Repair jumps larger than this (default: 10)\n"
      "  --no-timestamp-repair   Keep original discontinuous timestamps\n"
      "  --no-progress           Disable analysis progress output\n\n"
      "Detection options:\n"
      "  --min-duration SEC      Minimum aligned repeat (default: 5)\n"
      "  --threshold VALUE       Cosine threshold (default: 0.90)\n"
      "  --top-k COUNT           Neighbors per base vector (default: 12)\n"
      "  --max-gap SEC           Allowed gap between anchors (default: 2.5)\n"
      "  --offset-bin SEC        Alignment offset resolution (default: 1)\n"
      "  --programme-min SEC     Minimum programme guess (default: 120)\n"
      "  --short-repeat-max SEC  Maximum short repeat (default: 180)\n"
      "  --ad-block-gap SEC      Join short repeats into a break (default: 20)\n"
      "  --ad-break-min SEC      Minimum inferred ad break (default: 10)\n"
      "  --ad-min-occurrences N  Minimum appearances per auto ad (default: 2)\n"
      "  --ad-min-families N     Minimum families per auto break (default: 2)\n"
      "  --programme-audio-threshold VALUE (default: 0.93)\n"
      "  --programme-video-threshold VALUE (default: 0.985)\n"
      "  --programme-confirm-margin VALUE (default: 0.02)\n"
      "  --marker-max SEC        Maximum auto marker length (default: 30)\n"
      "  --marker-min-occurrences N (default: 3)\n"
      "  --store                 Store vectors after scanning\n"
      "  --no-self               Disable repeats within the input\n"
      "  --no-programme-inference Disable programme timeline guesses\n"
      "  --output FILE           Write JSON to FILE (default: stdout)\n";
}

Arguments parseArguments(int argc, char** argv) {
  if (argc < 2) throw std::runtime_error("missing command");
  Arguments result;
  result.command = argv[1];
  const std::unordered_set<std::string> boolean_flags{
      "--store", "--no-self", "--no-progress",
      "--no-programme-inference", "--no-timestamp-repair", "--help"};
  for (int i = 2; i < argc; ++i) {
    const std::string key = argv[i];
    if (key.rfind("--", 0) != 0)
      throw std::runtime_error("unexpected argument: " + key);
    if (boolean_flags.count(key)) {
      result.flags.insert(key);
    } else {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
      result.values[key] = argv[++i];
    }
  }
  return result;
}

std::string require(const Arguments& arguments, const std::string& key) {
  const auto found = arguments.values.find(key);
  if (found == arguments.values.end() || found->second.empty())
    throw std::runtime_error("required option is missing: " + key);
  return found->second;
}

std::string optional(const Arguments& arguments, const std::string& key,
                     const std::string& fallback) {
  const auto found = arguments.values.find(key);
  return found == arguments.values.end() ? fallback : found->second;
}

std::string jsonQuote(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (unsigned char ch : value) {
    if (ch == '"' || ch == '\\') output << '\\' << static_cast<char>(ch);
    else if (ch == '\n') output << "\\n";
    else if (ch == '\r') output << "\\r";
    else if (ch == '\t') output << "\\t";
    else if (ch >= 0x20) output << static_cast<char>(ch);
  }
  output << '"';
  return output.str();
}

template <typename T>
T parseNumber(const std::string& value, const char* name) {
  std::istringstream input(value);
  T number{};
  input >> number;
  if (!input || !input.eof())
    throw std::runtime_error(std::string("invalid ") + name + ": " + value);
  return number;
}

recdup::AnalysisMode parseMode(const std::string& value) {
  if (value == "auto") return recdup::AnalysisMode::Auto;
  if (value == "video") return recdup::AnalysisMode::Video;
  if (value == "audio") return recdup::AnalysisMode::Audio;
  if (value == "both") return recdup::AnalysisMode::Both;
  throw std::runtime_error("invalid --mode: " + value);
}

std::string contentType(const Arguments& arguments) {
  const std::string fallback =
      arguments.command == "scan" ? "recording" : "unknown";
  const std::string value = optional(arguments, "--content-type", fallback);
  if (value == "advertisement" || value == "programme" ||
      value == "break_out" || value == "break_in" ||
      value == "recording" || value == "unknown")
    return value;
  throw std::runtime_error(
      "invalid --content-type: " + value +
      " (expected advertisement, programme, break_out, break_in, recording, "
      "or unknown)");
}

std::optional<std::size_t> maximumRecordingsArgument(
    const Arguments& arguments) {
  const auto found = arguments.values.find("--max-recordings");
  if (found == arguments.values.end()) return std::nullopt;
  if (found->second.empty() || found->second.front() == '-')
    throw std::runtime_error("--max-recordings must be zero or positive");
  const auto value = parseNumber<std::uint64_t>(found->second,
                                                "max-recordings");
  if (value > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("--max-recordings is too large");
  return static_cast<std::size_t>(value);
}

std::uint64_t fnv1a(std::uint64_t hash, const std::string& value) {
  for (unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string generatedId(const std::string& input_path) {
  const auto absolute = std::filesystem::absolute(std::filesystem::u8path(input_path));
  std::error_code error;
  const auto size = std::filesystem::file_size(absolute, error);
  std::uint64_t hash = fnv1a(14695981039346656037ULL, absolute.u8string());
  if (!error) hash = fnv1a(hash, std::to_string(size));
  std::ostringstream output;
  output << std::hex << hash;
  return output.str();
}

recdup::SearchOptions searchOptions(const Arguments& arguments) {
  recdup::SearchOptions options;
  options.minimum_similarity = parseNumber<float>(
      optional(arguments, "--threshold", "0.90"), "threshold");
  options.top_k = parseNumber<std::size_t>(
      optional(arguments, "--top-k", "12"), "top-k");
  options.minimum_duration_seconds = parseNumber<double>(
      optional(arguments, "--min-duration", "5"), "min-duration");
  options.maximum_anchor_gap_seconds = parseNumber<double>(
      optional(arguments, "--max-gap", "2.5"), "max-gap");
  options.offset_bin_seconds = parseNumber<double>(
      optional(arguments, "--offset-bin", "1"), "offset-bin");
  options.find_internal_duplicates = !arguments.flags.count("--no-self");
  if (options.minimum_similarity < -1.0F || options.minimum_similarity > 1.0F)
    throw std::runtime_error("--threshold must be between -1 and 1");
  if (options.top_k == 0) throw std::runtime_error("--top-k must be positive");
  if (options.minimum_duration_seconds <= 0.0 ||
      options.maximum_anchor_gap_seconds <= 0.0 ||
      options.offset_bin_seconds <= 0.0)
    throw std::runtime_error("duration and alignment options must be positive");
  return options;
}

recdup::ProgrammeInferenceOptions inferenceOptions(
    const Arguments& arguments) {
  recdup::ProgrammeInferenceOptions options;
  options.minimum_programme_seconds = parseNumber<double>(
      optional(arguments, "--programme-min", "120"), "programme-min");
  options.maximum_short_repeat_seconds = parseNumber<double>(
      optional(arguments, "--short-repeat-max", "180"), "short-repeat-max");
  options.ad_block_gap_seconds = parseNumber<double>(
      optional(arguments, "--ad-block-gap", "20"), "ad-block-gap");
  options.minimum_ad_break_seconds = parseNumber<double>(
      optional(arguments, "--ad-break-min", "10"), "ad-break-min");
  const std::string ad_occurrences =
      optional(arguments, "--ad-min-occurrences", "2");
  if (ad_occurrences.empty() || ad_occurrences.front() == '-')
    throw std::runtime_error("--ad-min-occurrences must be at least 2");
  options.minimum_ad_occurrences = parseNumber<std::size_t>(
      ad_occurrences, "ad-min-occurrences");
  const std::string ad_families =
      optional(arguments, "--ad-min-families", "2");
  if (ad_families.empty() || ad_families.front() == '-')
    throw std::runtime_error("--ad-min-families must be at least 1");
  options.minimum_ad_families =
      parseNumber<std::size_t>(ad_families, "ad-min-families");
  options.minimum_audio_similarity = parseNumber<double>(
      optional(arguments, "--programme-audio-threshold", "0.93"),
      "programme-audio-threshold");
  options.minimum_video_similarity = parseNumber<double>(
      optional(arguments, "--programme-video-threshold", "0.985"),
      "programme-video-threshold");
  options.audio_video_confirmation_margin = parseNumber<double>(
      optional(arguments, "--programme-confirm-margin", "0.02"),
      "programme-confirm-margin");
  options.maximum_marker_seconds = parseNumber<double>(
      optional(arguments, "--marker-max", "30"), "marker-max");
  const std::string marker_occurrences =
      optional(arguments, "--marker-min-occurrences", "3");
  if (marker_occurrences.empty() || marker_occurrences.front() == '-')
    throw std::runtime_error(
        "--marker-min-occurrences must be at least 2");
  options.minimum_marker_occurrences = parseNumber<std::size_t>(
      marker_occurrences, "marker-min-occurrences");
  if (options.minimum_programme_seconds <= 0.0 ||
      options.maximum_short_repeat_seconds <= 0.0 ||
      options.ad_block_gap_seconds < 0.0)
    throw std::runtime_error("programme inference durations are invalid");
  if (!std::isfinite(options.minimum_ad_break_seconds) ||
      options.minimum_ad_break_seconds < 0.0)
    throw std::runtime_error("--ad-break-min must be zero or positive");
  if (options.minimum_ad_occurrences < 2)
    throw std::runtime_error("--ad-min-occurrences must be at least 2");
  if (options.minimum_ad_families < 1)
    throw std::runtime_error("--ad-min-families must be at least 1");
  if (!std::isfinite(options.minimum_audio_similarity) ||
      options.minimum_audio_similarity < 0.0 ||
      options.minimum_audio_similarity > 1.0)
    throw std::runtime_error(
        "--programme-audio-threshold must be between 0 and 1");
  if (!std::isfinite(options.minimum_video_similarity) ||
      options.minimum_video_similarity < 0.0 ||
      options.minimum_video_similarity > 1.0)
    throw std::runtime_error(
        "--programme-video-threshold must be between 0 and 1");
  if (!std::isfinite(options.audio_video_confirmation_margin) ||
      options.audio_video_confirmation_margin < 0.0 ||
      options.audio_video_confirmation_margin > 1.0)
    throw std::runtime_error(
        "--programme-confirm-margin must be between 0 and 1");
  if (!std::isfinite(options.maximum_marker_seconds) ||
      options.maximum_marker_seconds <= 0.0)
    throw std::runtime_error("--marker-max must be positive");
  if (options.minimum_marker_occurrences < 2)
    throw std::runtime_error(
        "--marker-min-occurrences must be at least 2");
  return options;
}

struct AnalyzedInput {
  recdup::MediaFeatures media;
  std::vector<recdup::FeatureVector> vectors;
};

std::string clockText(double seconds) {
  if (!std::isfinite(seconds) || seconds < 0.0) return "--:--:--";
  const auto whole = static_cast<std::uint64_t>(seconds);
  const auto hours = whole / 3600U;
  const auto minutes = (whole % 3600U) / 60U;
  const auto secs = whole % 60U;
  std::ostringstream output;
  output << std::setfill('0') << std::setw(2) << hours << ':'
         << std::setw(2) << minutes << ':' << std::setw(2) << secs;
  return output.str();
}

class ProgressPrinter {
 public:
  void update(double processed_seconds, double total_seconds) {
    processed_seconds_ = processed_seconds;
    total_seconds_ = total_seconds;
    has_progress_ = true;
    const auto now = std::chrono::steady_clock::now();
    if (now - last_print_ < std::chrono::seconds(1)) return;
    print(now);
  }

  void finish() {
    if (!has_progress_) return;
    print(std::chrono::steady_clock::now());
    std::cerr << '\n';
    has_progress_ = false;
  }

 private:
  void print(std::chrono::steady_clock::time_point now) {
    const double elapsed =
        std::chrono::duration<double>(now - started_).count();
    const double speed = elapsed > 0.0 ? processed_seconds_ / elapsed : 0.0;
    std::ostringstream line;
    line << '\r' << "Progress: ";
    if (total_seconds_ > 0.0) {
      const double percent = std::clamp(
          processed_seconds_ * 100.0 / total_seconds_, 0.0, 100.0);
      line << std::fixed << std::setprecision(1) << std::setw(5) << percent
           << "%  " << clockText(processed_seconds_) << " / "
           << clockText(total_seconds_);
    } else {
      line << clockText(processed_seconds_);
    }
    line << "  " << std::fixed << std::setprecision(2) << speed << "x";
    if (total_seconds_ > processed_seconds_ && speed > 0.0)
      line << "  ETA "
           << clockText((total_seconds_ - processed_seconds_) / speed);
    line << "        ";
    std::cerr << line.str() << std::flush;
    last_print_ = now;
  }

  std::chrono::steady_clock::time_point started_ =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last_print_ =
      started_ - std::chrono::seconds(1);
  double processed_seconds_ = 0.0;
  double total_seconds_ = 0.0;
  bool has_progress_ = false;
};

AnalyzedInput analyze(const Arguments& arguments) {
  const std::string input = require(arguments, "--input");
  const auto input_fs_path = std::filesystem::u8path(input);
  if (!std::filesystem::is_regular_file(input_fs_path))
    throw std::runtime_error("input is not a regular file: " + input);
  const std::string source_id = optional(arguments, "--id", generatedId(input));
  const std::string name = optional(
      arguments, "--name", input_fs_path.filename().u8string());

  recdup::AnalyzerOptions analyzer_options;
  analyzer_options.mode = parseMode(optional(arguments, "--mode", "auto"));
  analyzer_options.audio_hop_seconds = parseNumber<double>(
      optional(arguments, "--audio-hop", "0.5"), "audio-hop");
  if (!std::isfinite(analyzer_options.audio_hop_seconds) ||
      analyzer_options.audio_hop_seconds <= 0.0 ||
      analyzer_options.audio_hop_seconds > analyzer_options.bucket_seconds)
    throw std::runtime_error("--audio-hop must be greater than 0 and at most 1");
  analyzer_options.repair_timestamp_discontinuities =
      !arguments.flags.count("--no-timestamp-repair");
  analyzer_options.timestamp_jump_threshold_seconds = parseNumber<double>(
      optional(arguments, "--timestamp-jump", "10"), "timestamp-jump");
  if (analyzer_options.timestamp_jump_threshold_seconds <= 0.0)
    throw std::runtime_error("--timestamp-jump must be positive");

  ProgressPrinter progress;
  if (!arguments.flags.count("--no-progress")) {
    analyzer_options.progress_callback = [&](double processed, double total) {
      progress.update(processed, total);
    };
  }
  std::cerr << "Analyzing " << input << "...\n";
  recdup::MediaAnalyzer analyzer(analyzer_options);
  recdup::MediaFeatures media;
  try {
    media = analyzer.analyze(input, source_id);
  } catch (...) {
    progress.finish();
    throw;
  }
  progress.finish();
  media.source_type = contentType(arguments);
  if (!media.timestamp_discontinuities.empty())
    std::cerr << "Repaired " << media.timestamp_discontinuities.size()
              << " timestamp discontinuities.\n";
  if (media.recoverable_decode_errors != 0 || media.dropped_packets != 0)
    std::cerr << "Skipped " << media.dropped_packets
              << " damaged packets ("
              << media.recoverable_decode_errors
              << " recoverable FFmpeg errors).\n";
  auto vectors = recdup::buildBaseVectors(media, name);
  std::cerr << "Extracted " << vectors.size() << " base vectors.\n";
  return {std::move(media), std::move(vectors)};
}

recdup::VectorDatabase loadOrCreate(const std::string& path) {
  return std::filesystem::exists(std::filesystem::u8path(path))
                                      ? recdup::VectorDatabase::load(path)
                                      : recdup::VectorDatabase::create();
}

void reportEvictions(const std::vector<std::string>& source_ids) {
  if (source_ids.empty()) return;
  std::cerr << "Evicted " << source_ids.size()
            << " oldest recording source"
            << (source_ids.size() == 1 ? "" : "s") << ".\n";
}

std::vector<std::string> applyRecordingLimit(
    const Arguments& arguments, recdup::VectorDatabase& database) {
  const auto limit = maximumRecordingsArgument(arguments);
  return limit ? database.setMaximumRecordings(*limit)
               : std::vector<std::string>{};
}

void writeOutput(const Arguments& arguments, const std::string& json) {
  const auto output = arguments.values.find("--output");
  if (output == arguments.values.end()) {
    std::cout << json;
    return;
  }
  std::ofstream file(output->second, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot create output: " + output->second);
  file << json;
  if (!file) throw std::runtime_error("failed to write output: " + output->second);
}

int run(const Arguments& arguments) {
  if (arguments.command == "help" || arguments.flags.count("--help")) {
    usage(std::cout);
    return 0;
  }
  if (arguments.values.count("--max-recordings") &&
      arguments.command != "init" && arguments.command != "configure" &&
      arguments.command != "ingest" &&
      !(arguments.command == "scan" && arguments.flags.count("--store")))
    throw std::runtime_error(
        "--max-recordings requires init, ingest, or scan --store");
  if (arguments.command == "init") {
    const std::string path = require(arguments, "--db");
    if (std::filesystem::exists(std::filesystem::u8path(path)))
      throw std::runtime_error("database already exists: " + path);
    auto database = recdup::VectorDatabase::create();
    reportEvictions(applyRecordingLimit(arguments, database));
    database.save(path);
    std::cout << "Created database: " << path << '\n';
    return 0;
  }
  if (arguments.command == "info") {
    const std::string path = require(arguments, "--db");
    auto database = recdup::VectorDatabase::load(path);
    std::cout << "{\"schema_version\":4,\"path\":" << jsonQuote(path)
              << ",\"vectors\":" << database.size()
              << ",\"sources\":" << database.sourceCount()
              << ",\"recordings\":" << database.recordingCount()
              << ",\"max_recordings\":" << database.maximumRecordings()
              << ",\"extractors\":" << database.extractorCount() << "}\n";
    return 0;
  }
  if (arguments.command == "configure") {
    const std::string path = require(arguments, "--db");
    if (!arguments.values.count("--max-recordings"))
      throw std::runtime_error(
          "configure requires --max-recordings");
    auto database = recdup::VectorDatabase::load(path);
    const auto evicted = applyRecordingLimit(arguments, database);
    database.save(path);
    reportEvictions(evicted);
    std::cout << "Configured database for " << database.maximumRecordings()
              << " recordings (" << database.recordingCount()
              << " currently stored).\n";
    return 0;
  }
  if (arguments.command == "ingest") {
    const std::string path = require(arguments, "--db");
    auto analyzed = analyze(arguments);
    auto database = loadOrCreate(path);
    auto evicted = applyRecordingLimit(arguments, database);
    auto more_evicted = database.replaceSource(
        analyzed.media.source_id, std::move(analyzed.vectors));
    evicted.insert(evicted.end(), more_evicted.begin(), more_evicted.end());
    database.save(path);
    reportEvictions(evicted);
    std::cout << "Stored source " << analyzed.media.source_id << " ("
              << database.size() << " vectors total).\n";
    return 0;
  }
  if (arguments.command == "scan") {
    const auto search_options = searchOptions(arguments);
    std::optional<recdup::ProgrammeInferenceOptions> inference_options;
    if (!arguments.flags.count("--no-programme-inference"))
      inference_options.emplace(inferenceOptions(arguments));

    auto analyzed = analyze(arguments);
    std::optional<recdup::VectorDatabase> database;
    std::size_t database_vectors_before = 0;
    const auto db_argument = arguments.values.find("--db");
    if (db_argument != arguments.values.end()) {
      if (std::filesystem::exists(std::filesystem::u8path(db_argument->second)))
        database.emplace(recdup::VectorDatabase::load(db_argument->second));
      else if (!arguments.flags.count("--store"))
        throw std::runtime_error("database does not exist: " + db_argument->second);
    }
    if (database) database_vectors_before = database->size();
    auto matches = recdup::detectDuplicates(
        analyzed.vectors, database ? &*database : nullptr,
        search_options);
    std::optional<recdup::ProgrammeInference> inference;
    if (inference_options) {
      inference.emplace(recdup::inferProgrammeTimeline(
          analyzed.media, matches, *inference_options));
    }

    bool stored = false;
    if (arguments.flags.count("--store")) {
      if (db_argument == arguments.values.end())
        throw std::runtime_error("--store requires --db");
      if (!database) database.emplace(recdup::VectorDatabase::create());
      auto evicted = applyRecordingLimit(arguments, *database);
      auto more_evicted = database->replaceSource(
          analyzed.media.source_id, std::move(analyzed.vectors));
      evicted.insert(evicted.end(), more_evicted.begin(), more_evicted.end());
      database->save(db_argument->second);
      reportEvictions(evicted);
      stored = true;
    }
    writeOutput(arguments, recdup::makeResultJson(
        analyzed.media, matches, database_vectors_before, stored,
        database ? database->size() : database_vectors_before,
        inference ? &*inference : nullptr));
    std::cerr << "Found " << matches.size() << " aligned duplicate spans.\n";
    if (inference)
      std::cerr << "Guessed " << inference->programme_guesses.size()
                << " programme spans from "
                << inference->content_families.size()
                << " repeat families.\n";
    return 0;
  }
  throw std::runtime_error("unknown command: " + arguments.command);
}

}  // namespace

int main(int argc, char** argv) {
  try {
#ifdef _WIN32
    auto utf8_arguments = unicodeArguments();
    std::vector<char*> utf8_argv;
    utf8_argv.reserve(utf8_arguments.size());
    for (auto& argument : utf8_arguments) utf8_argv.push_back(argument.data());
    argc = static_cast<int>(utf8_argv.size());
    argv = utf8_argv.data();
#endif
    if (argc == 1) {
      usage(std::cout);
      return 0;
    }
    return run(parseArguments(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "recdup: " << error.what() << '\n';
    return 1;
  }
}
