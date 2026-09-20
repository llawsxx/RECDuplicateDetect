#include "recdup/media_analyzer.hpp"
#include "recdup/timestamp_normalizer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#if RECDUP_HAS_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/imgutils.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}
#endif

namespace recdup {

MediaAnalyzer::MediaAnalyzer(AnalyzerOptions options) : options_(options) {
  if (options_.bucket_seconds <= 0.0)
    throw std::invalid_argument("bucket duration must be positive");
  if (!std::isfinite(options_.audio_hop_seconds) ||
      options_.audio_hop_seconds <= 0.0 ||
      options_.audio_hop_seconds > options_.bucket_seconds)
    throw std::invalid_argument(
        "audio feature step must be positive and no longer than the window");
  if (options_.timestamp_jump_threshold_seconds <= 0.0 ||
      options_.timestamp_backwards_tolerance_seconds < 0.0)
    throw std::invalid_argument("timestamp repair thresholds are invalid");
}

#if !RECDUP_HAS_FFMPEG

MediaFeatures MediaAnalyzer::analyze(const std::string&,
                                     const std::string&) const {
  throw std::runtime_error(
      "this build has no FFmpeg support; configure with -DFFMPEG_ROOT=<path>");
}

#else
namespace {

constexpr int kVideoWidth = 32;
constexpr int kVideoHeight = 18;
constexpr int kAudioRate = 8000;
constexpr std::size_t kAudioFft = 512;

std::string ffError(int code) {
  std::array<char, AV_ERROR_MAX_STRING_SIZE> buffer{};
  av_strerror(code, buffer.data(), buffer.size());
  return buffer.data();
}

void check(int code, const char* operation) {
  if (code < 0)
    throw std::runtime_error(std::string(operation) + ": " + ffError(code));
}

bool recoverableDecodeError(int code) {
  return code == AVERROR_INVALIDDATA || code == AVERROR(EAGAIN);
}

struct FormatDeleter {
  void operator()(AVFormatContext* value) const {
    if (value) avformat_close_input(&value);
  }
};
struct CodecDeleter {
  void operator()(AVCodecContext* value) const { avcodec_free_context(&value); }
};
struct FrameDeleter {
  void operator()(AVFrame* value) const { av_frame_free(&value); }
};
struct PacketDeleter {
  void operator()(AVPacket* value) const { av_packet_free(&value); }
};
struct SwsDeleter {
  void operator()(SwsContext* value) const { sws_freeContext(value); }
};
struct SwrDeleter {
  void operator()(SwrContext* value) const { swr_free(&value); }
};

using FormatPtr = std::unique_ptr<AVFormatContext, FormatDeleter>;
using CodecPtr = std::unique_ptr<AVCodecContext, CodecDeleter>;
using FramePtr = std::unique_ptr<AVFrame, FrameDeleter>;
using PacketPtr = std::unique_ptr<AVPacket, PacketDeleter>;
using SwsPtr = std::unique_ptr<SwsContext, SwsDeleter>;
using SwrPtr = std::unique_ptr<SwrContext, SwrDeleter>;

struct Accumulator {
  std::vector<float> video_sum = std::vector<float>(128, 0.0F);
  std::vector<float> audio_samples;
  std::vector<float> audio;
  double video_weight = 0.0;
  unsigned video_samples = 0;
  std::int64_t start_byte = std::numeric_limits<std::int64_t>::max();
  std::int64_t end_byte = -1;
};

void normalize(std::vector<float>& values) {
  double norm = 0.0;
  for (float value : values) norm += static_cast<double>(value) * value;
  if (norm <= 1.0e-20) return;
  const float scale = static_cast<float>(1.0 / std::sqrt(norm));
  for (float& value : values) value *= scale;
}

void updateBytes(Accumulator& bucket, std::int64_t position, int size) {
  if (position < 0 || size <= 0) return;
  bucket.start_byte = std::min(bucket.start_byte, position);
  bucket.end_byte = std::max(bucket.end_byte, position + size - 1);
}

std::vector<float> videoFeature(const std::uint8_t* gray, int stride) {
  std::vector<float> feature(128, 0.0F);
  double mean = 0.0;
  for (int y = 0; y < kVideoHeight; ++y)
    for (int x = 0; x < kVideoWidth; ++x) mean += gray[y * stride + x];
  mean /= kVideoWidth * kVideoHeight;

  constexpr double pi = 3.14159265358979323846;
  for (int v = 0; v < 8; ++v) {
    for (int u = 0; u < 8; ++u) {
      double coefficient = 0.0;
      for (int y = 0; y < kVideoHeight; ++y) {
        const double cy = std::cos(pi * (2 * y + 1) * v /
                                   (2.0 * kVideoHeight));
        for (int x = 0; x < kVideoWidth; ++x) {
          const double cx = std::cos(pi * (2 * x + 1) * u /
                                     (2.0 * kVideoWidth));
          coefficient += (gray[y * stride + x] - mean) * cx * cy;
        }
      }
      feature[v * 8 + u] = static_cast<float>(
          coefficient / (255.0 * kVideoWidth * kVideoHeight));
    }
  }
  feature[0] = static_cast<float>(mean / 255.0);

  for (int cell_y = 0; cell_y < 4; ++cell_y) {
    const int y0 = cell_y * kVideoHeight / 4;
    const int y1 = (cell_y + 1) * kVideoHeight / 4;
    for (int cell_x = 0; cell_x < 8; ++cell_x) {
      const int x0 = cell_x * kVideoWidth / 8;
      const int x1 = (cell_x + 1) * kVideoWidth / 8;
      double sum = 0.0;
      for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) sum += gray[y * stride + x];
      const double cell_mean = sum / ((y1 - y0) * (x1 - x0));
      feature[64 + cell_y * 8 + cell_x] =
          static_cast<float>((cell_mean - mean) / 255.0);
    }
  }
  for (int y = 0; y < kVideoHeight; ++y) {
    for (int x = 0; x < kVideoWidth; ++x) {
      const unsigned bin = gray[y * stride + x] >> 4U;
      feature[96 + bin] += 1.0F / (kVideoWidth * kVideoHeight);
    }
  }

  double gradient_total = 0.0;
  for (int y = 1; y + 1 < kVideoHeight; ++y) {
    for (int x = 1; x + 1 < kVideoWidth; ++x) {
      const double gx = static_cast<double>(gray[y * stride + x + 1]) -
                        gray[y * stride + x - 1];
      const double gy = static_cast<double>(gray[(y + 1) * stride + x]) -
                        gray[(y - 1) * stride + x];
      const double ax = std::abs(gx);
      const double ay = std::abs(gy);
      const unsigned direction = ax > 2.0 * ay ? 0U
                                 : ay > 2.0 * ax ? 1U
                                 : gx * gy >= 0.0 ? 2U : 3U;
      const unsigned quadrant =
          static_cast<unsigned>(y >= kVideoHeight / 2) * 2U +
          static_cast<unsigned>(x >= kVideoWidth / 2);
      const float magnitude = static_cast<float>(std::sqrt(gx * gx + gy * gy));
      feature[112 + quadrant * 4U + direction] += magnitude;
      gradient_total += magnitude;
    }
  }
  if (gradient_total > 0.0) {
    for (std::size_t i = 112; i < 128; ++i)
      feature[i] /= static_cast<float>(gradient_total);
  }
  normalize(feature);
  return feature;
}

void fft(std::array<std::complex<double>, kAudioFft>& values) {
  for (std::size_t i = 1, j = 0; i < kAudioFft; ++i) {
    std::size_t bit = kAudioFft >> 1U;
    for (; j & bit; bit >>= 1U) j ^= bit;
    j ^= bit;
    if (i < j) std::swap(values[i], values[j]);
  }
  constexpr double pi = 3.14159265358979323846;
  for (std::size_t length = 2; length <= kAudioFft; length <<= 1U) {
    const auto root = std::polar(1.0, -2.0 * pi / static_cast<double>(length));
    for (std::size_t start = 0; start < kAudioFft; start += length) {
      std::complex<double> weight(1.0, 0.0);
      for (std::size_t offset = 0; offset < length / 2; ++offset) {
        const auto even = values[start + offset];
        const auto odd = values[start + offset + length / 2] * weight;
        values[start + offset] = even + odd;
        values[start + offset + length / 2] = even - odd;
        weight *= root;
      }
    }
  }
}

std::vector<float> audioFeature(const float* samples, int count) {
  std::vector<float> feature(128, 0.0F);
  if (count <= 0) return feature;
  double square_sum = 0.0;
  double absolute_sum = 0.0;
  double peak = 0.0;
  int crossings = 0;
  for (int i = 0; i < count; ++i) {
    const double value = samples[i];
    square_sum += value * value;
    absolute_sum += std::abs(value);
    peak = std::max(peak, std::abs(value));
    if (i > 0 && ((samples[i - 1] < 0.0F) != (samples[i] < 0.0F))) ++crossings;
  }

  std::array<double, 24> band_sum{};
  std::array<double, 24> band_square_sum{};
  std::array<double, 24> band_delta_sum{};
  std::array<double, 24> raw_band_sum{};
  std::array<double, 24> previous{};
  std::array<std::array<double, 12>, 4> quarter_sum{};
  std::array<std::size_t, 4> quarter_count{};
  std::size_t windows = 0;
  const int hop = static_cast<int>(kAudioFft / 2);
  for (int start = 0; start < count; start += hop) {
    std::array<std::complex<double>, kAudioFft> spectrum{};
    constexpr double pi = 3.14159265358979323846;
    for (std::size_t i = 0; i < kAudioFft; ++i) {
      const int position = start + static_cast<int>(i);
      const double sample = position < count ? samples[position] : 0.0;
      const double window = 0.5 - 0.5 * std::cos(2.0 * pi * i / (kAudioFft - 1));
      spectrum[i] = sample * window;
    }
    fft(spectrum);
    std::array<double, 24> window_bands{};
    for (std::size_t bin = 1; bin < kAudioFft / 2; ++bin) {
      const double frequency = static_cast<double>(bin) * kAudioRate / kAudioFft;
      if (frequency < 60.0 || frequency > 3800.0) continue;
      const double ratio = std::log(frequency / 60.0) / std::log(3800.0 / 60.0);
      const auto band = std::min<std::size_t>(23, static_cast<std::size_t>(ratio * 24.0));
      window_bands[band] += std::norm(spectrum[bin]);
    }
    const auto quarter = std::min<std::size_t>(
        3, static_cast<std::size_t>((start + hop) * 4LL /
                                    std::max(1, count)));
    for (std::size_t band = 0; band < window_bands.size(); ++band) {
      raw_band_sum[band] += window_bands[band];
      const double value = std::log1p(window_bands[band]);
      band_sum[band] += value;
      band_square_sum[band] += value * value;
      if (windows > 0) band_delta_sum[band] += std::abs(value - previous[band]);
      previous[band] = value;
      quarter_sum[quarter][band / 2] += value;
    }
    ++quarter_count[quarter];
    ++windows;
    if (start + static_cast<int>(kAudioFft) >= count) break;
  }

  double band_total = 0.0;
  double weighted_band = 0.0;
  double log_sum = 0.0;
  const double window_count = static_cast<double>(std::max<std::size_t>(1, windows));
  const double delta_count = static_cast<double>(std::max<std::size_t>(1, windows - 1));
  for (std::size_t i = 0; i < band_sum.size(); ++i) {
    const double mean = band_sum[i] / window_count;
    const double variance = std::max(0.0, band_square_sum[i] / window_count -
                                           mean * mean);
    feature[i] = static_cast<float>(mean);
    feature[24 + i] = static_cast<float>(std::sqrt(variance));
    feature[48 + i] = static_cast<float>(band_delta_sum[i] / delta_count);
    const double raw_mean = raw_band_sum[i] / window_count;
    band_total += raw_mean;
    weighted_band += raw_mean * i;
    log_sum += std::log(raw_mean + 1.0e-12);
  }
  for (std::size_t quarter = 0; quarter < quarter_sum.size(); ++quarter) {
    const double divisor = static_cast<double>(
        std::max<std::size_t>(1, quarter_count[quarter]));
    for (std::size_t band = 0; band < quarter_sum[quarter].size(); ++band)
      feature[72 + quarter * 12 + band] =
          static_cast<float>(quarter_sum[quarter][band] / divisor);
  }

  const double arithmetic = band_total / raw_band_sum.size();
  const double geometric = std::exp(log_sum / raw_band_sum.size());
  const double rms = std::sqrt(square_sum / count);
  feature[120] = static_cast<float>(rms);
  feature[121] = static_cast<float>(crossings / static_cast<double>(count));
  feature[122] = static_cast<float>(
      band_total > 0.0 ? weighted_band / (23.0 * band_total) : 0.0);
  feature[123] = static_cast<float>(
      arithmetic > 0.0 ? geometric / arithmetic : 0.0);
  feature[124] = static_cast<float>(absolute_sum / count);
  feature[125] = static_cast<float>(peak);
  feature[126] = static_cast<float>(peak > 0.0 ? rms / peak : 0.0);
  const double low = raw_band_sum[0] + raw_band_sum[1] + raw_band_sum[2] +
                     raw_band_sum[3] + 1.0e-12;
  const double high = raw_band_sum[20] + raw_band_sum[21] + raw_band_sum[22] +
                      raw_band_sum[23] + 1.0e-12;
  feature[127] = static_cast<float>(std::log(low / high));
  normalize(feature);
  return feature;
}

CodecPtr openDecoder(AVFormatContext* format, int stream_index) {
  const AVCodecParameters* parameters = format->streams[stream_index]->codecpar;
  const AVCodec* decoder = avcodec_find_decoder(parameters->codec_id);
  if (!decoder) throw std::runtime_error("no decoder for selected stream");
  CodecPtr context(avcodec_alloc_context3(decoder));
  if (!context) throw std::bad_alloc();
  check(avcodec_parameters_to_context(context.get(), parameters),
        "copy codec parameters");
  AVDictionary* decoder_options = nullptr;
  av_dict_set(&decoder_options, "threads", "auto", 0);
  const int status = avcodec_open2(context.get(), decoder, &decoder_options);
  av_dict_free(&decoder_options);
  check(status, "open decoder");
  return context;
}

double rawFrameTime(const AVFrame* frame, const AVStream* stream,
                    double origin) {
  if (frame->best_effort_timestamp == AV_NOPTS_VALUE)
    return std::numeric_limits<double>::quiet_NaN();
  return frame->best_effort_timestamp * av_q2d(stream->time_base) - origin;
}

}  // namespace

MediaFeatures MediaAnalyzer::analyze(const std::string& input_path,
                                     const std::string& source_id) const {
  AVFormatContext* raw_format = nullptr;
  check(avformat_open_input(&raw_format, input_path.c_str(), nullptr, nullptr),
        "open input");
  FormatPtr format(raw_format);
  check(avformat_find_stream_info(format.get(), nullptr), "read stream information");
  const double total_seconds =
      format->duration == AV_NOPTS_VALUE
          ? 0.0
          : format->duration / static_cast<double>(AV_TIME_BASE);
  double processed_seconds = 0.0;
  auto reportProgress = [&](double seconds) {
    processed_seconds = std::max(processed_seconds, seconds);
    if (options_.progress_callback)
      options_.progress_callback(processed_seconds, total_seconds);
  };

  const bool want_video = options_.mode != AnalysisMode::Audio;
  const bool want_audio = options_.mode != AnalysisMode::Video;
  const int video_index = want_video
                              ? av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO,
                                                    -1, -1, nullptr, 0)
                              : AVERROR_STREAM_NOT_FOUND;
  const int audio_index = want_audio
                              ? av_find_best_stream(format.get(), AVMEDIA_TYPE_AUDIO,
                                                    -1, -1, nullptr, 0)
                              : AVERROR_STREAM_NOT_FOUND;
  if (options_.mode == AnalysisMode::Video && video_index < 0)
    throw std::runtime_error("input contains no decodable video stream");
  if (options_.mode == AnalysisMode::Audio && audio_index < 0)
    throw std::runtime_error("input contains no decodable audio stream");
  if (options_.mode == AnalysisMode::Both && (video_index < 0 || audio_index < 0))
    throw std::runtime_error("--mode both requires video and audio streams");
  if (video_index < 0 && audio_index < 0)
    throw std::runtime_error("input contains no supported video or audio stream");

  CodecPtr video_decoder = video_index >= 0 ? openDecoder(format.get(), video_index) : nullptr;
  CodecPtr audio_decoder = audio_index >= 0 ? openDecoder(format.get(), audio_index) : nullptr;
  SwsPtr scaler;
  if (video_decoder && options_.extract_perceptual_video) {
    scaler.reset(sws_getContext(video_decoder->width, video_decoder->height,
                                video_decoder->pix_fmt, kVideoWidth, kVideoHeight,
                                AV_PIX_FMT_GRAY8, SWS_BILINEAR, nullptr, nullptr, nullptr));
    if (!scaler) throw std::runtime_error("cannot create video scaler");
  }
  SwrPtr resampler;
  if (audio_decoder) {
    AVChannelLayout mono{};
    av_channel_layout_default(&mono, 1);
    SwrContext* raw_resampler = nullptr;
    const int allocation = swr_alloc_set_opts2(
        &raw_resampler, &mono, AV_SAMPLE_FMT_FLT, kAudioRate,
        &audio_decoder->ch_layout, audio_decoder->sample_fmt,
        audio_decoder->sample_rate, 0, nullptr);
    av_channel_layout_uninit(&mono);
    check(allocation, "allocate audio resampler");
    resampler.reset(raw_resampler);
    check(swr_init(resampler.get()), "initialize audio resampler");
  }

  std::vector<Accumulator> accumulators;
  auto bucketFor = [&](double seconds) -> Accumulator& {
    const auto index = static_cast<std::size_t>(std::max(0.0, std::floor(seconds / options_.bucket_seconds)));
    if (index > 7U * 24U * 3600U / options_.bucket_seconds)
      throw std::runtime_error("media timestamp exceeds the seven-day safety limit");
    if (accumulators.size() <= index) accumulators.resize(index + 1);
    return accumulators[index];
  };
  std::vector<FeatureBucket> audio_buckets;
  std::vector<float> previous_audio_samples;
  std::size_t previous_audio_index = std::numeric_limits<std::size_t>::max();
  const auto audio_window_samples = static_cast<std::size_t>(
      std::max(1.0, std::round(options_.bucket_seconds * kAudioRate)));
  auto byteAtFraction = [](const Accumulator& source, double fraction) {
    if (source.start_byte == std::numeric_limits<std::int64_t>::max() ||
        source.end_byte < source.start_byte)
      return std::int64_t{-1};
    fraction = std::clamp(fraction, 0.0, 1.0);
    return source.start_byte + static_cast<std::int64_t>(std::llround(
                                   (source.end_byte - source.start_byte) *
                                   fraction));
  };
  auto emitAudioWindows = [&](std::size_t index,
                              const std::vector<float>& first,
                              const Accumulator* second) {
    if (first.empty()) return;
    std::vector<float> samples;
    samples.reserve(first.size() +
                    (second == nullptr ? 0 : second->audio_samples.size()));
    samples.insert(samples.end(), first.begin(), first.end());
    if (second != nullptr)
      samples.insert(samples.end(), second->audio_samples.begin(),
                     second->audio_samples.end());

    const double interval_start = index * options_.bucket_seconds;
    const double interval_end = interval_start + options_.bucket_seconds;
    const auto first_step = static_cast<std::int64_t>(std::ceil(
        (interval_start - 1.0e-9) / options_.audio_hop_seconds));
    for (std::int64_t step = std::max<std::int64_t>(0, first_step);;
         ++step) {
      const double start = step * options_.audio_hop_seconds;
      if (start >= interval_end - 1.0e-9) break;
      const auto sample_offset = static_cast<std::size_t>(std::max(
          0.0, std::round((start - interval_start) * kAudioRate)));
      if (sample_offset + audio_window_samples > samples.size()) continue;

      FeatureBucket bucket;
      bucket.start_seconds = start;
      bucket.end_seconds = start + options_.bucket_seconds;
      const double start_fraction =
          first.empty() ? 0.0
                        : sample_offset / static_cast<double>(first.size());
      bucket.start_byte = byteAtFraction(accumulators[index], start_fraction);
      const auto end_offset = sample_offset + audio_window_samples;
      if (end_offset <= first.size()) {
        bucket.end_byte = byteAtFraction(
            accumulators[index], end_offset / static_cast<double>(first.size()));
      } else if (second != nullptr && !second->audio_samples.empty()) {
        bucket.end_byte = byteAtFraction(
            *second, (end_offset - first.size()) /
                         static_cast<double>(second->audio_samples.size()));
      }
      bucket.audio = audioFeature(
          samples.data() + static_cast<std::ptrdiff_t>(sample_offset),
          static_cast<int>(audio_window_samples));
      bucket.has_audio = true;
      audio_buckets.push_back(std::move(bucket));
    }
  };
  auto finalizeAudioBucket = [&](std::size_t index) {
    if (index >= accumulators.size()) return;
    auto& accumulator = accumulators[index];
    if (accumulator.audio_samples.empty()) return;
    accumulator.audio = audioFeature(accumulator.audio_samples.data(),
                                     static_cast<int>(accumulator.audio_samples.size()));
    if (previous_audio_index != std::numeric_limits<std::size_t>::max()) {
      if (previous_audio_index + 1 == index)
        emitAudioWindows(previous_audio_index, previous_audio_samples,
                         &accumulator);
      else
        emitAudioWindows(previous_audio_index, previous_audio_samples, nullptr);
    }
    previous_audio_samples = std::move(accumulator.audio_samples);
    previous_audio_index = index;
  };

  const double origin = format->start_time == AV_NOPTS_VALUE
                            ? 0.0
                            : format->start_time / static_cast<double>(AV_TIME_BASE);
  std::vector<TimestampDiscontinuity> timestamp_discontinuities;
  std::size_t recoverable_decode_errors = 0;
  std::size_t dropped_packets = 0;
  double timeline_frontier = 0.0;
  TimestampNormalizer video_timestamps(
      "video", options_.repair_timestamp_discontinuities,
      options_.timestamp_jump_threshold_seconds,
      options_.timestamp_backwards_tolerance_seconds);
  TimestampNormalizer audio_timestamps(
      "audio", options_.repair_timestamp_discontinuities,
      options_.timestamp_jump_threshold_seconds,
      options_.timestamp_backwards_tolerance_seconds);
  double default_video_duration = 1.0 / 25.0;
  if (video_index >= 0) {
    const AVRational frame_rate = av_guess_frame_rate(
        format.get(), format->streams[video_index], nullptr);
    if (frame_rate.num > 0 && frame_rate.den > 0)
      default_video_duration = av_q2d(av_inv_q(frame_rate));
  }
  std::size_t active_audio_bucket = 0;
  bool has_active_audio_bucket = false;
  PacketPtr packet(av_packet_alloc());
  FramePtr frame(av_frame_alloc());
  if (!packet || !frame) throw std::bad_alloc();
  std::array<std::uint8_t, kVideoWidth * kVideoHeight> gray{};

  auto receiveVideo = [&](std::int64_t packet_position, int packet_size) {
    while (true) {
      const int status = avcodec_receive_frame(video_decoder.get(), frame.get());
      if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) break;
      if (recoverableDecodeError(status)) {
        ++recoverable_decode_errors;
        av_frame_unref(frame.get());
        break;
      }
      check(status, "decode video frame");
      const double raw_seconds = rawFrameTime(
          frame.get(), format->streams[video_index], origin);
      double frame_duration = default_video_duration;
      if (frame->duration > 0)
        frame_duration = frame->duration *
                         av_q2d(format->streams[video_index]->time_base);
      const double seconds = video_timestamps.map(
          raw_seconds, std::isfinite(raw_seconds), frame_duration,
          timeline_frontier, timestamp_discontinuities);
      auto& accumulator = bucketFor(seconds);
      if (options_.extract_perceptual_video && accumulator.video_samples < 2) {
        std::uint8_t* output[] = {gray.data(), nullptr, nullptr, nullptr};
        int strides[] = {kVideoWidth, 0, 0, 0};
        sws_scale(scaler.get(), frame->data, frame->linesize, 0, frame->height,
                  output, strides);
        auto feature = videoFeature(gray.data(), kVideoWidth);
        for (std::size_t i = 0; i < feature.size(); ++i)
          accumulator.video_sum[i] += feature[i];
        accumulator.video_weight += 1.0;
        ++accumulator.video_samples;
      }
      updateBytes(accumulator, packet_position, packet_size);
      reportProgress(seconds);
      av_frame_unref(frame.get());
    }
  };

  auto receiveAudio = [&](std::int64_t packet_position, int packet_size) {
    while (true) {
      const int status = avcodec_receive_frame(audio_decoder.get(), frame.get());
      if (status == AVERROR(EAGAIN) || status == AVERROR_EOF) break;
      if (recoverableDecodeError(status)) {
        ++recoverable_decode_errors;
        av_frame_unref(frame.get());
        break;
      }
      check(status, "decode audio frame");
      const double raw_seconds = rawFrameTime(
          frame.get(), format->streams[audio_index], origin);
      const int capacity = static_cast<int>(av_rescale_rnd(
          swr_get_delay(resampler.get(), audio_decoder->sample_rate) + frame->nb_samples,
          kAudioRate, audio_decoder->sample_rate, AV_ROUND_UP));
      std::vector<float> converted(std::max(1, capacity));
      std::uint8_t* output[] = {reinterpret_cast<std::uint8_t*>(converted.data())};
      const int produced = swr_convert(resampler.get(), output, capacity,
                                       const_cast<const std::uint8_t**>(frame->extended_data),
                                       frame->nb_samples);
      if (produced < 0) {
        if (!recoverableDecodeError(produced))
          check(produced, "resample audio frame");
        ++recoverable_decode_errors;
        av_frame_unref(frame.get());
        break;
      }
      if (produced > 0) {
        const double frame_duration =
            produced / static_cast<double>(kAudioRate);
        const double seconds = audio_timestamps.map(
            raw_seconds, std::isfinite(raw_seconds), frame_duration,
            timeline_frontier, timestamp_discontinuities);
        int offset = 0;
        while (offset < produced) {
          const double current_seconds =
              seconds + offset / static_cast<double>(kAudioRate);
          const auto bucket_index = static_cast<std::size_t>(
              std::max(0.0, std::floor(current_seconds /
                                       options_.bucket_seconds)));
          auto& accumulator = bucketFor(current_seconds);
          if (has_active_audio_bucket && bucket_index > active_audio_bucket) {
            for (std::size_t i = active_audio_bucket; i < bucket_index; ++i)
              finalizeAudioBucket(i);
          }
          active_audio_bucket = std::max(active_audio_bucket, bucket_index);
          has_active_audio_bucket = true;

          const double bucket_end =
              (bucket_index + 1) * options_.bucket_seconds;
          const int until_boundary = std::max(
              1, static_cast<int>(std::ceil(
                     (bucket_end - current_seconds) * kAudioRate - 1.0e-6)));
          const int take = std::min(produced - offset, until_boundary);
          if (accumulator.audio_samples.empty())
            accumulator.audio_samples.reserve(static_cast<std::size_t>(
                std::ceil(options_.bucket_seconds * kAudioRate)));
          accumulator.audio_samples.insert(accumulator.audio_samples.end(),
                                           converted.begin() + offset,
                                           converted.begin() + offset + take);
          updateBytes(accumulator, packet_position, packet_size);
          offset += take;
        }
        reportProgress(seconds + frame_duration);
      }
      av_frame_unref(frame.get());
    }
  };

  while (true) {
    const int status = av_read_frame(format.get(), packet.get());
    if (status == AVERROR_EOF) break;
    if (recoverableDecodeError(status)) {
      ++recoverable_decode_errors;
      ++dropped_packets;
      av_packet_unref(packet.get());
      continue;
    }
    check(status, "read media packet");
    if ((packet->flags & AV_PKT_FLAG_CORRUPT) != 0) {
      ++recoverable_decode_errors;
      ++dropped_packets;
      av_packet_unref(packet.get());
      continue;
    }
    if (packet->stream_index == video_index) {
      int send_status = avcodec_send_packet(video_decoder.get(), packet.get());
      if (send_status == AVERROR(EAGAIN)) {
        receiveVideo(packet->pos, packet->size);
        send_status = avcodec_send_packet(video_decoder.get(), packet.get());
      }
      if (send_status < 0) {
        if (!recoverableDecodeError(send_status))
          check(send_status, "send video packet");
        ++recoverable_decode_errors;
        ++dropped_packets;
        av_packet_unref(packet.get());
        continue;
      }
      receiveVideo(packet->pos, packet->size);
    } else if (packet->stream_index == audio_index) {
      int send_status = avcodec_send_packet(audio_decoder.get(), packet.get());
      if (send_status == AVERROR(EAGAIN)) {
        receiveAudio(packet->pos, packet->size);
        send_status = avcodec_send_packet(audio_decoder.get(), packet.get());
      }
      if (send_status < 0) {
        if (!recoverableDecodeError(send_status))
          check(send_status, "send audio packet");
        ++recoverable_decode_errors;
        ++dropped_packets;
        av_packet_unref(packet.get());
        continue;
      }
      receiveAudio(packet->pos, packet->size);
    }
    av_packet_unref(packet.get());
  }
  if (video_decoder) {
    const int status = avcodec_send_packet(video_decoder.get(), nullptr);
    if (status < 0 && !recoverableDecodeError(status))
      check(status, "flush video decoder");
    if (status < 0) ++recoverable_decode_errors;
    receiveVideo(-1, 0);
  }
  if (audio_decoder) {
    const int status = avcodec_send_packet(audio_decoder.get(), nullptr);
    if (status < 0 && !recoverableDecodeError(status))
      check(status, "flush audio decoder");
    if (status < 0) ++recoverable_decode_errors;
    receiveAudio(-1, 0);
  }
  for (std::size_t i = 0; i < accumulators.size(); ++i)
    finalizeAudioBucket(i);
  if (previous_audio_index != std::numeric_limits<std::size_t>::max())
    emitAudioWindows(previous_audio_index, previous_audio_samples, nullptr);
  MediaFeatures result;
  const auto input_fs_path = std::filesystem::u8path(input_path);
  result.input_path = std::filesystem::absolute(input_fs_path).u8string();
  result.source_id = source_id;
  std::error_code size_error;
  result.file_size = static_cast<std::int64_t>(
      std::filesystem::file_size(input_fs_path, size_error));
  if (size_error) result.file_size = -1;
  result.recoverable_decode_errors = recoverable_decode_errors;
  result.dropped_packets = dropped_packets;
  result.duration_seconds = std::max(
      timeline_frontier,
      accumulators.empty() ? 0.0
                           : (accumulators.size() - 1) * options_.bucket_seconds);
  result.timestamp_discontinuities = std::move(timestamp_discontinuities);
  result.buckets.reserve(accumulators.size());
  for (std::size_t i = 0; i < accumulators.size(); ++i) {
    auto& source = accumulators[i];
    FeatureBucket bucket;
    bucket.start_seconds = i * options_.bucket_seconds;
    bucket.end_seconds = std::min(result.duration_seconds,
                                  (i + 1) * options_.bucket_seconds);
    bucket.start_byte = source.start_byte == std::numeric_limits<std::int64_t>::max()
                            ? -1
                            : source.start_byte;
    bucket.end_byte = source.end_byte;
    if (source.video_weight > 0.0) {
      bucket.video = std::move(source.video_sum);
      for (float& value : bucket.video) value /= static_cast<float>(source.video_weight);
      normalize(bucket.video);
      bucket.has_video = true;
    }
    if (!source.audio.empty()) {
      bucket.audio = std::move(source.audio);
      bucket.has_audio = true;
    }
    result.buckets.push_back(std::move(bucket));
  }
  result.audio_buckets = std::move(audio_buckets);
  if (options_.progress_callback)
    options_.progress_callback(result.duration_seconds, result.duration_seconds);
  return result;
}

#endif
}  // namespace recdup
