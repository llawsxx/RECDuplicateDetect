#include "recdup/timestamp_normalizer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace recdup {

TimestampNormalizer::TimestampNormalizer(
    std::string stream_name, bool repair_discontinuities,
    double forward_jump_threshold_seconds, double backward_tolerance_seconds)
    : stream_name_(std::move(stream_name)),
      repair_discontinuities_(repair_discontinuities),
      forward_jump_threshold_seconds_(forward_jump_threshold_seconds),
      backward_tolerance_seconds_(backward_tolerance_seconds) {
  if (forward_jump_threshold_seconds_ <= 0.0 ||
      backward_tolerance_seconds_ < 0.0)
    throw std::invalid_argument("invalid timestamp normalization thresholds");
}

double TimestampNormalizer::map(
    double raw_seconds, bool has_timestamp, double frame_duration,
    double& shared_frontier_seconds,
    std::vector<TimestampDiscontinuity>& discontinuities) {
  frame_duration = std::max(1.0e-6, frame_duration);
  if (!has_timestamp || !std::isfinite(raw_seconds)) {
    const double output = initialized_ ? next_seconds_ : shared_frontier_seconds;
    initialized_ = true;
    previous_output_seconds_ = output;
    next_seconds_ = output + frame_duration;
    shared_frontier_seconds = std::max(shared_frontier_seconds, next_seconds_);
    return output;
  }

  if (!initialized_) {
    initialized_ = true;
    double output = std::max(0.0, raw_seconds);
    if (repair_discontinuities_ &&
        output > shared_frontier_seconds + forward_jump_threshold_seconds_) {
      offset_seconds_ = shared_frontier_seconds - raw_seconds;
      output = shared_frontier_seconds;
      discontinuities.push_back({stream_name_, "forward", raw_seconds,
                                 raw_seconds, output});
    }
    previous_raw_seconds_ = raw_seconds;
    previous_output_seconds_ = output;
    next_seconds_ = output + frame_duration;
    shared_frontier_seconds = std::max(shared_frontier_seconds, next_seconds_);
    return output;
  }

  double output = raw_seconds + offset_seconds_;
  if (repair_discontinuities_) {
    const double forward_frontier =
        std::max(next_seconds_, shared_frontier_seconds);
    std::string direction;
    if (output > forward_frontier + forward_jump_threshold_seconds_)
      direction = "forward";
    else if (output < next_seconds_ - backward_tolerance_seconds_)
      direction = "backward";

    if (!direction.empty()) {
      // Decoder queues for audio and video can be different depths. The shared
      // frontier decides whether a gap is real, but the stream's own expected
      // time is the stable correction target.
      output = next_seconds_;
      offset_seconds_ = output - raw_seconds;
      discontinuities.push_back({stream_name_, direction,
                                 previous_raw_seconds_, raw_seconds, output});
    } else {
      // Preserve normal PTS spacing; frame duration is only a prediction and
      // can overlap the next timestamp after resampling or variable frame rate.
      output = std::max(output, previous_output_seconds_);
    }
  } else {
    output = std::max(0.0, output);
  }

  previous_raw_seconds_ = raw_seconds;
  previous_output_seconds_ = output;
  next_seconds_ = output + frame_duration;
  shared_frontier_seconds = std::max(shared_frontier_seconds, next_seconds_);
  return output;
}

}  // namespace recdup
