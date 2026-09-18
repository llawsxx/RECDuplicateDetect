#pragma once

#include "recdup/types.hpp"

#include <string>
#include <vector>

namespace recdup {

class TimestampNormalizer {
 public:
  TimestampNormalizer(std::string stream_name, bool repair_discontinuities,
                      double forward_jump_threshold_seconds,
                      double backward_tolerance_seconds);

  double map(double raw_seconds, bool has_timestamp, double frame_duration,
             double& shared_frontier_seconds,
             std::vector<TimestampDiscontinuity>& discontinuities);

 private:
  std::string stream_name_;
  bool repair_discontinuities_ = true;
  double forward_jump_threshold_seconds_ = 10.0;
  double backward_tolerance_seconds_ = 1.0;
  bool initialized_ = false;
  double offset_seconds_ = 0.0;
  double previous_raw_seconds_ = 0.0;
  double previous_output_seconds_ = 0.0;
  double next_seconds_ = 0.0;
};

}  // namespace recdup
