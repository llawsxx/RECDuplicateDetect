#pragma once

#include "recdup/types.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace recdup {

std::string makeResultJson(const MediaFeatures& media,
                           const std::vector<MatchSpan>& matches,
                           std::size_t database_vectors,
                           bool stored,
                           std::size_t database_vectors_after = 0,
                           const ProgrammeInference* inference = nullptr);

}  // namespace recdup
