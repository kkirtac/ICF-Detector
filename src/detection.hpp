#pragma once

#include <vector>
#include <algorithm>
#include <tuple>


std::vector<std::tuple<int, int, int, int>>
nms(const std::vector<std::tuple<int, int, int, int, float>> &wins_in);
