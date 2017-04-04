#pragma once

#include <vector>
#include <string>
#include <utility>

#define BOOST_ALL_DYN_LINK

#include <boost/foreach.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

std::vector<std::string>
list_files(const std::string &dir_path, const std::vector<std::string> &extensions);
