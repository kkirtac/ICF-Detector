#pragma once

#include <vector>
#include <string>
#include <utility>
#include <tuple>
#include <random>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>

std::vector<cv::Mat>
integrate_channels(const std::vector<cv::Mat> &channels);

std::vector<cv::Mat>
compute_channels(const cv::Mat &image);

std::vector<std::tuple<int, int, int, int, int>>
generate_features(int channel_count, const std::tuple<int, int> &win_size,
                  const std::tuple<int, int> &min_size, int feature_count);

cv::Mat_<float>
extract_integral_features(const std::vector<cv::Mat> &integral_channels,
    const std::vector<std::tuple<int, int, int, int, int>> &features,
    const std::tuple<int, int> &pin);

void
save_features(const std::vector<std::tuple<int, int, int, int, int>> &features,
    const std::string &filename);

std::vector<std::tuple<int, int, int, int, int>>
load_features(const std::string &filename);

std::tuple<cv::Mat_<float>, cv::Mat_<float>>
sample_train_features(const std::string &pos_dir_path, const std::string &neg_dir_path,
    const std::vector<std::tuple<int, int, int, int, int>> &features);

std::tuple<cv::Mat_<float>, cv::Mat_<float>>
sample_train_features(const std::string &pos_dir_path, const std::string &neg_dir_path,
	int neg_win_per_img, const cv::Size neg_win_size,
    const std::vector<std::tuple<int, int, int, int, int>> &features,
    std::string filename = "5000_1000_2_4.featurevals", int numpos=0, int numneg=0);
