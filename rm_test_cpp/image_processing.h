#pragma once

#include <opencv2/core.hpp>
#include <string>

cv::Mat Input_image(const std::string& image_path = "./images/outpost_frame.png");
void Out_image(const cv::Mat& img, const std::string& filename);
void Dispose_image(const cv::Mat& img, cv::Mat& result, const std::string& color = "orange");
void Dispose_video(const std::string& file_path = "./input/video.avi");
