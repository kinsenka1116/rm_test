#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

struct Armor {
    std::string color;
    cv::Point corners[4]; // 左上、左下、右下、右上；与 Python 原程序一致
    cv::Point center;
};

struct DetectionResult {
    cv::Mat result;
    cv::Mat mask;
    cv::Mat mask_blue;
    cv::Mat light;
    std::vector<Armor> armors;
};

cv::Mat Input_image(const std::string& image_path = "./images/outpost_frame.png");
void Out_image(const cv::Mat& img, const std::string& filename);
DetectionResult Dispose_image(const cv::Mat& img, const std::string& color = "orange");
void Dispose_video(const std::string& file_path = "./input/video.avi");
void Write_information(const std::vector<Armor>& armors);
