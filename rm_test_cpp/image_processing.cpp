#include "image_processing.h"

#include <opencv2/geometry/2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

cv::Mat Creat_mask_by_inRange(const cv::Mat& hsv, const std::string& color) {
    cv::Mat mask;
    if (color == "orange") {
        cv::inRange(hsv, cv::Scalar(10, 76, 204), cv::Scalar(40, 255, 255), mask);
    } else if (color == "blue") {
        cv::inRange(hsv, cv::Scalar(100, 100, 178), cv::Scalar(130, 255, 255), mask);
    } else if (color == "red") {
        cv::Mat mask1, mask2;
        cv::inRange(hsv, cv::Scalar(160, 150, 150), cv::Scalar(180, 255, 255), mask1);
        cv::inRange(hsv, cv::Scalar(0, 150, 108), cv::Scalar(10, 255, 255), mask2);
        cv::bitwise_or(mask1, mask2, mask);
    } else {
        throw std::invalid_argument("Unknown color: " + color);
    }
    return mask;
}

cv::Mat Blur_img(const cv::Mat& img, const std::string& way = "Gaussian", int kernel_size = 3) {
    cv::Mat blurred;
    if (way == "Gaussian") {
        cv::GaussianBlur(img, blurred, cv::Size(kernel_size, kernel_size), 0);
    } else if (way == "Median") {
        cv::medianBlur(img, blurred, kernel_size);
    } else if (way == "Bilateral") {
        cv::bilateralFilter(img, blurred, kernel_size, 75, 75);
    } else if (way == "blur") {
        cv::blur(img, blurred, cv::Size(kernel_size, kernel_size));
    } else if (way == "None") {
        blurred = img;
    } else {
        throw std::invalid_argument("Unknown blur method: " + way);
    }
    return blurred;
}

cv::Mat Close_img(const cv::Mat& img, int kernel_size = 3) {
    cv::Mat result;
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_8U);
    cv::morphologyEx(img, result, cv::MORPH_CLOSE, kernel);
    return result;
}

cv::Mat Open_img(const cv::Mat& img, int kernel_size = 3) {
    cv::Mat result;
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_8U);
    cv::morphologyEx(img, result, cv::MORPH_OPEN, kernel);
    return result;
}

cv::Mat Get_mask(const cv::Mat& img, const std::string& color) {
    cv::Mat hsv;
    cv::cvtColor(Blur_img(img), hsv, cv::COLOR_BGR2HSV);
    cv::Mat mask = Creat_mask_by_inRange(hsv, color);
    if (color == "orange") {
        cv::bitwise_or(Creat_mask_by_inRange(hsv, "red"), mask, mask);
    }
    Out_image(Creat_mask_by_inRange(hsv, "blue"), "mask_blue");
    return Close_img(mask, 5);
}

bool check_light(const cv::RotatedRect& rect) {
    const float width = std::min(rect.size.width, rect.size.height);
    const float height = std::max(rect.size.width, rect.size.height);
    return width * height >= 20 && width > 0 && height / width >= 2;
}

std::vector<cv::RotatedRect> find_light(const cv::Mat& mask, cv::Mat& result_light) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::RotatedRect> lights;
    for (const auto& contour : contours) {
        cv::RotatedRect rect = cv::minAreaRect(contour);
        if (!check_light(rect)) continue;
        cv::Point2f corners[4];
        rect.points(corners);
        std::vector<cv::Point> points;
        for (const auto& p : corners) points.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
        lights.push_back(rect);
        cv::drawContours(result_light, std::vector<std::vector<cv::Point>>{points}, 0, cv::Scalar(0, 255, 0), 2);
    }
    return lights;
}

float get_angle(const cv::RotatedRect& rect) {
    float angle = rect.angle;
    if (rect.size.width < rect.size.height) angle += 90;
    angle = std::fmod(angle, 180.0f);
    return angle < 0 ? angle + 180.0f : angle;
}

bool check_match(const cv::RotatedRect& light1, const cv::RotatedRect& light2) {
    const float height1 = std::max(light1.size.width, light1.size.height);
    const float height2 = std::max(light2.size.width, light2.size.height);
    if (height1 <= 0) return false;
    const float dx = std::abs(light1.center.x - light2.center.x);
    return std::abs(get_angle(light1) - get_angle(light2)) <= 20 &&
           std::abs(height1 - height2) / height1 <= 0.3f &&
           std::abs(light1.center.y - light2.center.y) <= 0.5f * height1 &&
           dx <= 5 * height1 && dx / height1 >= 1.0f;
}

std::array<cv::Point, 4> sort_points(const cv::RotatedRect& rect) {
    cv::Point2f corners[4];
    rect.points(corners);
    int x_min = static_cast<int>(corners[0].x), x_max = x_min;
    int y_min = static_cast<int>(corners[0].y), y_max = y_min;
    for (const auto& p : corners) {
        const int x = static_cast<int>(p.x), y = static_cast<int>(p.y);
        x_min = std::min(x_min, x); x_max = std::max(x_max, x);
        y_min = std::min(y_min, y); y_max = std::max(y_max, y);
    }
    return {cv::Point(x_min, y_min), cv::Point(x_min, y_max),
            cv::Point(x_max, y_min), cv::Point(x_max, y_max)};
}

void print_aim(const std::vector<cv::Point>& aim_points, const std::string& color) {
    const cv::Moments m = cv::moments(aim_points);
    std::ofstream out("./outputs/information.txt", std::ios::trunc);
    if (!out) throw std::runtime_error("Cannot write outputs/information.txt");
    const int cx = m.m00 != 0 ? static_cast<int>(m.m10 / m.m00) : 0;
    const int cy = m.m00 != 0 ? static_cast<int>(m.m01 / m.m00) : 0;
    auto point = [](const cv::Point& p) { return "[" + std::to_string(p.x) + " " + std::to_string(p.y) + "]"; };
    out << "\t中心点:(" << cx << ',' << cy << ")\n"
        << "\t颜色:" << color << "\n"
        << "\t坐标:左上:" << point(aim_points[0]) << " 左下:" << point(aim_points[1])
        << " 右下:" << point(aim_points[2]) << " 右上:" << point(aim_points[3]) << "\n\n";
}

void putText(const cv::Point& point, const std::string& text, cv::Mat& result) {
    cv::putText(result, text, point, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
}

void make_text(const std::vector<cv::Point>& aim_points, cv::Mat& result) {
    putText(aim_points[0], "左上", result);
    putText(aim_points[1], "左下", result);
    putText(aim_points[2], "右下", result);
    putText(aim_points[3], "右上", result);
}

void solvePnP(const std::vector<cv::Point>&) {
    // The Python source leaves this function unimplemented.
}

void draw_aim(const cv::RotatedRect& light1, const cv::RotatedRect& light2,
              cv::Mat& result, const std::string& color) {
    const auto points1 = sort_points(light1);
    const auto points2 = sort_points(light2);
    std::vector<cv::Point> aim_points;
    if (light1.center.x < light2.center.x) {
        aim_points = {points1[0], points1[1], points2[3], points2[2]};
    } else {
        aim_points = {points2[0], points2[1], points1[3], points1[2]};
    }
    print_aim(aim_points, color);
    make_text(aim_points, result);
    cv::drawContours(result, std::vector<std::vector<cv::Point>>{aim_points}, 0, cv::Scalar(0, 255, 0), 2);
    solvePnP(aim_points);
}

void Match_light(const std::vector<cv::RotatedRect>& lights, cv::Mat& result, const std::string& color) {
    for (size_t i = 0; i < lights.size(); ++i) {
        for (size_t j = i + 1; j < lights.size(); ++j) {
            if (check_match(lights[i], lights[j])) draw_aim(lights[i], lights[j], result, color);
        }
    }
}

} // namespace

cv::Mat Input_image(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) throw std::runtime_error("error: cannot read image: " + image_path);
    return img;
}

void Out_image(const cv::Mat& img, const std::string& filename) {
    std::filesystem::create_directories("outputs");
    if (!cv::imwrite("outputs/" + filename + ".png", img)) {
        throw std::runtime_error("error: cannot write image: " + filename);
    }
}

void Dispose_image(const cv::Mat& img, cv::Mat& result, const std::string& color) {
    cv::Mat mask = Get_mask(img, color);
    Out_image(mask, "mask");
    cv::Mat result_light = img.clone();
    const auto lights = find_light(mask, result_light);
    Out_image(result_light, "light");
    Match_light(lights, result, color);
}

void Dispose_video(const std::string& file_path) {
    cv::VideoCapture cap(file_path);
    if (!cap.isOpened()) {
        std::cerr << "error\n";
        return;
    }
    cv::Mat img;
    while (cap.read(img)) {
        cv::Mat result = img.clone();
        Dispose_image(img, result);
        cv::imshow("img", result);
        cv::waitKey(1);
    }
}
