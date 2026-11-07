#include "image_processing.h"

#include <opencv2/geometry.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
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
        throw std::invalid_argument("unknown color: " + color);
    }
    return mask;
}

cv::Mat Blur_img(const cv::Mat& img, const std::string& way = "Gaussian", int kernel_size = 3) {
    cv::Mat output;
    if (way == "Gaussian") cv::GaussianBlur(img, output, cv::Size(kernel_size, kernel_size), 0);
    else if (way == "Median") cv::medianBlur(img, output, kernel_size);
    else if (way == "Bilateral") cv::bilateralFilter(img, output, kernel_size, 75, 75);
    else if (way == "blur") cv::blur(img, output, cv::Size(kernel_size, kernel_size));
    else if (way == "None") output = img;
    else throw std::invalid_argument("unknown blur method: " + way);
    return output;
}

cv::Mat Close_img(const cv::Mat& img, int kernel_size = 3) {
    cv::Mat output;
    cv::morphologyEx(img, output, cv::MORPH_CLOSE, cv::Mat::ones(kernel_size, kernel_size, CV_8U));
    return output;
}

cv::Mat Open_img(const cv::Mat& img, int kernel_size = 3) {
    cv::Mat output;
    cv::morphologyEx(img, output, cv::MORPH_OPEN, cv::Mat::ones(kernel_size, kernel_size, CV_8U));
    return output;
}

bool check_light(const cv::RotatedRect& rect) {
    const float width = std::min(rect.size.width, rect.size.height);
    const float height = std::max(rect.size.width, rect.size.height);
    return width * height >= 20 && width > 0 && height / width >= 2;
}

std::vector<cv::RotatedRect> find_light(const cv::Mat& mask, cv::Mat& result_light, bool draw) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::RotatedRect> lights;
    for (const auto& contour : contours) {
        const cv::RotatedRect rect = cv::minAreaRect(contour);
        if (!check_light(rect)) continue;
        lights.push_back(rect);
        if (draw) {
            cv::Point2f corners[4];
            rect.points(corners);
            std::vector<cv::Point> points;
            for (const auto& p : corners) points.emplace_back(static_cast<int>(p.x), static_cast<int>(p.y));
            cv::drawContours(result_light, std::vector<std::vector<cv::Point>>{points}, 0,
                             cv::Scalar(0, 255, 0), 2);
        }
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
    cv::Point2f raw[4];
    rect.points(raw);
    std::array<cv::Point, 4> points;
    for (int i = 0; i < 4; ++i) points[i] = cv::Point(static_cast<int>(raw[i].x), static_cast<int>(raw[i].y));
    std::array<int, 4> order{0, 1, 2, 3};
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return points[a].y < points[b].y; });
    const int top_left = points[order[0]].x < points[order[1]].x ? order[0] : order[1];
    const int top_right = points[order[0]].x < points[order[1]].x ? order[1] : order[0];
    const int bottom_left = points[order[2]].x < points[order[3]].x ? order[2] : order[3];
    const int bottom_right = points[order[2]].x < points[order[3]].x ? order[3] : order[2];
    return {points[top_left], points[bottom_left], points[top_right], points[bottom_right]};
}

Armor draw_aim(const cv::RotatedRect& light1, const cv::RotatedRect& light2,
               cv::Mat& result, const std::string& color, bool annotate) {
    const auto points1 = sort_points(light1);
    const auto points2 = sort_points(light2);
    Armor armor;
    armor.color = color;
    const auto& left = light1.center.x < light2.center.x ? points1 : points2;
    const auto& right = light1.center.x < light2.center.x ? points2 : points1;
    armor.corners[0] = left[0];
    armor.corners[1] = left[1];
    armor.corners[2] = right[3];
    armor.corners[3] = right[2];
    const cv::Moments m = cv::moments(std::vector<cv::Point>(std::begin(armor.corners), std::end(armor.corners)));
    if (m.m00 != 0) armor.center = cv::Point(static_cast<int>(m.m10 / m.m00), static_cast<int>(m.m01 / m.m00));
    if (annotate) {
        // OpenCV Hershey 字体不支持中文，使用对应的 ASCII 缩写。
        const char* labels[4] = {"LT", "LB", "RB", "RT"};
        for (int i = 0; i < 4; ++i)
            cv::putText(result, labels[i], armor.corners[i], cv::FONT_HERSHEY_SIMPLEX,
                        1, cv::Scalar(0, 255, 0), 2);
    }
    cv::drawContours(result, std::vector<std::vector<cv::Point>>{
        std::vector<cv::Point>(std::begin(armor.corners), std::end(armor.corners))},
        0, cv::Scalar(0, 255, 0), 2);
    // Python 原版 solvePnP 为空函数，保持这个阶段不计算位姿。
    return armor;
}

DetectionResult process(const cv::Mat& img, const std::string& color, bool video_mode) {
    if (img.empty()) throw std::invalid_argument("empty input image");
    DetectionResult output;
    output.result = img.clone();
    output.light = img.clone();
    cv::Mat hsv;
    cv::cvtColor(Blur_img(img), hsv, cv::COLOR_BGR2HSV);
    if (!video_mode) output.mask_blue = Creat_mask_by_inRange(hsv, "blue");
    cv::Mat mask = Creat_mask_by_inRange(hsv, "orange");
    cv::bitwise_or(Creat_mask_by_inRange(hsv, "red"), mask, mask);
    if (!video_mode && color != "orange") mask = Creat_mask_by_inRange(hsv, color);
    output.mask = Close_img(mask, 5);
    const auto lights = find_light(output.mask, output.light, !video_mode);
    for (size_t i = 0; i < lights.size(); ++i)
        for (size_t j = i + 1; j < lights.size(); ++j)
            if (check_match(lights[i], lights[j]))
                output.armors.push_back(draw_aim(lights[i], lights[j], output.result, color, !video_mode));
    return output;
}

} // namespace

cv::Mat Input_image(const std::string& image_path) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) throw std::runtime_error("cannot read image: " + image_path);
    return img;
}

void Out_image(const cv::Mat& img, const std::string& filename) {
    std::filesystem::create_directories("outputs");
    if (!cv::imwrite("outputs/" + filename + ".png", img))
        throw std::runtime_error("cannot write outputs/" + filename + ".png");
}

DetectionResult Dispose_image(const cv::Mat& img, const std::string& color) {
    return process(img, color, false);
}

void Write_information(const std::vector<Armor>& armors) {
    std::filesystem::create_directories("outputs");
    std::ofstream out("outputs/information.txt", std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write outputs/information.txt");
    for (const auto& armor : armors) {
        out << "\t中心点:(" << armor.center.x << ',' << armor.center.y << ")\n"
            << "\t颜色:" << armor.color << "\n"
            << "\t坐标:左上:[" << armor.corners[0].x << ' ' << armor.corners[0].y << "]"
            << " 左下:[" << armor.corners[1].x << ' ' << armor.corners[1].y << "]"
            << " 右下:[" << armor.corners[2].x << ' ' << armor.corners[2].y << "]"
            << " 右上:[" << armor.corners[3].x << ' ' << armor.corners[3].y << "]\n\n";
    }
    if (!out) throw std::runtime_error("failed writing outputs/information.txt");
}

void Dispose_video(const std::string& file_path) {
    cv::VideoCapture cap(file_path);
    if (!cap.isOpened()) throw std::runtime_error("cannot open video: " + file_path);
    const int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
    cv::VideoWriter out("Output_video.avi", fourcc, 30, cv::Size(1280, 1024));
    if (!out.isOpened()) throw std::runtime_error("cannot write Output_video.avi");
    cv::Mat img;
    while (cap.read(img)) {
        const auto detection = process(img, "orange", true);
        out.write(detection.result);
    }
}
