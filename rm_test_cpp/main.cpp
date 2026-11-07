#include "image_processing.h"

#include <opencv2/highgui.hpp>
#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string image_path = "./images/outpost_frame.png";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image_path" && i + 1 < argc) image_path = argv[++i];
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: rm_test_cpp [--image_path PATH]\n";
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << '\n';
            return 2;
        }
    }
    try {
        cv::Mat img = Input_image(image_path);
        DetectionResult detection = Dispose_image(img);
        Out_image(detection.mask_blue, "mask_blue");
        Out_image(detection.mask, "mask");
        Out_image(detection.light, "light");
        Out_image(detection.result, "result");
        Write_information(detection.armors);
        std::cout << "Image: " << img.cols << 'x' << img.rows << ", " << img.channels()
                  << " channels; armor candidates: " << detection.armors.size() << '\n';
        // 原 Python 程序没有打开窗口，省去无窗口情况下的 waitKey(0)。
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
