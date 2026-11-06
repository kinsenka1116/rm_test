#include "image_processing.h"

#include <opencv2/highgui.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string image_path = "./images/outpost_frame.png";
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--image_path" && i + 1 < argc) {
            image_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: rm_test_cpp [--image_path PATH]\n";
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << '\n';
            return 2;
        }
    }

    try {
        cv::Mat img = Input_image(image_path);
        cv::Mat result = img.clone();
        Dispose_image(img, result);
        Out_image(result, "result");
        cv::waitKey(0);
        cv::destroyAllWindows();
    } catch (const cv::Exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
