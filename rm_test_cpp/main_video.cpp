#include "image_processing.h"

#include <exception>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    std::string video_path = "./input/video.avi";
    if (argc == 2) video_path = argv[1];
    else if (argc > 2) {
        std::cerr << "Usage: rm_test_video [VIDEO_PATH]\n";
        return 2;
    }
    try {
        Dispose_video(video_path);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
