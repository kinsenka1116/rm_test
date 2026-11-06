cmake_minimum_required(VERSION 3.16)
project(rm_test_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


find_package(OpenCV REQUIRED)
add_executable(rm_test_cpp main.cpp image_processing.cpp)
target_include_directories(rm_test_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(rm_test_cpp PRIVATE ${OpenCV_LIBS})
