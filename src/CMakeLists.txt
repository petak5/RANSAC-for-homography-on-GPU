cmake_minimum_required(VERSION 3.15)
project(opencvTest)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package (Eigen3 3.4 REQUIRED NO_MODULE)

# add_compile_options(-Ofast -g)
include_directories(Inc Src Img)
file(GLOB_RECURSE SOURCES Src/*.cpp)
add_executable(opencvTest ${SOURCES})
target_link_libraries(opencvTest ${OpenCV_LIBS})
target_link_libraries (opencvTest Eigen3::Eigen)
