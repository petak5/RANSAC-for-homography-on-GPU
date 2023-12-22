.PHONY: all mac clean

all: main.cpp
	gcc -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=100 main.cpp -o main -lOpenCL -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_flann -lopencv_ml -lopencv_dnn -lopencv_photo -lopencv_stitching -lopencv_videostab -lopencv_xfeatures2d -lopencv_ximgproc -lopencv_xobjdetect -lopencv_xphoto

mac: main.cpp
	clang++ -framework OpenCL -Wall -Wextra -std=c++14 -DCL_HPP_TARGET_OPENCL_VERSION=120 -DCL_HPP_MINIMUM_OPENCL_VERSION=120 main.cpp -o main

clean:
	rm -f *.o *.a *.so main