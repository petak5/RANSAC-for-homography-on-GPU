.PHONY: all mac clean

all: main.c
	gcc -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=100 main.c -o main -lOpenCL

mac: main.cpp
	clang++ -framework OpenCL -Wall -Wextra -std=c++14 -DCL_HPP_TARGET_OPENCL_VERSION=120 -DCL_HPP_MINIMUM_OPENCL_VERSION=120 main.cpp -o main

clean:
	rm -f *.o *.a *.so main