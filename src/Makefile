.PHONY: all mac clean

all: main.c
	gcc -Wall -Wextra -D CL_TARGET_OPENCL_VERSION=100 main.c -o main -lOpenCL

mac: main.c
	clang -framework OpenCL -Wall -Wextra main.c -o main

clean:
	rm -f *.o *.a *.so main