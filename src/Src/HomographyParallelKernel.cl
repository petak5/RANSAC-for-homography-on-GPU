// HomographyParallelKernel.cl

typedef struct {
    float x;
    float y;
} Point2f;

__kernel void ransacHomography( __global Point2f* srcPoints,
                                __global Point2f* dstPoints,
                                int numMatches,
                                __global float* H)
{
    for (int i = 0; i < 9; ++i) {
        H[i] = (i % 4 == 0) ? 1.0 : 0.0f;
    }
}
