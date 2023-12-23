#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <HomographyParallel.hpp>
#include <RANSAC.hpp>
#include <fstream>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

HomographyParallel::HomographyParallel()
{
}

HomographyParallel::~HomographyParallel()
{
}

// Function to read the content of a file into a string
std::string readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

cl_program createProgram(cl_context context, cl_device_id device, const char* source) {
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &err);
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    return program;
}

cv::Mat HomographyParallel::find(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB)
{
    std::string kernelString = readFile("../Src/HomographyParallelKernel.cl");
    const char *kernelSource = kernelString.c_str();

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue commandQueue = clCreateCommandQueue(context, device, 0, NULL);

    // Create an OpenCL program from the source code
    cl_program program = createProgram(context, device, kernelSource);

    // Create an OpenCL kernel from the program
    cl_kernel kernel = clCreateKernel(program, "ransacHomography", NULL);

    // Set kernel arguments
    int numMatches = 4; // Replace with the actual number of matches
    cl_mem srcPointsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cv::Point2f) * numMatches, pointsA.data(), NULL);
    cl_mem dstPointsBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cv::Point2f) * numMatches, pointsB.data(), NULL);
    cl_mem homographyMatrixBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 9, NULL, NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &srcPointsBuffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &dstPointsBuffer);
    clSetKernelArg(kernel, 2, sizeof(int), &numMatches);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &homographyMatrixBuffer);

    // Define global and local work sizes
    size_t globalSize = numMatches; // Set the global size based on the problem size
    size_t localSize = 1;  // Set the local size based on the device capabilities

    // Execute the kernel
    clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
    clFinish(commandQueue);

    // Retrieve the homography matrix from the device
    float homographyMatrix[9];
    clEnqueueReadBuffer(commandQueue, homographyMatrixBuffer, CL_TRUE, 0, sizeof(float) * 9, homographyMatrix, 0, NULL, NULL);

    // Convert the homography matrix to a cv::Mat for further use
    cv::Mat homographyMat = cv::Mat(3, 3, CV_32F, homographyMatrix);
    std::cout << homographyMat << std::endl;

    // Cleanup: Release OpenCL resources
    clReleaseMemObject(srcPointsBuffer);
    clReleaseMemObject(dstPointsBuffer);
    clReleaseMemObject(homographyMatrixBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);






    unsigned numPoints = 4;

    assert(pointsA.size() >= numPoints);
    assert(pointsB.size() >= numPoints);
    assert(pointsA.size() == pointsB.size());

    Performance perf;

    using std::chrono::high_resolution_clock;
    using std::chrono::duration;

    auto t1 = high_resolution_clock::now();

    std::vector<cv::Point2f> pickedPointsA, pickedPointsB;

    cv::Mat bestH;
    int bestInliers = 0;
    std::vector<cv::Point2f> bestPointsA, bestPointsB;

    for (int iter = 0; iter < maxIterations; iter++) {
        // 1. Randomly select a sample of 4 points
        selectRandomSample(pointsA, pointsB, pickedPointsA, pickedPointsB);

        // 2. Compute homography for this sample (using DLT and SVD)
        cv::Mat H = this->DLT(pickedPointsA, pickedPointsB);

        // 3. Count inliers
        int inliers = countInliers(pointsA, pointsB, H, distanceThreshold);

        // 4. Update best homography if current one has more inliers
        if (inliers > bestInliers) {
            bestInliers = inliers;
            bestH = H;
        }
    }

    if (bestInliers == 0) {
        std::cout << "No inliers found" << std::endl;
        exit(1);
    }

    auto t2 = high_resolution_clock::now();

    this->perf.timeTaken = t2 - t1;
    this->perf.inlierCount = bestInliers;

    // 5. Optional: refine homography using all inliers from the best model

    // H = this->findHomography(pointsA, pointsB)

    return bestH;
}

cv::Mat HomographyParallel::DLT(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB)
{
    // Compute mean and standard deviation for each set of points
    cv::Scalar meanA = cv::mean(pointsA);
    cv::Scalar meanB = cv::mean(pointsB);
    unsigned numPoints = pointsA.size();
    double stdDevA = 0, stdDevB = 0;

    for (int i = 0; i < numPoints; ++i) {
        stdDevA += pow(pointsA[i].x - meanA[0], 2) + pow(pointsA[i].y - meanA[1], 2);
        stdDevB += pow(pointsB[i].x - meanB[0], 2) + pow(pointsB[i].y - meanB[1], 2);
    }

    stdDevA = sqrt(stdDevA / (2 * numPoints));
    stdDevB = sqrt(stdDevB / (2 * numPoints));

    // Normalization matrices
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1 / stdDevA, 0, -meanA[0] / stdDevA,
                                            0, 1 / stdDevA, -meanA[1] / stdDevA,
                                            0, 0, 1);

    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1 / stdDevB, 0, -meanB[0] / stdDevB,
                                            0, 1 / stdDevB, -meanB[1] / stdDevB,
                                            0, 0, 1);

    // Normalize coordinates
    cv::Mat srcPoints(3, numPoints, CV_64F);
    cv::Mat dstPoints(3, numPoints, CV_64F);

    for (int i = 0; i < numPoints; ++i) {
        srcPoints.at<double>(0, i) = pointsA[i].x;
        srcPoints.at<double>(1, i) = pointsA[i].y;
        srcPoints.at<double>(2, i) = 1;

        dstPoints.at<double>(0, i) = pointsB[i].x;
        dstPoints.at<double>(1, i) = pointsB[i].y;
        dstPoints.at<double>(2, i) = 1;
    }

    // Apply normalization
    srcPoints = T1 * srcPoints;
    dstPoints = T2 * dstPoints;

    // Build the design matrix for DLT algorithm
    cv::Mat A(2 * numPoints, 9, CV_64F);
    for (int i = 0; i < numPoints; ++i) {
        double x1 = srcPoints.at<double>(0, i);
        double y1 = srcPoints.at<double>(1, i);
        double x2 = dstPoints.at<double>(0, i);
        double y2 = dstPoints.at<double>(1, i);

        A.at<double>(2 * i, 0) = 0;
        A.at<double>(2 * i, 1) = 0;
        A.at<double>(2 * i, 2) = 0;
        A.at<double>(2 * i, 3) = -x1;
        A.at<double>(2 * i, 4) = -y1;
        A.at<double>(2 * i, 5) = -1;
        A.at<double>(2 * i, 6) = y2 * x1;
        A.at<double>(2 * i, 7) = y2 * y1;
        A.at<double>(2 * i, 8) = y2;

        A.at<double>(2 * i + 1, 0) = x1;
        A.at<double>(2 * i + 1, 1) = y1;
        A.at<double>(2 * i + 1, 2) = 1;
        A.at<double>(2 * i + 1, 3) = 0;
        A.at<double>(2 * i + 1, 4) = 0;
        A.at<double>(2 * i + 1, 5) = 0;
        A.at<double>(2 * i + 1, 6) = -x2 * x1;
        A.at<double>(2 * i + 1, 7) = -x2 * y1;
        A.at<double>(2 * i + 1, 8) = -x2;
    }

    // Solve for the homography matrix using SVD
    cv::Mat U, D, Vt;
    cv::SVD::compute(A, D, U, Vt);

    // Extract the column of V corresponding to the smallest singular value
    cv::Mat h = Vt.row(Vt.rows - 1);

    // Reshape the matrix to obtain the homography matrix
    cv::Mat H = h.reshape(0, 3);

    // // Denormalize the homography matrix
    H = T2.inv() * H * T1;

    // normalize matrix
    H = H / H.at<double>(2, 2);

    return H;
}
