#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Load image
    Mat img_rgb = imread(argv[1], IMREAD_COLOR);
    if (img_rgb.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }
    // get int value from argv[2]
    int threshold = atoi(argv[2]);
    Mat img = imread(argv[1], IMREAD_GRAYSCALE);

    // Harris corner detection parameters
    int blockSize = 2; // Size of neighborhood considered for corner detection
    int apertureSize = 3; // Aperture parameter for the Sobel operator
    double k = 0.04; // Harris detector free parameter

    // Detect Harris corners
    Mat dst = Mat::zeros(img.size(), CV_32FC1);
    cornerHarris(img, dst, blockSize, apertureSize, k);

    // Normalize and convert to 8-bit for visualization
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);

    // Mark corners in the image
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int)dst_norm.at<float>(i, j) > threshold) { // Threshold for detecting corners
                cv::Vec3b &pixel = img_rgb.at<cv::Vec3b>(i, j);
                pixel[2] = 255;
                pixel[1] = 0;
                pixel[0] = 0;
            }
        }
    }

    // Display the result
    namedWindow("Harris Corners", WINDOW_AUTOSIZE);
    imshow("Harris Corners", dst_norm_scaled);
    waitKey(0);

    return 0;
}
