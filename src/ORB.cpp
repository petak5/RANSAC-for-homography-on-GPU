#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Load the image
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Image not found!" << std::endl;
        return -1;
    }

    int nfeatures = 500; // Number of features to detect
    float scaleFactor = 1.2f;
    int nlevels = 8;
    int edgeThreshold = 31;
    int firstLevel = 0;
    int WTA_K = 2;
    auto scoreType = cv::ORB::HARRIS_SCORE;
    int patchSize = 31;
    int fastThreshold = 20;

    cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold);

    // Detect keypoints
    std::vector<cv::KeyPoint> keypoints;
    orb->detect(img, keypoints);

    // Compute descriptors
    cv::Mat descriptors;
    orb->compute(img, keypoints, descriptors);

    // Display keypoints for visualization
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);

    cv::imshow("ORB Keypoints", img_keypoints);
    cv::waitKey(0);

    return 0;
}
