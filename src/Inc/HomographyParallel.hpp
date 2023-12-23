#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <Homography.hpp>

class HomographyParallel
{
private:
    cv::Mat DLT(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
public:
    HomographyParallel();
    ~HomographyParallel();
    Performance perf;
    int maxIterations = 10000;
    double distanceThreshold = 3;
    cv::Mat find(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB);
    static cv::Mat eigenDLT(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
};