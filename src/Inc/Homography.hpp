#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>

typedef cv::Mat (*HomographyFinder)(std::vector<cv::Point2f>, std::vector<cv::Point2f>);

class Performance
{
public:
    std::chrono::duration<double, std::milli> timeTaken;
    int inlierCount;
    std::vector<int> total;
    friend std::ostream& operator<<(std::ostream& os, const Performance& perf);
};

class Homography
{
private:
    HomographyFinder findHomography;
public:
    Homography(HomographyFinder findHomography);
    ~Homography();
    Performance perf;
    int maxIterations = 10000;
    double distanceThreshold = 3;
    cv::Mat find(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB);
    static cv::Mat CV(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
    static cv::Mat DLT(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
};
