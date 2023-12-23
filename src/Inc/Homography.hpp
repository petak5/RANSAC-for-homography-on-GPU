#pragma once
#include <opencv2/opencv.hpp>

typedef cv::Mat (*HomographyFinder)(std::vector<cv::Point2f>, std::vector<cv::Point2f>);

class Homography
{
private:
    HomographyFinder findHomography;
public:
    Homography(HomographyFinder findHomography);
    ~Homography();
    cv::Mat find(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB);
    static cv::Mat DLT(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
};

