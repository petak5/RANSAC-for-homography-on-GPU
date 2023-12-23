#pragma once
#include <opencv2/opencv.hpp>

void selectRandomSample(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB, std::vector<cv::Point2f> &pickedPointsA, std::vector<cv::Point2f> &pickedPointsB);
int countInliers(std::vector<cv::Point2f> const &srcPoints, std::vector<cv::Point2f> const &dstPoints, cv::Mat const &H, double distanceThreshold);
