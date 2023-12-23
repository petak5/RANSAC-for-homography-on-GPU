#pragma once
#include <opencv2/opencv.hpp>

cv::Mat drawMatchedPoints(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
