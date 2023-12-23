#pragma once
#include <opencv2/opencv.hpp>
#include <Homography.hpp>

cv::Mat drawMatchedPoints(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
void test_speed(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, Homography homography, size_t start_i, size_t end_i, size_t step_i);
void test_accuracy(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, Homography homography, size_t start_i, size_t end_i, size_t step_i);
