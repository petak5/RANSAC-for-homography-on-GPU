#include <RANSAC.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>


void selectRandomSample(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB, std::vector<cv::Point2f> &pickedPointsA, std::vector<cv::Point2f> &pickedPointsB)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // Create an index vector representing indices of the paired values
    std::vector<size_t> indices(pointsA.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Shuffle the indices
    std::shuffle(indices.begin(), indices.end(), gen);

    // Create two new vectors to store the picked pairs of values
    std::vector<cv::Point2f> newPickedPointsA;
    std::vector<cv::Point2f> newPickedPointsB;

    // Pick the first 4 indices and store the corresponding pairs
    for (size_t i = 0; i < 4 && i < indices.size(); ++i) {
        size_t index = indices[i];
        newPickedPointsA.push_back(pointsA[index]);
        newPickedPointsB.push_back(pointsB[index]);
    }
    pickedPointsA = newPickedPointsA;
    pickedPointsB = newPickedPointsB;
}

cv::Point2f applyHomography(const cv::Mat& H, const cv::Point2f& point)
{
    double x = point.x, y = point.y;
    double Z = H.at<double>(2, 0) * x + H.at<double>(2, 1) * y + H.at<double>(2, 2);
    double X = (H.at<double>(0, 0) * x + H.at<double>(0, 1) * y + H.at<double>(0, 2)) / Z;
    double Y = (H.at<double>(1, 0) * x + H.at<double>(1, 1) * y + H.at<double>(1, 2)) / Z;
    cv::Point2f finalPoint;
    finalPoint.x = X;
    finalPoint.y = Y;
    return finalPoint;
}

int countInliers(std::vector<cv::Point2f> const &pointsA, std::vector<cv::Point2f> const &pointsB, cv::Mat const &H, double distanceThreshold)
{
    int inliers = 0;

    for (size_t i = 0; i < pointsA.size(); i++) {
        cv::Point2f projectedPoint = applyHomography(H, pointsA[i]);
        double dx = projectedPoint.x - pointsB[i].x;
        double dy = projectedPoint.y - pointsB[i].y;
        double distance = std::sqrt(dx * dx + dy * dy);

        if (distance < distanceThreshold) {
            inliers++;
        }
    }

    return inliers;
}
