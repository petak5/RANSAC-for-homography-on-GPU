#include <opencv2/opencv.hpp>

cv::Mat drawMatchedPoints(cv::Mat img1, cv::Mat img2, std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB)
{
    // Draw points on image 1
    for (const auto& point : pointsA) {
        cv::circle(img1, point, 5, cv::Scalar(0, 0, 255), -1); // Red color
    }

    // Draw points on image 2
    for (const auto& point : pointsB) {
        cv::circle(img2, point, 5, cv::Scalar(0, 0, 255), -1); // Red color
    }

    // Create a blank image to draw the combined result
    cv::Mat img_matched_points(img1.rows, img1.cols + img2.cols, img1.type());

    // Copy images to the img_matched_points
    img1.copyTo(img_matched_points(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(img_matched_points(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    // Draw lines connecting corresponding points
    for (size_t i = 0; i < pointsA.size(); ++i) {
        cv::Point2f offset(img1.cols, 0); // Offset for the second image
        cv::line(img_matched_points, pointsA[i], pointsB[i] + offset, cv::Scalar(0, 255, 0), 2); // Green color
    }

    return img_matched_points;
}