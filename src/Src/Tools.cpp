#include <Tools.hpp>
#include <opencv2/opencv.hpp>
#include <RANSAC.hpp>

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

// Test speed of a homography
// Prints CSV with values to standard output
void test_speed(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, Homography* homography, size_t start_i, size_t end_i, size_t step_i)
{
    for (int i = start_i; i < end_i; i += step_i) {
        homography->maxIterations = i;

        using std::chrono::high_resolution_clock;
        using std::chrono::duration;

        auto t1 = std::chrono::high_resolution_clock::now();
        cv::Mat H = homography->find(points1, points2);
        // cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC, i);
        auto t2 = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = t2 - t1;
        std::cout << i << "," << elapsed.count() << std::endl;
    }
}

// Test accuracy of a homography
// Prints CSV with values to standard output
void test_accuracy(std::vector<cv::Point2f> points1, std::vector<cv::Point2f> points2, Homography* homography, size_t start_i, size_t end_i, size_t step_i)
{
    for (int i = start_i; i < end_i; i += step_i) {
        homography->maxIterations = i;

        cv::Mat H = homography->find(points1, points2);

        double distanceTotal = 0.0;

        for (size_t i = 0; i < points1.size(); i++) {
            cv::Point2f projectedPoint = applyHomography(H, points1[i]);
            double dx = projectedPoint.x - points2[i].x;
            double dy = projectedPoint.y - points2[i].y;
            distanceTotal += std::sqrt(dx * dx + dy * dy);
        }

        double distanceAvg = distanceTotal / points1.size();

        std::cout << i << "," << distanceAvg << std::endl;
    }
}