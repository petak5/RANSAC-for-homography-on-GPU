#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Load the image
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);

    if (img1.empty() || img2.empty()) {
        std::cerr << "Error: Images not found!" << std::endl;
        return -1;
    }
    cv::Ptr<cv::ORB> orb = cv::ORB::create();

    // Detect keypoints and compute descriptors for both images
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    orb->detectAndCompute(img1, cv::Mat(), keypoints1, descriptors1);
    orb->detectAndCompute(img2, cv::Mat(), keypoints2, descriptors2);

    // Create BFMatcher object
    cv::BFMatcher matcher(cv::NORM_HAMMING);

    // Match descriptors
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    // // Optional: Filter matches (for example, by distance)
    // double max_dist = 0; double min_dist = 100;
    // for (int i = 0; i < descriptors1.rows; i++) {
    //     double dist = matches[i].distance;
    //     if (dist < min_dist) min_dist = dist;
    //     if (dist > max_dist) max_dist = dist;
    // }
    // std::vector<cv::DMatch> good_matches;
    // for (int i = 0; i < descriptors1.rows; i++) {
    //     if (matches[i].distance <= std::max(2 * min_dist, 30.0)) {
    //         good_matches.push_back(matches[i]);
    //     }
    // }

    // // Draw matches
    // cv::Mat img_matches;
    // cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);

    // // Show detected matches
    // cv::imshow("Matches", img_matches);
    // cv::waitKey(0);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    for (size_t i = 0; i < matches.size(); i++) {
        points1.push_back(keypoints1[matches[i].queryIdx].pt);
        points2.push_back(keypoints2[matches[i].trainIdx].pt);
    }

    // Find homography matrix
    cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);

    // Warp image
    cv::Mat img1_warped;
    cv::warpPerspective(img1, img1_warped, H, img2.size());

    // Display the warped image
    cv::imshow("Warped Image", img1_warped);
    cv::waitKey(0);

    return 0;
}



