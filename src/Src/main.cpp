#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <main.hpp>
#include <Homography.hpp>
#include <RANSAC.hpp>
#include <Tools.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    // Load the image
    cv::Mat img1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat img2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    cv::Mat img1_rgb = cv::imread(argv[1]);
    cv::Mat img2_rgb = cv::imread(argv[2]);

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

    // Optional: Filter matches (for example, by distance)
    double max_dist = 0; double min_dist = 100;
    for (int i = 0; i < descriptors1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors1.rows; i++) {
        if (matches[i].distance <= std::max(2 * min_dist, 70.0)) {
            good_matches.push_back(matches[i]);
        }
    }

    // Draw matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);

    // // Show detected matches
    // cv::imshow("Matches", img_matches);
    // cv::waitKey(0);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;

    // Pick which matches to use
    // auto matches_to_use = good_matches;
    auto matches_to_use = matches;

    cout << "Used matches size: " << matches_to_use.size() << endl;
    for (size_t i = 0; i < matches_to_use.size(); i++) {
        points1.push_back(keypoints1[matches_to_use[i].queryIdx].pt);
        points2.push_back(keypoints2[matches_to_use[i].trainIdx].pt);
    }

    // Find homography matrix
    auto homography_ref = new Homography(Homography::CV);
    cv::Mat H_ref = homography_ref->find(points1, points2);

    std::cout << "Reference homography inliers: " << countInliers(points1, points2, H_ref, 3) << std::endl;

    std::cout << "Reference homography" << std::endl << H_ref << std::endl;
    std::cout << homography_ref->perf << std::endl;


    auto homography_our = new Homography(Homography::DLT);
    cv::Mat H_our = homography_our->find(points1, points2);
    std::cout << "Our homography" << std::endl << H_our << std::endl;
    std::cout << homography_our->perf << std::endl;

    auto homography_our_eigen = new Homography(Homography::eigenDLT);
    cv::Mat H_our_eigen = homography_our_eigen->find(points1, points2);
    std::cout << "Our homography (eigen)" << std::endl << H_our_eigen << std::endl;
    std::cout << homography_our_eigen->perf << std::endl;

    // "Refining"
    std::vector<cv::Point2f> inlierPointsA, inlierPointsB;

    for (size_t i = 0; i < points1.size(); i++) {
        cv::Point2f projectedPoint = applyHomography(H_our_eigen, points1[i]);
        double dx = projectedPoint.x - points2[i].x;
        double dy = projectedPoint.y - points2[i].y;
        double distance = std::sqrt(dx * dx + dy * dy);

        if (distance < 3) {
            inlierPointsA.push_back(points1[i]);
            inlierPointsB.push_back(points2[i]);
        }
    }

    // Compute homography from inliers
    // H_our_eigen = homography_our_eigen->find(inlierPointsA, inlierPointsB);

    // Draw inliers
    cv::Mat img_matches_inlier = drawMatchedPoints(img1_rgb, img2_rgb, inlierPointsA, inlierPointsB);

    // // Warp image
    cv::Mat img1_ref_warped;
    cv::warpPerspective(img1, img1_ref_warped, H_ref, img2.size());
    cv::Mat img1_our_warped;
    cv::warpPerspective(img1, img1_our_warped, H_our_eigen, img2.size());

    cv::imwrite("image_A.jpg", img1);
    cv::imwrite("image_B.jpg", img2);
    cv::imwrite("image_matches.jpg", img_matches);
    cv::imwrite("image_matches_inlier.jpg", img_matches_inlier);
    cv::imwrite("image_warped_reference.jpg", img1_ref_warped);
    cv::imwrite("image_warped_our.jpg", img1_our_warped);

    return 0;
}
