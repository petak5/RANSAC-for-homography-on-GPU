#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

using namespace std;
using namespace cv;

cv::Mat myFindHomography(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);
void selectRandomSample(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB, std::vector<cv::Point2f> &pickedPointsA, std::vector<cv::Point2f> &pickedPointsB);
int countInliers(std::vector<cv::Point2f> const &srcPoints, std::vector<cv::Point2f> const &dstPoints, cv::Mat const &H, double distanceThreshold);
cv::Point2f applyHomography(const cv::Mat& H, const cv::Point2f& point);
cv::Mat myFindHomographyHelper(int numPoints, std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB);

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

    // // Draw matches
    // cv::Mat img_matches;
    // cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);

    // // Show detected matches
    // cv::imshow("Matches", img_matches);
    // cv::waitKey(0);

    std::vector<cv::Point2f> points1;
    std::vector<cv::Point2f> points2;
    cout << "Good matches size: " << good_matches.size() << endl;
    for (size_t i = 0; i < good_matches.size(); i++) {
        points1.push_back(keypoints1[good_matches[i].queryIdx].pt);
        points2.push_back(keypoints2[good_matches[i].trainIdx].pt);
    }

    // Find homography matrix
    // cv::Mat H = cv::findHomography(points1, points2, cv::RANSAC);
    cv::Mat H = myFindHomography(points1, points2);

    // Warp image
    cv::Mat img1_warped;
    cv::warpPerspective(img1, img1_warped, H, img2.size());

    // Display the warped image
    cv::imshow("Warped Image", img1_warped);
    cv::waitKey(0);

    return 0;
}

cv::Mat myFindHomography(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB)
{
    unsigned numPoints = 3.5;

    assert(pointsA.size() >= numPoints);
    assert(pointsB.size() >= numPoints);
    assert(pointsA.size() == pointsB.size());

    std::vector<cv::Point2f> pickedPointsA, pickedPointsB;

    int maxIterations = 10000;
    double distanceThreshold = 3;

    cv::Mat bestH;
    cv::Mat bestH1;
    int bestInliers = 0;
    // cv::Mat H = cv::findHomography(pointsA, pointsB, cv::RANSAC);
    // return H;
    for (int iter = 0; iter < maxIterations; iter++) {
        // 1. Randomly select a sample of 4 points
            // Create a random number generator
        selectRandomSample(pointsA, pointsB, pickedPointsA, pickedPointsB);

        // 2. Compute homography for this sample (using DLT and SVD)
        cv::Mat H1 = myFindHomographyHelper(numPoints, pickedPointsA, pickedPointsB);
        cv::Mat H2 = cv::findHomography(pickedPointsA, pickedPointsB);

        // 3. Count inliers
        int inliers = countInliers(pointsA, pointsB, H2, distanceThreshold);

        // 4. Update best homography if current one has more inliers
        if (inliers > bestInliers) {
            
            bestInliers = inliers;
            bestH1 = H1;
            bestH = H2;
        }
    }
    if (bestInliers == 0) {
        cout << "No inliers found" << endl;
        exit(1);
    }
    // 5. Optional: refine homography using all inliers from the best model
cout << "Found homography with " << bestH1 << "matrix" << endl;
cout << "Found homography with " << bestH << "matrix" << endl;
    return bestH;
}

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
        Point2f projectedPoint = applyHomography(H, pointsA[i]);
        double dx = projectedPoint.x - pointsB[i].x;
        double dy = projectedPoint.y - pointsB[i].y;
        double distance = std::sqrt(dx * dx + dy * dy);

        if (distance < distanceThreshold) {
            inliers++;
        }
    }

    return inliers;
}

cv::Mat myFindHomographyHelper(int numPoints, std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB)
{
    // Compute mean and standard deviation for each set of points
    cv::Scalar meanA = cv::mean(pointsA);
    cv::Scalar meanB = cv::mean(pointsB);
    double stdDevA = 0, stdDevB = 0;

    for (int i = 0; i < numPoints; ++i) {
        stdDevA += pow(pointsA[i].x - meanA[0], 2) + pow(pointsA[i].y - meanA[1], 2);
        stdDevB += pow(pointsB[i].x - meanB[0], 2) + pow(pointsB[i].y - meanB[1], 2);
    }

    stdDevA = sqrt(stdDevA / (2 * numPoints));
    stdDevB = sqrt(stdDevB / (2 * numPoints));

    // Normalization matrices
    cv::Mat T1 = (cv::Mat_<double>(3, 3) << 1 / stdDevA, 0, -meanA[0] / stdDevA,
                                            0, 1 / stdDevA, -meanA[1] / stdDevA,
                                            0, 0, 1);

    cv::Mat T2 = (cv::Mat_<double>(3, 3) << 1 / stdDevB, 0, -meanB[0] / stdDevB,
                                            0, 1 / stdDevB, -meanB[1] / stdDevB,
                                            0, 0, 1);

    // Normalize coordinates
    cv::Mat srcPoints(3, numPoints, CV_64F);
    cv::Mat dstPoints(3, numPoints, CV_64F);

    for (int i = 0; i < numPoints; ++i) {
        srcPoints.at<double>(0, i) = pointsA[i].x;
        srcPoints.at<double>(1, i) = pointsA[i].y;
        srcPoints.at<double>(2, i) = 1;

        dstPoints.at<double>(0, i) = pointsB[i].x;
        dstPoints.at<double>(1, i) = pointsB[i].y;
        dstPoints.at<double>(2, i) = 1;
    }

    // Apply normalization
    srcPoints = T1 * srcPoints;
    dstPoints = T2 * dstPoints;

    // Build the design matrix for DLT algorithm
    cv::Mat A(2 * numPoints, 9, CV_64F);
    for (int i = 0; i < numPoints; ++i) {
        double x1 = srcPoints.at<double>(0, i);
        double y1 = srcPoints.at<double>(1, i);
        double x2 = dstPoints.at<double>(0, i);
        double y2 = dstPoints.at<double>(1, i);

        A.at<double>(2 * i, 0) = 0;
        A.at<double>(2 * i, 1) = 0;
        A.at<double>(2 * i, 2) = 0;
        A.at<double>(2 * i, 3) = -x1;
        A.at<double>(2 * i, 4) = -y1;
        A.at<double>(2 * i, 5) = -1;
        A.at<double>(2 * i, 6) = y2 * x1;
        A.at<double>(2 * i, 7) = y2 * y1;
        A.at<double>(2 * i, 8) = y2;

        A.at<double>(2 * i + 1, 0) = x1;
        A.at<double>(2 * i + 1, 1) = y1;
        A.at<double>(2 * i + 1, 2) = 1;
        A.at<double>(2 * i + 1, 3) = 0;
        A.at<double>(2 * i + 1, 4) = 0;
        A.at<double>(2 * i + 1, 5) = 0;
        A.at<double>(2 * i + 1, 6) = -x2 * x1;
        A.at<double>(2 * i + 1, 7) = -x2 * y1;
        A.at<double>(2 * i + 1, 8) = -x2;
    }

    // Solve for the homography matrix using SVD
    cv::Mat U, D, Vt;
    cv::SVD::compute(A, D, U, Vt);

    // Extract the column of V corresponding to the smallest singular value
    cv::Mat h = Vt.row(Vt.rows - 1);

    // Reshape the matrix to obtain the homography matrix
    cv::Mat H = h.reshape(0, 3);

    // // Denormalize the homography matrix
    H = T2.inv() * H * T1;

    // normalize matrix
    H = H / H.at<double>(2, 2);

    return H;
}
