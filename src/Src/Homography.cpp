// class Homography, which will have method called find() that will be calling the findHomography function which has been declared on constructor
#include <opencv2/opencv.hpp>
#include <Homography.hpp>
#include <RANSAC.hpp>
#include <iostream>
#include <vector>

Homography::Homography(HomographyFinder findHomography)
{
    this->findHomography = findHomography;
}

Homography::~Homography()
{
}

cv::Mat Homography::find(std::vector<cv::Point2f> &pointsA, std::vector<cv::Point2f> &pointsB)
{
    unsigned numPoints = 4;

    assert(pointsA.size() >= numPoints);
    assert(pointsB.size() >= numPoints);
    assert(pointsA.size() == pointsB.size());

    std::vector<cv::Point2f> pickedPointsA, pickedPointsB;

    int maxIterations = 10000;
    double distanceThreshold = 3;

    cv::Mat bestH;
    int bestInliers = 0;
    int bestInliers_ref = 0;
    std::vector<cv::Point2f> bestPointsA_our, bestPointsB_our, bestPointsA_ref, bestPointsB_ref;

    for (int iter = 0; iter < maxIterations; iter++) {
        // 1. Randomly select a sample of 4 points
        selectRandomSample(pointsA, pointsB, pickedPointsA, pickedPointsB);

        // 2. Compute homography for this sample (using DLT and SVD)
        cv::Mat H_our = this->findHomography(pickedPointsA, pickedPointsB);
        cv::Mat H_ref = cv::findHomography(pickedPointsA, pickedPointsB);

        // std::cout << H_our << std::endl;
        // std::cout << H_ref << std::endl << std::endl;

        // 3. Count inliers
        int inliers_our = countInliers(pointsA, pointsB, H_our, distanceThreshold);
        int inliers_ref = countInliers(pointsA, pointsB, H_ref, distanceThreshold);

        // 4. Update best homography if current one has more inliers
        if (inliers_our > bestInliers) {
            bestInliers = inliers_our;
            bestH = H_our;
            bestPointsA_our = pickedPointsA;
            bestPointsB_our = pickedPointsB;

            std::cout << "New best inliers our: " << inliers_our << " | " << inliers_ref << std::endl;
        }
        if (inliers_ref > bestInliers_ref) {
            bestInliers_ref = inliers_ref;
            bestPointsA_ref = pickedPointsA;
            bestPointsB_ref = pickedPointsB;

            std::cout << "New best inliers ref: " << inliers_our << " | " << inliers_ref << std::endl;
        }
    }
    if (bestInliers == 0) {
        std::cout << "No inliers found" << std::endl;
        exit(1);
    }
    // 5. Optional: refine homography using all inliers from the best model

    std::cout << "Best our points A & B\n" << bestPointsA_our << std::endl;
    std::cout << bestPointsB_our << std::endl;
    std::cout << "Best ref points A & B\n" << bestPointsA_ref << std::endl;
    std::cout << bestPointsB_ref << std::endl;
    std::cout << "Our inliers count: " << bestInliers << std::endl;
    std::cout << "Ref inliers count: " << bestInliers_ref << std::endl;

    return bestH;
}

cv::Mat Homography::CV(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB)
{
    return cv::findHomography(pointsA, pointsB, cv::RANSAC);
}

cv::Mat Homography::DLT(std::vector<cv::Point2f> pointsA, std::vector<cv::Point2f> pointsB)
{
    // Compute mean and standard deviation for each set of points
    cv::Scalar meanA = cv::mean(pointsA);
    cv::Scalar meanB = cv::mean(pointsB);
    unsigned numPoints = pointsA.size();
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
