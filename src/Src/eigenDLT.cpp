#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <Homography.hpp>
#include <vector>


cv::Mat Homography::eigenDLT(std::vector<cv::Point2f> srcPoints, std::vector<cv::Point2f> dstPoints) {
    Eigen::MatrixXd A(8, 9);

    for (size_t i = 0; i < srcPoints.size(); ++i) {
        double x = srcPoints[i].x, y = srcPoints[i].y;
        double u = dstPoints[i].x, v = dstPoints[i].y;

        A(2 * i, 0) = -x;
        A(2 * i, 1) = -y;
        A(2 * i, 2) = -1;
        A(2 * i, 3) = 0;
        A(2 * i, 4) = 0;
        A(2 * i, 5) = 0;
        A(2 * i, 6) = x * u;
        A(2 * i, 7) = y * u;
        A(2 * i, 8) = u;

        A(2 * i + 1, 0) = 0;
        A(2 * i + 1, 1) = 0;
        A(2 * i + 1, 2) = 0;
        A(2 * i + 1, 3) = -x;
        A(2 * i + 1, 4) = -y;
        A(2 * i + 1, 5) = -1;
        A(2 * i + 1, 6) = x * v;
        A(2 * i + 1, 7) = y * v;
        A(2 * i + 1, 8) = v;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd h = svd.matrixV().col(8);
    h /= h(8); // Normalize so that the last element is 1

    cv::Mat H;
    cv::eigen2cv(h, H);
    // resize M
    H = H.reshape(0, 3);
    // for (int i = 0; i < 3; ++i) {
    //     for (int j = 0; j < 3; ++j) {
    //         H.at<double>(i, j) = h(3 * i + j);
    //     }
    // }

    return H;
}
