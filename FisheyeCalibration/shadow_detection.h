#pragma once

#include "opencv2/xfeatures2d.hpp"

using namespace cv::xfeatures2d;

void getSurfRotation(const Mat& ref_color, const Mat& rot_color, Mat& H) {
	Mat img_ref, img_rot;
	cvtColor(ref_color, img_ref, COLOR_BGR2GRAY);
	cvtColor(rot_color, img_rot, COLOR_BGR2GRAY);

	int minHessian = 400;
	Ptr<SURF> detector = SURF::create(minHessian);

	vector<KeyPoint> keypoints_ref, keypoints_rot;
	Mat descriptors_ref, descriptors_rot;
	detector->detectAndCompute(img_ref, noArray(), keypoints_ref, descriptors_ref);
	detector->detectAndCompute(img_rot, noArray(), keypoints_rot, descriptors_rot);

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::FLANNBASED);
	vector< vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors_ref, descriptors_rot, knn_matches, 2);

	const float ratio_thresh = 0.75f;
	vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	vector<Point2f> ref;
	vector<Point2f> rot;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		ref.push_back(keypoints_ref[good_matches[i].queryIdx].pt);
		rot.push_back(keypoints_rot[good_matches[i].trainIdx].pt);
	}
	H = estimateAffinePartial2D(rot, ref);
	//double s_inv = 1 / sqrt(H.at<double>(0, 0) * H.at<double>(0, 0) + H.at<double>(0, 1) * H.at<double>(0, 1));

	//Mat unscale = (Mat_<double>(2, 3) << s_inv, s_inv, 1, s_inv, s_inv, 1);
	//H = H.mul(unscale);
}

void getBinaryCloudMask(const Mat& img, Mat& bright, Mat& binary) {
	Mat hsv, v, diff, very_dark;
	vector<Mat> hsv_channels;

	cvtColor(img, hsv, COLOR_BGR2HSV);
	split(hsv, hsv_channels);
	v = hsv_channels[2];

	// Running maximum brightness for each pixel
	max(bright, v, bright);

	// Difference between brightest observed and current
	subtract(bright, v, diff);

	// Anything with too small of a difference is floored to zero (Avoids segmentation when there is no segmentation to be done). 
	threshold(diff, diff, 20, 255, THRESH_TOZERO);

	// Dynamic thresholding to binary by minimizing within-class variance 
	threshold(diff, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
}

void getApertureMask(const Mat& img, Mat& mask) {
	Mat img_thresh, img_binary, img_filled, img_filled_inv, img_final;

	cvtColor(img, img_binary, COLOR_RGB2GRAY);
	threshold(img_binary, img_thresh, 30, 255, THRESH_BINARY);
	img_thresh.copyTo(img_filled);
	
	floodFill(img_filled, Point(0, 0), Scalar(255));
	floodFill(img_filled, Point(0, OUTPUT_RESOLUTION_PX - 1), Scalar(255));
	floodFill(img_filled, Point(OUTPUT_RESOLUTION_PX - 1, 0), Scalar(255));
	floodFill(img_filled, Point(OUTPUT_RESOLUTION_PX - 1, OUTPUT_RESOLUTION_PX - 1), Scalar(255));
	
	bitwise_not(img_filled, img_filled_inv);
	mask = img_thresh | img_filled_inv;

	floodFill(mask, Point(0, 0), Scalar(0));
	floodFill(mask, Point(0, OUTPUT_RESOLUTION_PX - 1), Scalar(0));
	floodFill(mask, Point(OUTPUT_RESOLUTION_PX - 1, 0), Scalar(0));
	floodFill(mask, Point(OUTPUT_RESOLUTION_PX - 1, OUTPUT_RESOLUTION_PX - 1), Scalar(0));
}