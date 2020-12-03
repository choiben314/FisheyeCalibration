#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/ccalib/omnidir.hpp"
#include <opencv2/core/utils/filesystem.hpp>

using namespace std;
using namespace cv;

/*** USER-DEFINED PARAMS ***/

const string LOCATION = "elgin_07_27_2020";
const string VIDEO = "DJI_0001"; // .MOV
const int VIDEO_INIT_SECOND = 0;
const string REG_VIDEO = "DJI_0001"; // .MOV
const int REG_INIT_SECOND = 0;
const double APERTURE_DISTANCE_PX = 283;
const double OUTPUT_FPS = 10;
const double OUTPUT_RESOLUTION_PX = 512;
const int MEDIAN_BLUR_RADIUS = 23;
const string DRIVE_LETTER = "E";

// File Paths
const string CAPTURE_MODE = "live_fisheye";
const string LIVE_STREAM_PATH = "rtmp://10.1.1.1/live/drone";
const string LIVE_IMAGES_PATH = DRIVE_LETTER + ":/NIFA/calibration/" + CAPTURE_MODE + "/image_";
const string CALIB_IMAGES_PATH = DRIVE_LETTER + ":/NIFA/calibration/" + CAPTURE_MODE + "/";
const string CALIB_IMAGES_LOCATIONS_PATH = DRIVE_LETTER + ":/NIFA/calibration/params/" + CAPTURE_MODE + "_paths.xml";
const string CAMERA_MODEL_PATH = DRIVE_LETTER + ":/NIFA/calibration/camera_models/calib_results.txt";
const string GCP_LOCATION = DRIVE_LETTER + ":/NIFA/footage/" + LOCATION + "/";
const string GCP_PATH = GCP_LOCATION + "gcp.xml";
const string GCP_FRAME_PATH = GCP_LOCATION + VIDEO + ".MOV";
const string GCP_REG_FRAME_PATH = GCP_LOCATION + REG_VIDEO + ".MOV";
const string GCP_PIXEL_COORDS_PATH = GCP_LOCATION + "pixel_coords.xml";
const string VIDEO_SAVE_PATH = DRIVE_LETTER + ":/NIFA/datasets/video/" + LOCATION + "_" + VIDEO + ".avi";
const string FRF_SAVE_PATH = DRIVE_LETTER + ":/NIFA/datasets/frf/" + LOCATION + "_" + VIDEO + ".frf";

// Calibration
const Size BOARD_SIZE = Size(7, 5);
const Size FINAL_SIZE = Size(1280, 720);

// Mouse
Point2d point;
bool clickStatus;

/*** FUNCTION DEFINITIONS ***/

//// IO Functions

// Get vector of filenames in a directory
static bool readStringList(const string& dir, vector<string>& l) {
    cv::glob(dir + "*", l, false);
}

// Get sequential numbered file paths and save to xml
void getLocationsFile(const string &xml, const string& image_dir, vector<string>& imageList) {
    imageList.clear();
    FileStorage fs(xml, FileStorage::READ);
    if (fs.isOpened()) {
        fs["images"] >> imageList;
    } else {
        readStringList(image_dir, imageList);
        if (!imageList.empty()) {
            FileStorage fs(xml, FileStorage::WRITE);
            fs << "images" << imageList;
        }
    }
}

// Set clickStatus to true on mouse click
static void onMouseClick(int event, int x, int y, int /*flags*/, void* /*param*/) {
    if (event == EVENT_LBUTTONDOWN || event == EVENT_RBUTTONDOWN)
    {
        point = Point2d((double)x, (double)y);
        clickStatus = true;
        cout << point << endl;
    }
}

void takeLivePhotos(string stream, string root, int start) {
    VideoCapture cap;
    cap.open(stream);

    namedWindow("Live Feed", 1);
    setMouseCallback("Live Feed", onMouseClick, 0);

    Mat frame;
    while (1) {
        cap >> frame;
        imshow("Live Feed", frame);

        if (clickStatus) {
            cout << "Writing image." << endl;
            string f = cv::format("%s%d.jpg", root.c_str(), start);
            cout << f << endl;
            imwrite(f, frame);
            start += 1;
            clickStatus = false;
        }

        char c = (char)waitKey(1);
        if (c == 27)
            break;
    }
}

//// CHESSBOARD CALIBRATION

// Downscale images if using original 4K footage
void downscale(Mat& old_frame, Mat& new_frame) {
    pyrDown(old_frame, new_frame);
    pyrDown(new_frame, new_frame);
    resize(new_frame, new_frame, FINAL_SIZE);
}

// Get image coordinates of chessboard corners
void getImagePoints(vector<vector<Point2d> >& imagePoints, Size& imageSize, vector<string>& imageList, Size BOARD_SIZE) {

    for (int currImage = 0; currImage < imageList.size(); currImage++) {
        cout << currImage << endl;
        Mat view = imread(imageList[currImage], IMREAD_COLOR);
        
        if (view.size() != FINAL_SIZE) {
            downscale(view, view);
        }

        imageSize = view.size();

        vector<Point2d> pointBuf;

        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
        bool found = findChessboardCorners(view, BOARD_SIZE, pointBuf, chessBoardFlags);

        if (found) {
            cout << "Chessboard found in " << imageList[currImage] << endl;
            Mat viewGray;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cornerSubPix(viewGray, pointBuf, Size(11, 11), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            imagePoints.push_back(pointBuf);
            drawChessboardCorners(view, BOARD_SIZE, Mat(pointBuf), found);
        }

        imshow("Image View", view);
        waitKey(1000);
    }
}

// Helper function: get XYZ coordinates of chessboard corners
static void calcBoardCornerHelper(Size BOARD_SIZE, double squareSize, vector<Point3d>& corners) {
    corners.clear();
    for (int i = 0; i < BOARD_SIZE.height; ++i)
        for (int j = 0; j < BOARD_SIZE.width; ++j)
            corners.push_back(Point3d(j * squareSize, i * squareSize, 0));
}

// Get XYZ object points for calibration
void getObjectPoints(vector<vector<Point3d> >& objectPoints, Size BOARD_SIZE, double squareSize) {
    double grid_width = squareSize * (BOARD_SIZE.width - 1);

    calcBoardCornerHelper(BOARD_SIZE, squareSize, objectPoints[0]);
    objectPoints[0][BOARD_SIZE.width - 1].x = objectPoints[0][0].x + grid_width;
}

// Get camera calibration parameters
void calibrateOmniCamera(vector<string>& imageList, Mat& K, Mat& D, Mat& xi, vector<Mat>& rvecs, vector<Mat>& tvecs) {
    FileStorage fs(CAMERA_MODEL_PATH, FileStorage::READ);
    if (fs.isOpened()) {
        fs["cameraMatrix"] >> K;
        fs["D"] >> D;
        fs["xi"] >> xi;
        fs["rvecs"] >> rvecs;
        fs["tvecs"] >> tvecs;
    } else {
        vector<vector<Point3d> > objectPoints(1);
        vector<vector<Point2d> > imagePoints;
        cv::Size imageSize;

        double squareSize = 50;

        Mat cameraMatrix, distCoeffs, _rvecs, _tvecs, idx;
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = 1;
        distCoeffs = Mat::zeros(4, 1, CV_64F);
        
        getImagePoints(imagePoints, imageSize, imageList, BOARD_SIZE);
        getObjectPoints(objectPoints, BOARD_SIZE, squareSize);
        objectPoints.resize(imagePoints.size(), objectPoints[0]);

        int flags = 0;
        TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 200, 0.0001);
        double rms = omnidir::calibrate(objectPoints, imagePoints, imageSize, K, xi, D, rvecs, tvecs, flags, criteria, idx);

        FileStorage fs(CAMERA_MODEL_PATH, FileStorage::WRITE);
        fs << "cameraMatrix" << K;
        fs << "D" << D;
        fs << "xi" << xi;
        fs << "rvecs" << rvecs;
        fs << "tvecs" << tvecs;
    }
}

// Transform, display, and save images in perspective model
void showTransformedImages(vector<string>& imageList, Mat& K, Mat& D, Mat& xi) {
    for (int currImage = 0; currImage < imageList.size(); currImage++) {
        Mat view, rview;
        view = imread(imageList[currImage], IMREAD_COLOR);

        if (view.size() != FINAL_SIZE) {
            downscale(view, view);
        }

        Mat map1, map2;
        omnidir::initUndistortRectifyMap(K, D, xi, cv::Mat(), cv::Mat(), view.size(), CV_64F, map1, map2, omnidir::RECTIFY_PERSPECTIVE);
        cv::remap(view, rview, map1, map2, INTER_LINEAR);
        //omnidir::undistortImage(view, rview, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);

        imshow("Original View", view);
        imshow("Image View", rview);

        cv::utils::fs::createDirectory(cv::format("%sundistorted", CALIB_IMAGES_PATH.c_str()));
        imwrite(cv::format("%sundistorted/image_%d.jpg", CALIB_IMAGES_PATH.c_str(), currImage), rview);

        waitKey(1000);
    }
}

//// REGISTRATION

// Get image frame for registration
void getRegistrationFrame(Mat& frame) {
    VideoCapture cap(GCP_REG_FRAME_PATH);
    int frame_rate = static_cast<int>(round(cap.get(CAP_PROP_FPS)));
    cap.set(CAP_PROP_POS_FRAMES, REG_INIT_SECOND * frame_rate);
    cap >> frame;
    if (frame.size() != FINAL_SIZE) {
        downscale(frame, frame);
    }
    cap.release();
}

// Get image coordinates of fiducial markers defined in GCP_PATH
static void markFiducials(const Mat& frame, Mat& world_coords, vector<Point2d>& pixel_coords) {
    FileStorage gcp_read(GCP_PATH, FileStorage::READ);
    gcp_read["gcp"] >> world_coords;

    FileStorage fs(GCP_PIXEL_COORDS_PATH, FileStorage::READ);
    if (fs.isOpened()) {
        fs["pixel_coords"] >> pixel_coords;
    } else {
        namedWindow("Mark fiducials", 1);
        setMouseCallback("Mark fiducials", onMouseClick, 0);
        imshow("Mark fiducials", frame);
        FileStorage fs(GCP_PIXEL_COORDS_PATH, FileStorage::WRITE);
        for (int row = 0; row < world_coords.rows; row++) {
            clickStatus = false;
            cout << "Mark location of: " << world_coords.row(row) << endl;
            while (!clickStatus) {
                waitKey(1);
            }

            pixel_coords.push_back(point);
            cout << "(" << point.x << ", " << point.y << ")" << endl;
        }
        fs << "pixel_coords" << pixel_coords;
    }
}

// Get camera pose via PNP solver
static void estimateCameraPose(const Mat &frame, const Mat& world_coords, const vector<Point2d>& pixel_coords, const Mat& K, const Mat& D, Mat& rvec, Mat& tvec, bool show, vector<Point2d>& new_pixel_coords) {
    solvePnPRansac(world_coords, pixel_coords, K, D, rvec, tvec);
    projectPoints(world_coords, rvec, tvec, K, D, new_pixel_coords);

    if (show) {
        Mat frameCopy = frame.clone();
        for (int i = 0; i < new_pixel_coords.size(); i++) {
            circle(frameCopy, new_pixel_coords[i], 3, Scalar(0, 0, 255), -1);
        }

        imshow("Reprojected fiducials", frameCopy);
    }

}

// Helper function: Get evenly spaced coordinates on a line between two points
void generateLineCoordinatesHelper(const Point3d& start, const Point3d& end, int numCoords, vector<Point3d>& line_coords) {
    if (start.x == end.x) {
        double total_distance = end.y - start.y;
        for (double i = 0; i < numCoords; i++) {
            double frac = i / (numCoords - 1);
            line_coords.push_back(Point3d(start.x, start.y + frac * total_distance, 0));
        }
    }
    else if (start.y == end.y) {
        double total_distance = end.x - start.x;
        for (double i = 0; i < numCoords; i++) {
            double frac = i / (numCoords - 1);
            line_coords.push_back(Point3d(start.x + frac * total_distance, start.y, 0));
        }
    }
    else {
        cout << "One set of corresponding coordinates must be equal." << endl;
    }
}

// Get coordinates for displaying East-North axes
void generateENZAxisCoordinates(const Point3d& min_corner, const Point3d& max_corner, Size coordFreq, Mat& K, Mat& D, Mat& rvec, Mat& tvec, vector<Point2d> &new_line_coords) {
    Point3d start_h = Point3d((min_corner.x + max_corner.x) / 2, min_corner.y, 0);
    Point3d end_h = Point3d((min_corner.x + max_corner.x) / 2, max_corner.y, 0);
    Point3d start_v = Point3d(min_corner.x, (min_corner.y + max_corner.y) / 2, 0);
    Point3d end_v = Point3d(max_corner.x, (min_corner.y + max_corner.y) / 2, 0);

    vector<Point3d> line_coords;
    generateLineCoordinatesHelper(start_v, end_v, coordFreq.height, line_coords);
    generateLineCoordinatesHelper(start_h, end_h, coordFreq.width, line_coords);

    projectPoints(line_coords, rvec, tvec, K, D, new_line_coords);
}

// Get coordinates for sampling on East-North plane
void generateENZPlaneCoordinates(const Point3d& min_corner, const Point3d& max_corner, const Size& size, Mat& K, Mat& D, Mat& rvec, Mat& tvec, vector<Point2d>& new_enz_coords) {
    Point3d top_left = Point3d(min_corner.x, max_corner.y, min_corner.z);
    vector<Point3d> left_edge;
    generateLineCoordinatesHelper(min_corner, top_left, size.height, left_edge);
    vector<Point3d> enz_coords;
    for (int i = 0; i < left_edge.size(); i++) {
        vector<Point3d> row;
        generateLineCoordinatesHelper(left_edge[i], Point3d(max_corner.x, left_edge[i].y, min_corner.z), size.width, row);

        for (int j = 0; j < row.size(); j++) {
            enz_coords.push_back(row[j]);
        }
    }
    projectPoints(enz_coords, rvec, tvec, K, D, new_enz_coords);
}

// Helper function: Bilinear interpolation for image pixels
cv::Vec3b getColorSubpixHelper(const cv::Mat& img, cv::Point2d pt)
{
    cv::Mat patch;
    cv::getRectSubPix(img, cv::Size(1, 1), pt, patch);
    return patch.at<cv::Vec3b>(0, 0);
}

// Get and display ENZ axis coordinates
void getCross(const Mat &frame, Point3d &min_corner, Point3d &max_corner, Size dim, Mat &K, Mat &D, Mat &xi, Mat &rvec, Mat &tvec, bool show, vector<Point2d> &line_coords) {
    generateENZAxisCoordinates(min_corner, max_corner, dim, K, D, rvec, tvec, line_coords);
    if (show) {
        Mat frameCopy = frame.clone();
        for (int i = 0; i < line_coords.size(); i++) {
            circle(frameCopy, line_coords[i], 3, Scalar(0, 255, 0), -1);
        }
        omnidir::undistortImage(frameCopy, frameCopy, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);
        imshow("Cross Coordinates", frameCopy);
    }
}

// Get and display ENZ sampling region coordinates
void getSamplingRegion(const Mat &frame, Point3d& min_corner, Point3d& max_corner, Size dim, Mat& K, Mat& D, Mat& rvec, Mat& tvec, bool show, vector<Point2d> &enz_coords) {
    generateENZPlaneCoordinates(min_corner, max_corner, dim, K, D, rvec, tvec, enz_coords);
    if (show) {
        Mat frameCopy = frame.clone();
        for (int i = 0; i < enz_coords.size(); i++) {
            circle(frameCopy, enz_coords[i], 3, Scalar(255, 0, 0), -1);
        }
        imshow("Sampling region", frameCopy);
    }
}

// Get image resampled to ENZ
void getRegisteredImage(const Mat& frame, vector<Point2d>& enz_coords, bool show, Mat& new_frame) {
    vector<Vec3b> registered;
    for (int i = 0; i < enz_coords.size(); i++) {
        registered.push_back(getColorSubpixHelper(frame, enz_coords[i]));
    }
    int length = sqrt(enz_coords.size());
    new_frame = Mat(registered).reshape(3, length); // FIX THIS THINGGGGG
    rotate(new_frame, new_frame, ROTATE_90_COUNTERCLOCKWISE);
    if (show) {
        imshow("Registered and resampled", new_frame);
    }
}

// Usage:

//takeLivePhotos(LIVE_STREAM_PATH, LIVE_IMAGES_PATH, 0);
//return 0;
//vector<string> imageList;

//Mat K, D, xi;
//vector<Mat> rvecs, tvecs;

//getLocationsFile(CALIB_IMAGES_LOCATIONS_PATH, CALIB_IMAGES_PATH, imageList);
//calibrateOmniCamera(imageList, K, D, xi, rvecs, tvecs);
//showTransformedImages(imageList, K, D, xi);