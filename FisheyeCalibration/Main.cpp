#include "opencv2/opencv.hpp"
#include "opencv2/ccalib/omnidir.hpp"

using namespace cv;
using namespace std;

/*** USER ADJUSTED PARAMS ***/
const string IMAGE_PATHS = "D:/NIFA/VID5.xml";
const string OMNI_MODEL_PATH = "D:/NIFA/omnidir_model_full.xml";
const string GCP_PATH = "D:/NIFA/lamberton_08_10_2020/gcp.xml";
const string GCP_FRAME_PATH = "D:/NIFA/lamberton_08_10_2020/frame1.JPG";

const bool CALIBRATED = true;
const bool SHOW_IMAGES = false;
const float RESIZE_FACTOR = 1.5;
const float CALIB_RESIZE_FACTOR = 1.375;

/*** FUNCTION DEFINITIONS ***/
static bool readStringList(const string& filename, vector<string>& l) {
    l.clear();
    FileStorage fs(filename, FileStorage::READ);
    if (!fs.isOpened())
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if (n.type() != FileNode::SEQ)
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for (; it != it_end; ++it)
        l.push_back((string)*it);
    return true;
}

void getImagePoints(vector<vector<Point2f> > &imagePoints, Size &imageSize, vector<string> &imageList, Size boardSize) {

    for (int currImage = 0; currImage < imageList.size(); currImage++) {
        
        Mat view;
        
        view = imread(imageList[currImage], IMREAD_COLOR);

        if (RESIZE_FACTOR != -1) {
            //pyrDown(view, view);
            resize(view, view, Size(view.cols / CALIB_RESIZE_FACTOR, view.rows / CALIB_RESIZE_FACTOR));
        }

        imageSize = view.size();

        vector<Point2f> pointBuf;
        
        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE;
        bool found = findChessboardCorners(view, boardSize, pointBuf, chessBoardFlags);
       
        if (found) {
            cout << "Chessboard found in " << imageList[currImage] << endl;
            Mat viewGray;
            int winSize = 11;
            cvtColor(view, viewGray, COLOR_BGR2GRAY);
            cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
                Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
            imagePoints.push_back(pointBuf);
            
            drawChessboardCorners(view, boardSize, Mat(pointBuf), found);
        }

        if (SHOW_IMAGES) {
            imshow("Image View", view);
            waitKey(1000);
        }
    }
}

static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners) {
    corners.clear();
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            corners.push_back(Point3f(j * squareSize, i * squareSize, 0));
}

void getObjectPoints(vector<vector<Point3f> >& objectPoints, Size boardSize, float squareSize) {
    float grid_width = squareSize * (boardSize.width - 1);

    calcBoardCornerPositions(boardSize, squareSize, objectPoints[0]);
    objectPoints[0][boardSize.width - 1].x = objectPoints[0][0].x + grid_width;
}

void generateFileNames(string prefix, string suffix, int lower, int upper) {
    for (int i = lower; i <= upper; i++) {
        cout << prefix << i << suffix << endl;
    }
}

Point2f point;
bool clickStatus;

static void onMouseClick(int event, int x, int y, int /*flags*/, void* /*param*/) {
    if (event == EVENT_LBUTTONDOWN)
    {
        point = Point2f((float)x, (float)y);
        clickStatus = true;
    }
}

static void markFiducials(Mat &orig_frame, Mat &world_coords, vector<Point2f> &pixel_coords) {
    namedWindow("Mark fiducials", 1);
    setMouseCallback("Mark fiducials", onMouseClick, 0);

    FileStorage gcp_read(GCP_PATH, FileStorage::READ);
    gcp_read["gcp"] >> world_coords;

    orig_frame = imread(GCP_FRAME_PATH);
    Mat frame = orig_frame.clone();

    Size original_size, new_size;
    original_size = frame.size();
    if (RESIZE_FACTOR != -1) {
        pyrDown(frame, frame);
        resize(frame, frame, Size(frame.cols / RESIZE_FACTOR, frame.rows / RESIZE_FACTOR));
    }
    new_size = frame.size();
    float width_ratio = (float)original_size.width / (float)new_size.width;
    float height_ratio = (float)original_size.height / (float)new_size.height;

    imshow("Mark fiducials", frame);

    //FileStorage pc_write(GCP_PIXEL_COORDS_PATH, FileStorage::WRITE);
    for (int row = 0; row < world_coords.rows; row++) {
        clickStatus = false;
        cout << "Mark location of: " << world_coords.row(row) << endl;
        while (!clickStatus) {
            waitKey(1);
        }

        float resize_ratio = RESIZE_FACTOR == -1 ? 1 : 2 * RESIZE_FACTOR;
        Point2f adjusted = Point2f(point.x * resize_ratio, point.y * resize_ratio);
        pixel_coords.push_back(adjusted);
        cout << "(" << adjusted.x << ", " << adjusted.y << ")" << endl;
    }
    //pc_write << "pixel_coords" << pixel_coords;
}

static void estimateCameraPose(Mat &orig_frame, const Mat &world_coords, const vector<Point2f> &pixel_coords, const Mat &K, const Mat &D, Mat &rvec, Mat &tvec, vector<Point2d>& new_pixel_coords) {
    //vector<Point2f> pixel_coords;
    //FileStorage pc_read(GCP_PIXEL_COORDS_PATH, FileStorage::READ);
    //pc_read["pixel_coords"] >> pixel_coords;

    solvePnPRansac(world_coords, pixel_coords, K, D, rvec, tvec);

    projectPoints(world_coords, rvec, tvec, K, D, new_pixel_coords);

    namedWindow("Reprojected fiducials");

    Mat frame = orig_frame.clone();
    if (RESIZE_FACTOR != -1) {
        pyrDown(frame, frame);
        resize(frame, frame, Size(frame.cols / RESIZE_FACTOR, frame.rows / RESIZE_FACTOR));
    }

    for (int i = 0; i < new_pixel_coords.size(); i++) {
        float resize_ratio = RESIZE_FACTOR == -1 ? 1 : 2 * RESIZE_FACTOR;
        Point2d new_image_coord = Point2d(new_pixel_coords[i].x / resize_ratio, new_pixel_coords[i].y / resize_ratio);
        circle(frame, new_image_coord, 3, Scalar(0, 0, 255), -1);
    }

    imshow("Reprojected fiducials", frame);
}

void generateLineCoordinates(const Point3f& start, const Point3f& end, int numCoords, vector<Point3f> &line_coords) {
    if (start.x == end.x) {
        float total_distance = end.y - start.y;
        for (float i = 0; i < numCoords; i++) {
            float frac = i / (numCoords - 1);
            line_coords.push_back(Point3f(start.x, start.y + frac * total_distance, 0));
        }
    }
    else if (start.y == end.y) {
        float total_distance = end.x - start.x;
        for (float i = 0; i < numCoords; i++) {
            float frac = i / (numCoords - 1);
            line_coords.push_back(Point3f(start.x + frac * total_distance, start.y, 0));
        }
    }
    else {
        cout << "One set of corresponding coordinates must be equal." << endl;
    }
}

/*** <MAIN FUNCTION ***/
int main(int argc, char** argv) {

    //generateFileNames("D:/NIFA/calibration_images/DJI_0", ".JPG", 182, 225);
    //return 0;

    vector<string> imageList;
    if (readStringList(IMAGE_PATHS, imageList)) {
        cout << "Successfully read image list." << endl;
    }
    else {
        cout << "Unable to read image list." << endl;
    }

    Mat K, D, xi, idx;
    vector<Mat> rvecs, tvecs;

    if (CALIBRATED) {
        FileStorage fs(OMNI_MODEL_PATH, FileStorage::READ);
        fs["cameraMatrix"] >> K;
        fs["D"] >> D;
        fs["xi"] >> xi;
        fs["rvecs"] >> rvecs;
        fs["tvecs"] >> tvecs;
    }
    else {
        vector<vector<Point3f> > objectPoints(1);
        vector<vector<Point2f> > imagePoints;
        cv::Size imageSize;

        Size boardSize;
        boardSize.width = 6;
        boardSize.height = 9;
        float squareSize = 50;

        Mat cameraMatrix, distCoeffs, _rvecs, _tvecs;
        cameraMatrix = Mat::eye(3, 3, CV_64F);
        cameraMatrix.at<double>(0, 0) = 1;
        distCoeffs = Mat::zeros(4, 1, CV_64F);

        getImagePoints(imagePoints, imageSize, imageList, boardSize);
        getObjectPoints(objectPoints, boardSize, squareSize);
        objectPoints.resize(imagePoints.size(), objectPoints[0]);

        int flags = 0;
        TermCriteria criteria(TermCriteria::COUNT + TermCriteria::EPS, 200, 0.0001);

        double rms = omnidir::calibrate(objectPoints, imagePoints, imageSize, K, xi, D, rvecs, tvecs, flags, criteria, idx);

        FileStorage fs(OMNI_MODEL_PATH, FileStorage::WRITE);
        fs << "cameraMatrix" << K;
        fs << "D" << D;
        fs << "xi" << xi;
        fs << "rvecs" << rvecs;
        fs << "tvecs" << tvecs;
    }

    if (SHOW_IMAGES) {
        // Transform images to perspective model and display.
        for (int currImage = 0; currImage < imageList.size(); currImage++) {
            cout << imageList[currImage] << endl;
            Mat view, rview;
            view = imread(imageList[currImage], IMREAD_COLOR);

            //Size new_size = Size(384 * RESIZE_FACTOR, 384 * RESIZE_FACTOR);

            //Matx33f Knew = Matx33f(new_size.width / 4, 0, new_size.width / 2,
            //    0, new_size.height / 4, new_size.height / 2,
            //    0, 0, 1);

            omnidir::undistortImage(view, rview, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);
            if (RESIZE_FACTOR != -1) {
                pyrDown(view, view);
                resize(view, view, Size(view.cols / RESIZE_FACTOR, view.rows / RESIZE_FACTOR));
            }

            imshow("Original View", view);
            imshow("Image View", rview);

            imwrite(cv::format("D:/NIFA/new_images_full/Image%d.jpg", currImage), rview);

            waitKey(1000);
        }
    }

    Mat frame, world_coords, rvec, tvec;
    vector<Point2f> pixel_coords;
    vector<Point2d> new_pixel_coords;
    markFiducials(frame, world_coords, pixel_coords);
    estimateCameraPose(frame, world_coords, pixel_coords, K, D, rvec, tvec, new_pixel_coords);

    Point3f start_v = Point3f(44.1285, -92.294, 0);
    Point3f end_v = Point3f(44.1305, -92.294, 0);
    //Point3f start_h = Point3f(44.12905, -92.291, 0);
    //Point3f end_h = Point3f(44.12905, -92.297, 0);
    Point3f start_h = Point3f(44.12905, -92.293, 0);
    Point3f end_h = Point3f(44.12905, -92.295, 0);
    vector<Point3f> line_coords;
    generateLineCoordinates(start_v, end_v, 11, line_coords);
    generateLineCoordinates(start_h, end_h, 11, line_coords);
    vector<Point2f> new_line_coords;
    projectPoints(line_coords, rvec, tvec, K, D, new_line_coords);
    
    for (int i = 0; i < new_line_coords.size(); i++) {
        circle(frame, new_line_coords[i], 3, Scalar(0, 255, 0), -1);
    }
    for (int i = 0; i < new_pixel_coords.size(); i++) {
        circle(frame, new_pixel_coords[i], 3, Scalar(0, 0, 255), -1);
    }

    omnidir::undistortImage(frame, frame, K, D, xi, omnidir::RECTIFY_PERSPECTIVE);
    if (RESIZE_FACTOR != -1) {
        pyrDown(frame, frame);
        resize(frame, frame, Size(frame.cols / RESIZE_FACTOR, frame.rows / RESIZE_FACTOR));
    }

    imshow("Reprojected w/ fiducials", frame);
    while (true) {
        waitKey(1);
    }

	return 0;
}