#include "calib.h"
#include "ocam_utils.h"
#include "eigen_aliases.h"
#include "lambda_twist.h"
#include "transform_utils.h"

using namespace std;
using namespace cv;

/*** <MAIN FUNCTION ***/
int main(int argc, char** argv) {

    //takeLivePhotos(LIVE_STREAM_PATH, LIVE_IMAGES_PATH, 0);
    //return 0;
    //vector<string> imageList;

    //Mat K, D, xi;
    //vector<Mat> rvecs, tvecs;

    //getLocationsFile(CALIB_IMAGES_LOCATIONS_PATH, CALIB_IMAGES_PATH, imageList);
    //calibrateOmniCamera(imageList, K, D, xi, rvecs, tvecs);
    //showTransformedImages(imageList, K, D, xi);

    double en_extent = 500;
    double num_pixels = 512;

    struct ocam_model o;
    const char* file = "./calib_results.txt";
    get_ocam_model(&o, file);

    Mat regFrame, resampledFrame, fiducials_LLA, fiducials_LEA;
    vector<Point2f> fiducials_PX;
    vector<Point2d> fiducials_reprojected;
    Point3d centroid_ECEF;

    Eigen::Matrix3d R_cam_LEA, R_cam_ENU;
    Eigen::Vector3d t_cam_LEA, t_cam_ENU;

    getRegistrationFrame(regFrame);
    markFiducials(regFrame, fiducials_LLA, fiducials_PX);
    LLA2Radians(fiducials_LLA);

    getECEFCentroid(fiducials_LLA, centroid_ECEF);
    multLLA2LEA(fiducials_LLA, fiducials_LEA, centroid_ECEF);

    findPose(fiducials_PX, fiducials_LEA, o, R_cam_LEA, t_cam_LEA);
    poseLEA2ENU(centroid_ECEF, R_cam_LEA, t_cam_LEA, R_cam_ENU, t_cam_ENU);

    sampleENUSquare(regFrame, o, R_cam_ENU, t_cam_ENU, en_extent, num_pixels, true, resampledFrame);

    //multENU2CAM(fiducials_LEA, fiducials_reprojected, R_cam_LEA, t_cam_LEA, o);

    waitKey();
    return 0;
}

/*
* TODOS:
* Cleanup code/add coord type suffixes
* RANSAC Lambda**
* Calculate reprojection error (after making fiducials super accurate)
* Generate datasets? At what resolution? 128x128
*/