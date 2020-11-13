#include "calib.h"
#include "ocam_utils.h"
#include "eigen_aliases.h"
#include "lambda_twist.h"
#include "transform_utils.h"
#include "shadow_detection.h"

using namespace std;
using namespace cv;

/*** <MAIN FUNCTION ***/
int main(int argc, char** argv) {

    /*
    * TODOS: 
    * Email Boris about shipping labels + Zipcar reimbursement
    * Convert to FRF and send to Shankar for preliminary pipeline dev
    * Collect ~90 minutes of data when extra batteries get here
    * 
    * Automatic EN extent selection
    *   v = cam2world(aperturePoint_imageCoords);
        theta = acos(v.dot(0,0,1));
        camHeight = t_cam_LEA.norm();
        EN_Extent = camHeight * tan( theta );
    *   
        apertureDist_pixels = 283;
        aperturePoint_imageCoords  =FINAL_SIZE * 0.5 + (apertureDist_pixels, 0);

        1: convert LEA origin to LLA
        2: convert cam center to LLA
        3: height = cam center alt - LEA origin alt
        --> "t_cam_LEA.norm()"
    */


    //takeLivePhotos(LIVE_STREAM_PATH, LIVE_IMAGES_PATH, 0);
    //return 0;
    //vector<string> imageList;

    //Mat K, D, xi;
    //vector<Mat> rvecs, tvecs;

    //getLocationsFile(CALIB_IMAGES_LOCATIONS_PATH, CALIB_IMAGES_PATH, imageList);
    //calibrateOmniCamera(imageList, K, D, xi, rvecs, tvecs);
    //showTransformedImages(imageList, K, D, xi);

    struct ocam_model o;
    const char* file = "./calib_results.txt";
    get_ocam_model(&o, file);

    Mat regFrame, resampledFrame, fiducials_LLA, fiducials_LEA, fiducials_REPROJ;
    vector<Point2d> fiducials_PX;
    Point3d centroid_ECEF;
    double centroid_alt;

    Eigen::Matrix3d R_cam_LEA, R_cam_ENU;
    Eigen::Vector3d t_cam_LEA, t_cam_ENU;

    getRegistrationFrame(regFrame);
    markFiducials(regFrame, fiducials_LLA, fiducials_PX);
    LLA2Radians(fiducials_LLA);

    getECEFCentroid(fiducials_LLA, centroid_ECEF);
    multLLA2LEA(fiducials_LLA, fiducials_LEA, centroid_ECEF);

    findPose(fiducials_PX, fiducials_LEA, o, R_cam_LEA, t_cam_LEA);
    poseLEA2ENU(centroid_ECEF, R_cam_LEA, t_cam_LEA, R_cam_ENU, t_cam_ENU);

    double en_extent = get_en_extent(APERTURE_DISTANCE_PX, centroid_ECEF, t_cam_LEA, o);
    double num_pixels = 512;

    sampleENUSquare(regFrame, o, R_cam_ENU, t_cam_ENU, en_extent, num_pixels, true, resampledFrame);

    multENU2CAM(fiducials_LEA, fiducials_REPROJ, R_cam_LEA, t_cam_LEA, o);

    Mat f_px = Mat(fiducials_PX.size(), 2, CV_64F, fiducials_PX.data());
    Mat f_reproj_error;
    reprojectionErrorIndividual(f_px, fiducials_LEA, R_cam_LEA, t_cam_LEA, o, f_reproj_error);
    Size center = FINAL_SIZE / 2;
    vector<Point2d> dist_error;
    for (int i = 0; i < f_px.rows; i++) {
        double dist = sqrt(pow(center.width - f_px.at<double>(i, 0), 2) + pow(center.width - f_px.at<double>(i, 1), 2));
        double error = sqrt(f_reproj_error.at<double>(i));
        dist_error.push_back(Point2d(dist, error));
    }

    FileStorage fs(GCP_LOCATION + "dist_error.csv", FileStorage::WRITE);
    fs << "dist_error" << dist_error;

    for (int i = 0; i < fiducials_REPROJ.rows; i++) {
        circle(regFrame, Point2d(fiducials_REPROJ.row(i)), 3, Scalar(0, 0, 255), -1);
    }

    imwrite(GCP_LOCATION + "reg_frame.png", regFrame);
    //imshow("Registration frame", resampledFrame);

    VideoCapture cap(GCP_FRAME_PATH);
    if (!cap.isOpened())
        return -1;

    int num_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int frame_rate = static_cast<int>(round(cap.get(CAP_PROP_FPS)));
    cout << num_frames << " frames." << endl;
    cout << frame_rate << " FPS." << endl;

    int	count = 1;

    Mat frame, img_ref, img_rot, H;

    string reference_image = GCP_LOCATION + "DJI_0002.MOV";
    int init_frame = 0;
    VideoCapture ref_cap(reference_image);
    if (!ref_cap.isOpened())
        return -1;
    ref_cap >> img_ref;
    if (img_ref.size() != FINAL_SIZE) {
        downscale(img_ref, img_ref);
    }

    // Extract first frame as reference
    //cap.set(CAP_PROP_POS_FRAMES, INIT_SECOND * frame_rate);
    //cap >> frame;
    //if (frame.size() != FINAL_SIZE) {
    //    downscale(frame, frame);
    //}

    //img_ref = frame.clone();

    Mat hsv, brightest, binary, img_rot_sampled, binary_sampled;
    vector<Mat> hsv_channels;

    cvtColor(img_ref, hsv, COLOR_BGR2HSV);
    split(hsv, hsv_channels);
    brightest = hsv_channels[2];

    VideoWriter video;
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = 10;
    video.open("E:/NIFA/datasets/" + LOCATION + "_" + VIDEO + ".avi", codec, fps, Size(num_pixels, num_pixels), true);

    while (1) {
        cap >> frame;

        if (++count % frame_rate == 1) {
            cout << "Processing frame #" << count << endl;

            if (frame.empty())
                break;

            if (frame.size() != FINAL_SIZE) {
                downscale(frame, frame);
            }

            getSurfRotation(img_ref, frame, H);
            warpAffine(frame, img_rot, H, img_ref.size());

            getBinaryCloudMask(img_rot, brightest, binary);
            medianBlur(binary, binary, 23);

            cvtColor(binary, binary, COLOR_GRAY2RGB);

            //sampleENUSquare(img_rot, o, R_cam_ENU, t_cam_ENU, en_extent, num_pixels, count == 2, img_rot_sampled);
            sampleENUSquare(binary, o, R_cam_ENU, t_cam_ENU, en_extent, num_pixels, false, binary_sampled);

            //Mat row1, row2, canvas;
            //hconcat(frame, frame, row1);
            //hconcat(img_rot_sampled, binary_sampled, row2);
            //vconcat(row1, row2, canvas);

            imshow("Frame", binary_sampled);
            video << binary_sampled;
            waitKey(1);
        }
    }
    return 0;
}