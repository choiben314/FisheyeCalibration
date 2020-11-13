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
    Mat img_ref, fiducials_LLA, fiducials_LEA, fiducials_REPROJ;
    vector<Point2d> fiducials_PX;
    Point3d centroid_ECEF;
    double centroid_alt;

    struct ocam_model o;
    const char* file = "./calib_results.txt";

    Eigen::Matrix3d R_cam_LEA, R_cam_ENU;
    Eigen::Vector3d t_cam_LEA, t_cam_ENU;

    // Camera calibration

    getRegistrationFrame(img_ref);
    markFiducials(img_ref, fiducials_LLA, fiducials_PX);
    LLA2Radians(fiducials_LLA);

    getECEFCentroid(fiducials_LLA, centroid_ECEF);
    multLLA2LEA(fiducials_LLA, fiducials_LEA, centroid_ECEF);

    get_ocam_model(&o, file);
    findPose(fiducials_PX, fiducials_LEA, o, R_cam_LEA, t_cam_LEA);
    poseLEA2ENU(centroid_ECEF, R_cam_LEA, t_cam_LEA, R_cam_ENU, t_cam_ENU);

    double en_extent = get_en_extent(APERTURE_DISTANCE_PX, centroid_ECEF, t_cam_LEA, o);

    // Reprojection validation

    multENU2CAM(fiducials_LEA, fiducials_REPROJ, R_cam_LEA, t_cam_LEA, o);
    reprojectExperiments(img_ref, fiducials_PX, fiducials_LEA, fiducials_REPROJ, R_cam_LEA, t_cam_LEA, o);

    // Input video and shadow detection
    VideoCapture cap(GCP_FRAME_PATH);
    if (!cap.isOpened())
        return -1;

    int num_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int frame_rate = static_cast<int>(round(cap.get(CAP_PROP_FPS)));
    cout << num_frames << " frames." << endl;
    cout << frame_rate << " FPS." << endl;

    Mat frame, img_rot, H, hsv, brightest, binary, img_rot_sampled, binary_sampled;
    vector<Mat> hsv_channels;

    cvtColor(img_ref, hsv, COLOR_BGR2HSV);
    split(hsv, hsv_channels);
    brightest = hsv_channels[2];

    VideoWriter video;
    video.open(VIDEO_SAVE_PATH, VideoWriter::fourcc('M', 'J', 'P', 'G'), OUTPUT_FPS, Size(OUTPUT_RESOLUTION_PX, OUTPUT_RESOLUTION_PX), true);

    int	count = 0;
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
            medianBlur(binary, binary, MEDIAN_BLUR_RADIUS);

            cvtColor(binary, binary, COLOR_GRAY2RGB);

            sampleENUSquare(img_rot, o, R_cam_ENU, t_cam_ENU, en_extent, OUTPUT_RESOLUTION_PX, false, img_rot_sampled);
            sampleENUSquare(binary, o, R_cam_ENU, t_cam_ENU, en_extent, OUTPUT_RESOLUTION_PX, false, binary_sampled);

            imshow("Frame", binary_sampled);
            imshow("Frame 2", img_rot_sampled);
            video << binary_sampled;
            waitKey(1);
        }
    }
    return 0;
}