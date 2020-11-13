#include "calib.h"
#include "ocam_utils.h"
#include "eigen_aliases.h"
#include "lambda_twist.h"
#include "transform_utils.h"
#include "shadow_detection.h"
#include "ShadowMapIO.hpp"
#include "FRF.cpp"

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
    //cap.set(CAP_PROP_POS_FRAMES, 160 * 30);
    int num_frames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    int frame_rate = static_cast<int>(round(cap.get(CAP_PROP_FPS)));
    int num_layers = ceil(num_frames / frame_rate);
    cout << num_frames << " frames." << endl;
    cout << frame_rate << " FPS." << endl;
    cout << num_layers << " layers." << endl;

    Mat frame, img_rot, H, hsv, brightest, binary, img_rot_sampled, binary_sampled;
    vector<Mat> hsv_channels;

    cvtColor(img_ref, hsv, COLOR_BGR2HSV);
    split(hsv, hsv_channels);
    brightest = hsv_channels[2];

    VideoWriter video;
    video.open(VIDEO_SAVE_PATH, VideoWriter::fourcc('M', 'J', 'P', 'G'), OUTPUT_FPS, Size(OUTPUT_RESOLUTION_PX, OUTPUT_RESOLUTION_PX), true);

    int	count = 0;
    int layer_num = 0;

    FRFImage shadowMap; //Create new FRF file

    //Set Image dimensions - this must be done now, when there are no layers in the image yet.
    if (!shadowMap.SetWidth(512U))
    std::cerr << "Failed to set image width.\r\n";
    if (!shadowMap.SetHeight(512U))
    std::cerr << "Failed to set image height.\r\n";

    ShadowMapInfoBlock myShadowMapInfoBlock;
    myShadowMapInfoBlock.FileTimeEpoch_Week = 0U;
    myShadowMapInfoBlock.FileTimeEpoch_TOW = std::nan("");

    Eigen::Vector2d UL(0, 0);
    Eigen::Vector2d UR(0, 511);
    Eigen::Vector2d LR(511, 511);
    Eigen::Vector2d LL(511, 0);
    Eigen::Vector3d UL_LLA, LR_LLA, UR_LLA, LL_LLA;
    positionPX2LLA(img_ref, UL, centroid_ECEF, en_extent, OUTPUT_RESOLUTION_PX, UL_LLA, R_cam_ENU, t_cam_ENU, o);
    positionPX2LLA(img_ref, LL, centroid_ECEF, en_extent, OUTPUT_RESOLUTION_PX, LL_LLA, R_cam_ENU, t_cam_ENU, o);
    positionPX2LLA(img_ref, LR, centroid_ECEF, en_extent, OUTPUT_RESOLUTION_PX, LR_LLA, R_cam_ENU, t_cam_ENU, o);
    positionPX2LLA(img_ref, UR, centroid_ECEF, en_extent, OUTPUT_RESOLUTION_PX, UR_LLA, R_cam_ENU, t_cam_ENU, o);

    imshow("Ref", img_ref);

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
            //video << binary_sampled;
            cvtColor(binary_sampled, binary_sampled, COLOR_RGB2GRAY);
            imshow("Frame", binary_sampled);
            imshow("Frame 2", img_rot_sampled);

		    //Add a new layer and set it up
		    FRFLayer* newLayer = shadowMap.AddLayer();
		    newLayer->Name = std::string("Shadow Map Layer");
		    newLayer->Description = std::string("0 = Unshadowed, 1 = Fully shadowed");
		    newLayer->UnitsCode = -1; //No units
		    newLayer->SetTypeCode(8U); //See Table 1 in the spec. We are going to use 8-bit unsigned integers for each pixel in this layer
		    newLayer->HasValidityMask = true; //Add validity info for each pixel in this layer
		    newLayer->SetAlphaAndBetaForGivenRange(0.0, 1.0); //Let the FRF lib set coefficients so values are in range [0,1]
		    newLayer->AllocateStorage(); //This needs to be called before the layer can be accessed

            for (uint32_t row = 0U; row < shadowMap.Rows(); row++) {
                for (uint32_t col = 0U; col < shadowMap.Cols(); col++) {
                    newLayer->SetValue(row, col, (int) binary_sampled.at<uchar>(row, col));
                }
            }
            myShadowMapInfoBlock.LayerTimeTags.push_back(layer_num++);
            waitKey(1);
        }
    }

    FRFVisualizationColormap* viz = shadowMap.AddVisualizationColormap();
    viz->LayerIndex = 0U; //Base the visualization on the first layer of the shadow map
    viz->SetPoints.push_back(std::make_tuple(0.0, 1.0, 1.0, 1.0)); //Map value 0 (unshadowed) to white (RGB all set to 1)
    viz->SetPoints.push_back(std::make_tuple(1.0, 0.0, 0.0, 0.0)); //Map value 1 (fully shadowed) to black (RGB all set to 0)

    //Make a Geo-Registration tag. This is only possible if the imagery is registered and it is not strictly required to create a valid shadow map file.
    //We will provide the GPS coordinates of the 4 corners of the image and let the FRF library fill in the block for us.
    FRFGeoRegistration GeoRegistrationTag;
    GeoRegistrationTag.Altitude = std::nan(""); //Raster layer is associated with the Earth's surface
    Eigen::Vector2d UL_LL(UL(0), UL(1));
    Eigen::Vector2d UR_LL(UR(1), UR(2));
    Eigen::Vector2d LL_LL(LL(0), LL(1));
    Eigen::Vector2d LR_LL(LR(1), LR(2));

    GeoRegistrationTag.RegisterFromCornerLocations(UL_LL, UR_LL, LL_LL, LR_LL);
    shadowMap.SetGeoRegistration(GeoRegistrationTag);

    if (!myShadowMapInfoBlock.AttachToFRFFile(shadowMap))
        std::cerr << "Error adding shadow map info block... do we have the right number of time tags? There should be 1 per layer.\r\n";

    //Save the shadow map file to disk
    shadowMap.SaveToDisk("E:/NIFA/datasets/frf/test.frf");
    LoadImageAndInspectShadowMapFile("E:/NIFA/datasets/frf/test.frf");

    return 0;
}