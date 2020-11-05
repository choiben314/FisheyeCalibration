#pragma once

//Compute latitude (radians), longitude (radians), and altitude (height above WGS84 ref. elipsoid, in meters) from ECEF position (in meters)
//Latitude and longitude are also both given with respect to the WGS84 reference elipsoid.
void positionECEF2LLA(Point3d const& Position, double& lat, double& lon, double& alt) {
	double x = Position.x;
	double y = Position.y;
	double z = Position.z;

	//Set constants
	double R_0 = 6378137.0;
	double R_P = 6356752.314;
	double ecc = 0.081819190842621;

	//Calculate longitude (radians)
	lon = atan2(y, x);

	//Compute intermediate values needed for lat and alt
	double eccSquared = ecc * ecc;
	double p = sqrt(x * x + y * y);
	double E = sqrt(R_0 * R_0 - R_P * R_P);
	double F = 54.0 * (R_P * z) * (R_P * z);
	double G = p * p + (1.0 - eccSquared) * z * z - eccSquared * E * E;
	double c = pow(ecc, 4.0) * F * p * p / pow(G, 3.0);
	double s = pow(1.0 + c + sqrt(c * c + 2.0 * c), 1.0 / 3.0);
	double P = (F / (3.0 * G * G)) / ((s + (1.0 / s) + 1.0) * (s + (1.0 / s) + 1.0));
	double Q = sqrt(1.0 + 2.0 * pow(ecc, 4.0) * P);
	double k_1 = -1.0 * P * eccSquared * p / (1.0 + Q);
	double k_2 = 0.5 * R_0 * R_0 * (1.0 + 1.0 / Q);
	double k_3 = -1.0 * P * (1.0 - eccSquared) * z * z / (Q * (1.0 + Q));
	double k_4 = -0.5 * P * p * p;
	double r_0 = k_1 + sqrt(k_2 + k_3 + k_4);
	double k_5 = (p - eccSquared * r_0);
	double U = sqrt(k_5 * k_5 + z * z);
	double V = sqrt(k_5 * k_5 + (1.0 - eccSquared) * z * z);

	double z_0 = (R_P * R_P * z) / (R_0 * V);
	double e_p = (R_0 / R_P) * ecc;

	//Calculate latitude (radians)
	lat = atan((z + z_0 * e_p * e_p) / p);

	//Calculate Altitude (m)
	alt = U * (1.0 - (R_P * R_P / (R_0 * V)));
}

//Return ECEF position corresponding to given lat (rad), lon (rad), and alt (m)
Point3d positionLLA2ECEF(double lat, double lon, double alt) {
	double a = 6378137.0;           //Semi-major axis of reference ellipsoid
	double ecc = 0.081819190842621; //First eccentricity of the reference ellipsoid
	double eccSquared = ecc * ecc;
	double N = a / sqrt(1.0 - eccSquared * sin(lat) * sin(lat));
	double X = (N + alt) * cos(lat) * cos(lon);
	double Y = (N + alt) * cos(lat) * sin(lon);
	double Z = (N * (1 - eccSquared) + alt) * sin(lat);
	return(Point3d(X, Y, Z));
}

//Convert an ECEF position to an LLA vector: <Latitude (radians), Longitude (radians), Altitude (m)>
Point3d positionECEF2LLA(Point3d const& PositionECEF) {
	double lat = 0.0;
	double lon = 0.0;
	double alt = 0.0;
	positionECEF2LLA(PositionECEF, lat, lon, alt);
	return Point3d(lat, lon, alt);
}

//Convert an LLA vector: <Latitude (radians), Longitude (radians), Altitude (m)> to an ECEF position
Point3d positionLLA2ECEF(Point3d const& PositionLLA) {
	return positionLLA2ECEF(PositionLLA.x, PositionLLA.y, PositionLLA.z);
}

Mat latLon_2_C_ECEF_NED(double lat, double lon) {
	//Compute matrix components
	double C11 = -sin(lat) * cos(lon);
	double C12 = -sin(lat) * sin(lon);
	double C13 = cos(lat);
	double C21 = -sin(lon);
	double C22 = cos(lon);
	double C23 = 0.0;
	double C31 = -cos(lat) * cos(lon);
	double C32 = -cos(lat) * sin(lon);
	double C33 = -sin(lat);

	Mat C_ECEF_NED = (Mat_<double>(3, 3) << C11, C12, C13, C21, C22, C23, C31, C32, C33); // every three is one row

	////Populate C_ECEF_NED
	//Eigen::Matrix3d C_ECEF_NED;
	//C_ECEF_NED(0, 0) = C11; C_ECEF_NED(0, 1) = C12; C_ECEF_NED(0, 2) = C13;
	//C_ECEF_NED(1, 0) = C21; C_ECEF_NED(1, 1) = C22; C_ECEF_NED(1, 2) = C23;
	//C_ECEF_NED(2, 0) = C31; C_ECEF_NED(2, 1) = C32; C_ECEF_NED(2, 2) = C33;

	return(C_ECEF_NED);
}

Mat latLon_2_C_NED_ECEF(double lat, double lon) {
	Mat C_ECEF_NED = latLon_2_C_ECEF_NED(lat, lon);
	Mat C_NED_ECEF = C_ECEF_NED.t();
	return(C_NED_ECEF);
}

Mat latLon_2_C_ECEF_ENU(double lat, double lon) {
	Mat C_ECEF_NED = latLon_2_C_ECEF_NED(lat, lon);
	Mat C_NED_ENU = (Mat_<double>(3, 3) << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
	Mat C_ECEF_ENU = C_NED_ENU * C_ECEF_NED;
	return(C_ECEF_ENU);
}

Mat latLon_2_C_ENU_ECEF(double lat, double lon) {
	Mat C_NED_ECEF = latLon_2_C_NED_ECEF(lat, lon);
	Mat C_ENU_NED = (Mat_<double>(3, 3) << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
	Mat C_ENU_ECEF = C_NED_ECEF * C_ENU_NED;
	return(C_ENU_ECEF);
}

void LLA2Radians(Mat& lla) {
	Mat degToRadians = (Mat_<double>(3, 3) << M_PI / 180, 0.0, 0.0, 0.0, M_PI / 180, 0.0, 0.0, 0.0, 1.0);
	lla *= degToRadians;
}

void getECEFCentroid(Mat& mult_lla, Point3d& ECEF_centroid) {
	int rows = mult_lla.rows;
	Eigen::MatrixXd ECEF_points(rows, 3);
	for (int row = 0; row < rows; row++) {
		Point3d ECEF_point = positionLLA2ECEF(mult_lla.at<double>(row, 0), mult_lla.at<double>(row, 1), mult_lla.at<double>(row, 2));
		ECEF_points.row(row) = Eigen::Vector3d(ECEF_point.x, ECEF_point.y, ECEF_point.z);
	}
	Eigen::Vector3d centroid_temp = ECEF_points.colwise().mean();
	ECEF_centroid = Point3d(centroid_temp[0], centroid_temp[1], centroid_temp[2]);
}

void multLLA2ENU(Mat& mult_lla, const Point3d& ECEF_origin, const Mat& C_ECEF_ENU) {
	int rows = mult_lla.rows;
	for (int row = 0; row < rows; row++) {
		Point3d ECEF_point = positionLLA2ECEF(mult_lla.at<double>(row, 0), mult_lla.at<double>(row, 1), mult_lla.at<double>(row, 2));
		Mat ENU_point = (C_ECEF_ENU * Mat(ECEF_point - ECEF_origin)).t();
		ENU_point.copyTo(mult_lla.row(row));
	}
}

void multLLA2LEA(Mat& mult_lla, Mat &mult_lea, const Point3d& ECEF_origin) {
	int rows = mult_lla.rows;
	mult_lla.copyTo(mult_lea);
	for (int row = 0; row < rows; row++) {
		Point3d ECEF_point = positionLLA2ECEF(mult_lla.at<double>(row, 0), mult_lla.at<double>(row, 1), mult_lla.at<double>(row, 2));
		Mat LEA_point = Mat(ECEF_point - ECEF_origin).t();
		LEA_point.copyTo(mult_lea.row(row));
	}
}

void multCam2World(vector<Point2f>& pixel_coords, vector<Point3f>& backprojected, ocam_model& o) {
	for (int i = 0; i < pixel_coords.size(); i++) {
		double point3D[3];
		double point2D[2] = { pixel_coords[i].y, pixel_coords[i].x };
		cam2world(point3D, point2D, &o);
		backprojected.push_back(Point3f(point3D[0], point3D[1], point3D[2]));
	}
}

void cv2EigenF(Mat& src, Eigen::Matrix3d& dst) {
	int rowA = 1;
	int rowB = 10;
	int rowC = 13;
	dst.col(0) << src.at<float>(rowA, 0), src.at<float>(rowA, 1), src.at<float>(rowA, 2);
	dst.col(1) << src.at<float>(rowB, 0), src.at<float>(rowB, 1), src.at<float>(rowB, 2);
	dst.col(2) << src.at<float>(rowC, 0), src.at<float>(rowC, 1), src.at<float>(rowC, 2);
}

void cv2EigenD(Mat& src, Eigen::Matrix3d& dst) {
	int rowA = 1;
	int rowB = 10;
	int rowC = 13;
	dst.col(0) << src.at<double>(rowA, 0), src.at<double>(rowA, 1), src.at<double>(rowA, 2);
	dst.col(1) << src.at<double>(rowB, 0), src.at<double>(rowB, 1), src.at<double>(rowB, 2);
	dst.col(2) << src.at<double>(rowC, 0), src.at<double>(rowC, 1), src.at<double>(rowC, 2);
}

void getPoseInputMatrices(Mat& world, vector<Point3f>& backproj, Eigen::Matrix3d& world_vec, Eigen::Matrix3d& bearing_vec) {
	Mat bp_matrix = Mat(backproj);
	cv2EigenF(bp_matrix, bearing_vec);
	cv2EigenD(world, world_vec);
}

void matToMatrix3d(Mat &in, Eigen::Matrix3d &out) {
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			out(i, j) = in.at<double>(i, j);
		}
	}
}

// each row of enu is a different point
void multENU2CAM(Mat& enu, vector<Point2d>& cam, Eigen::Matrix3d R, Eigen::Vector3d t, ocam_model& o) {
	for (int row = 0; row < enu.rows; row++) {
		Eigen::Vector3d v(enu.at<double>(row, 0), enu.at<double>(row, 1), enu.at<double>(row, 2));
		Eigen::Vector3d v_cam = R.inverse() * (v - t);
		double point3D[3] = { v_cam(0), v_cam(1), v_cam(2) };
		double point2D[2];

		world2cam(point2D, point3D, &o);

		cam.push_back(Point2d(point2D[1], point2D[0]));
	}
}

//Let R be "Radius" of image (actually the extent in both directions in East and North)
//Let N be the number of pixels in the image in vertical and horizontal direction
//Define:
//	NorthBounds = (-R, R)
//	EastBounds  = (-R, R)
//	GSD = 2*R/N

//(col, row)-pixel coords to reference coords
Eigen::Vector2d PixCoordsToRefCoords(Eigen::Vector2d const& PixCoords, double GSD, double N, const Eigen::Vector2d& NorthBounds, const Eigen::Vector2d& EastBounds) {
	return(Eigen::Vector2d(GSD * PixCoords(0) + EastBounds(0), GSD * (double(N - 1) - PixCoords(1)) + NorthBounds(0)));
}

//Reference coords to (col, row)-pixel coords
Eigen::Vector2d refCoordsToPixCoords(Eigen::Vector2d const& RefCoords, double GSD, double N, const Eigen::Vector2d& NorthBounds, const Eigen::Vector2d& EastBounds) {
	return(Eigen::Vector2d((RefCoords(0) - EastBounds(0)) / GSD, double(N - 1) - (RefCoords(1) - NorthBounds(0)) / GSD));
}

void findPose(vector<Point2f>& fiducials_PX, Mat &fiducials_LEA, ocam_model &o, Eigen::Matrix3d &R_cam_LEA, Eigen::Vector3d &t_cam_LEA) {
	vector<Point3f> fiducials_BACKPROJ;
	multCam2World(fiducials_PX, fiducials_BACKPROJ, o);

	Eigen::Matrix3d input_world, input_bearing;

	getPoseInputMatrices(fiducials_LEA, fiducials_BACKPROJ, input_world, input_bearing);

	std::Evector<std::tuple<Eigen::Vector3d, Eigen::Matrix3d>> PossiblePoses;
	LambdaTwistSolve(input_bearing, input_world, PossiblePoses, true);
	cout << endl << "Number of possible poses: " << PossiblePoses.size() << endl;
	std::tuple<Eigen::Vector3d, Eigen::Matrix3d> pose = PossiblePoses[0];
	
	R_cam_LEA = std::get<1>(pose);
	t_cam_LEA = std::get<0>(pose);
}

void poseLEA2ENU(Point3d &centroid_ECEF, Eigen::Matrix3d& R_cam_LEA, Eigen::Vector3d &t_cam_LEA, Eigen::Matrix3d& R_cam_ENU, Eigen::Vector3d& t_cam_ENU) {
	Eigen::Vector3d centroid_vec_ECEF(centroid_ECEF.x, centroid_ECEF.y, centroid_ECEF.z);
	Eigen::Vector3d cam_center_ECEF = t_cam_LEA + centroid_vec_ECEF;
	Point3d cam_center_pt_ECEF(cam_center_ECEF(0), cam_center_ECEF(1), cam_center_ECEF(2));
	Point3d cam_center_LLA = positionECEF2LLA(cam_center_pt_ECEF);
	Point3d origin_enu_LLA = Point3d(cam_center_LLA.x, cam_center_LLA.y, 395); // average this in a sec

	Mat C_ECEF_ENU = latLon_2_C_ECEF_ENU(origin_enu_LLA.x, origin_enu_LLA.y);
	Eigen::Matrix3d C_Matrix3d_ECEF_ENU;
	matToMatrix3d(C_ECEF_ENU, C_Matrix3d_ECEF_ENU);

	Point3d origin_enu_pt_ECEF = positionLLA2ECEF(origin_enu_LLA.x, origin_enu_LLA.y, origin_enu_LLA.z);
	Eigen::Vector3d origin_enu_ECEF = Eigen::Vector3d(origin_enu_pt_ECEF.x, origin_enu_pt_ECEF.y, origin_enu_pt_ECEF.z);

	R_cam_ENU = C_Matrix3d_ECEF_ENU * R_cam_LEA;
	t_cam_ENU = C_Matrix3d_ECEF_ENU * (cam_center_ECEF - origin_enu_ECEF);
}

void sampleENUSquare(Mat &inputFrame, ocam_model& o, Eigen::Matrix3d R, Eigen::Vector3d t, double en_extent, double num_pixels, bool showFrames, Mat &outputFrame) {

	Eigen::Vector2d NorthBounds(-en_extent, en_extent);
	Eigen::Vector2d EastBounds(-en_extent, en_extent);
	double GSD = 2 * en_extent / num_pixels;

	outputFrame = Mat(num_pixels, num_pixels, CV_8UC3);

	Mat frameCopy;
	if (showFrames) {
		inputFrame.copyTo(frameCopy);
	}

	for (int row = 0; row < num_pixels; row++) {
		for (int col = 0; col < num_pixels; col++) {
			Eigen::Vector2d pixel_coords(col, row);
			Eigen::Vector2d ref_coords = PixCoordsToRefCoords(pixel_coords, GSD, num_pixels, NorthBounds, EastBounds);

			Eigen::Vector3d pixel_enu(ref_coords(0), ref_coords(1), 0);

			Eigen::Vector3d pixel_cam = R.inverse() * (pixel_enu - t);
			double point_bearing[3] = { pixel_cam(0), pixel_cam(1), pixel_cam(2) };
			double point_pixel[2];

			world2cam(point_pixel, point_bearing, &o);

			if (showFrames) {
				circle(frameCopy, Point2f(point_pixel[1], point_pixel[0]), 3, Scalar(255, 0, 0), -1);
			}
			
			Point2f sample_point(point_pixel[1], point_pixel[0]);
			outputFrame.at<Vec3b>(row, col) = getColorSubpixHelper(inputFrame, sample_point);
		}
	}

	if (showFrames) {
		imshow("Sampling Region", frameCopy);
		imshow("Resampled", outputFrame);
	}
}