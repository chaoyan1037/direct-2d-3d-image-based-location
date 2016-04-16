
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#include <opencv/cv.h>
#include <opencv2/calib3d.hpp>

#include "geometry.h"
#include "Timer/timer.h"
#include "Epnp/epnp.h"

//estimate camera pose from 2d-3d correspondence
//both DLT method and ePnP method implemented

using namespace std;
using namespace cv;

//#define TSET_GEOMETRY

void Geometry::SetIntrinsicParameter(float f, int u, int v){
	K(0, 0) = f;	K(0, 1) = 0.0;	K(0, 2) = u;
	K(1, 0) = 0.0;	K(1, 1) = f;	K(1, 2) = v;
	K(2, 0) = 0.0;	K(2, 1) = 0.0;	K(2, 2) = 1.0;
	
	K_Inv = K.inv();

#ifdef TSET_GEOMETRY
	cout <<"K_Inv: " <<endl<<K_Inv << endl;
	cv::Matx33d	K1, K2;
	K1(0, 0) = 1.0/f;	K1(0, 1) = 0.0;		K1(0, 2) = 0.0;
	K1(1, 0) = 0.0;		K1(1, 1) = 1.0/f;	K1(1, 2) = 0.0;
	K1(2, 0) = 0.0;		K1(2, 1) = 0.0;		K1(2, 2) = 1.0;

	K2(0, 0) = 1.0;		K2(0, 1) = 0.0;		K2(0, 2) = -u;
	K2(1, 0) = 0.0;		K2(1, 1) = 1.0;		K2(1, 2) = -v;
	K2(2, 0) = 0.0;		K2(2, 1) = 0.0;		K2(2, 2) = 1.0;

	K_Inv = K1*K2;
	cout << "K_Inv: "<<endl<<K_Inv << endl;
#endif
}

void Geometry::SetK(const cv::Matx33d& Ki){
	K = Ki;
	K_Inv = K.inv();
}

//input projection matrix P and 2d image pixel points - 3d Euclidean coordinates match
//only for a single 2d-3d match pair
//return the squared pixel coordinate error
double Geometry::ComputeReprojectionError(const cv::Matx34d& P, const std::pair<cv::Vec2d, cv::Vec3d>& match)
{
	cv::Vec4d	point;
	point[0] = match.second[0];
	point[1] = match.second[1];
	point[2] = match.second[2];
	point[3] = 1;

	cv::Vec3d	reprojection = P*point;
	
	//it is near 0.0
	if (std::fabs(reprojection[2]) < 1e-12){
		return 99999.0;
	}
	else{
		reprojection[0] /= reprojection[2];
		reprojection[1] /= reprojection[2];
	}

	return  (reprojection[0] - match.first[0])*(reprojection[0] - match.first[0]) +
			(reprojection[1] - match.first[1])*(reprojection[1] - match.first[1]); 
}

//compute pose and return the number of inlier and inlier mask
//epnp need the K parameter
int Geometry::ComputePoseEPnP(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	cv::Matx33d &R, cv::Vec3d &T, std::vector<bool>& binlier){
	//init the rand number generate
	srand(int(time(0)));

	int inlier_num = 0;
	int inlier_num_best = 0;
	int index[12];
	int index_best[12];
	int match_num = (int)match_2d_3d.size();
	int inlier_num_thres = (match_num+1) >> 1;

	std::cout << "match_num when compute pose is: " << match_num << std::endl;
	if (match_num < 12){ return 0; std::cout << "not enough match_num" << std::endl; }

	epnp PnP;
	PnP.set_internal_parameters(K(0, 2), K(1, 2), K(0, 0), K(1, 1));
	PnP.set_maximum_number_of_correspondences(match_num);

	
	cv::Matx34d P, P_inlier;
	int prosac_time = 10;
	int stop = 0;

#pragma omp parallel for shared(prosac_time, stop, inlier_num_best, index_best)
	for (int RANSACnum = 0; RANSACnum < 4000; RANSACnum++){
		if (stop) {continue;}
		//every time generate 5 pair
		for (int j = 0; j < 5; j++){
			int n = rand()%prosac_time; 
			bool isRepeated = 0;
			for (int k = 0; k < j; k++){
				if (index[k] == n){
					isRepeated = 1;
					break;
				}
			}
			if (isRepeated){ --j; }
			else { index[j] = n; }
		}
		if (prosac_time < match_num){
#pragma omp atomic
			prosac_time++;
		}

		PnP.reset_correspondences();
		for (int i = 0; i < 5; i++){
			double Xw, Yw, Zw, u, v;
			Xw = match_2d_3d[index[i]].second[0];
			Yw = match_2d_3d[index[i]].second[1];
			Zw = match_2d_3d[index[i]].second[2];
			u  = match_2d_3d[index[i]].first[0];
			v  = match_2d_3d[index[i]].first[1];
			PnP.add_correspondence(Xw, Yw, Zw, u, v);
		}
		double R_est[3][3], T_est[3];
		PnP.compute_pose(R_est, T_est);
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				R(i, j) = R_est[i][j];
			}
			T[i] = T_est[i];
		}

		cv::Matx33d KR(K*R);
		cv::Vec3d	KT(K*T);
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				P(i, j) = KR(i, j);
			}
			P(i, 3) = KT[i];
		}
		//cout << P << endl;

		inlier_num = 0;
		//check if it is an inlier
		for (auto& _match : match_2d_3d){
			if (ComputeReprojectionError(P, _match) < 10.0){
				inlier_num++;
			}
		}

		//check better inlier P 
		if (inlier_num > inlier_num_best){
#pragma omp critical
			{
				if (inlier_num > inlier_num_best)
				{
					inlier_num_best = inlier_num;
					P_inlier = P;
					for (int i = 0; i < 5; i++){
						index_best[i] = index[i];
					}
				}
			}
		}
		//check if stop RANSAC
		if (inlier_num > 100 || inlier_num > inlier_num_thres&&inlier_num >= 12){
#pragma omp atomic
			stop++;
		}
	}

	vector<int> inlier_match_index_list;
	vector<bool> bInlier_(match_num, 0);
	int inlier_num_last = 0, inlier_num_cur = 0;
	double err2 = 0.0;
	//refine the inlier
	while (1)
	{
		inlier_match_index_list.clear();
		//find inlier
		for (int i = 0; i < match_num; i++){
			bInlier_[i] = 0;
			if (ComputeReprojectionError(P_inlier, match_2d_3d[i]) < 10.0){
				inlier_match_index_list.push_back(i);
				bInlier_[i] = 1;
			}
		}
		inlier_num_cur = (int)inlier_match_index_list.size();
		//no better then stop
		if (inlier_num_cur <= inlier_num_last){ break;}
		//otherwise refine the R T
		PnP.reset_correspondences();
		for (int i = 0; i < inlier_num_cur; i++){
			double Xw, Yw, Zw, u, v;
			Xw = match_2d_3d[inlier_match_index_list[i]].second[0];
			Yw = match_2d_3d[inlier_match_index_list[i]].second[1];
			Zw = match_2d_3d[inlier_match_index_list[i]].second[2];
			u  = match_2d_3d[inlier_match_index_list[i]].first[0];
			v  = match_2d_3d[inlier_match_index_list[i]].first[1];
			PnP.add_correspondence(Xw, Yw, Zw, u, v);
		}

		double R_est[3][3], T_est[3];
		err2 = PnP.compute_pose(R_est, T_est);
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				R(i, j) = R_est[i][j];
			}
			T[i] = T_est[i];
		}

		cv::Matx33d KR(K*R);
		cv::Vec3d KT(K*T);
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				P_inlier(i, j) = KR(i,j);
			}
			P_inlier(i,3) = KT[i];
		}
		
		inlier_num_last = inlier_num_cur;
		binlier = bInlier_;
	}
	std::cout << "epnp inlier match num: " << inlier_num_last << std::endl;
	std::cout << "epnp reprojection error: " << err2 << std::endl;
	return inlier_num_last;
}

//Compute the pose use DLT
int Geometry::ComputePoseDLT(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	cv::Matx33d &R, cv::Vec3d &T, std::vector<bool>& binlier, cv::Matx33d & K_estimated){
	//init the rand number generate
	srand(int(time(0)));

	int match_num = (int)match_2d_3d.size();
	int inlier_num_thres = (match_num + 1) >> 1;

	std::cout << "match_num when compute pose is: " << match_num << std::endl;
	if (match_num < 12){ return 0; std::cout << "not enough match_num" << std::endl; }

	cv::Matx34d P, P_inlier;
	
	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> match_2d_3d_normalized;
	cv::Matx33d	mat_2d_scaling_inv;
	cv::Matx44d mat_3d_scaling;
	bool bNormalized = 0;

	bNormalized = Normalize(match_2d_3d, match_2d_3d_normalized, mat_2d_scaling_inv, mat_3d_scaling);

	int prosac_time = 12<match_num?12:match_num;
	int inlier_num_best = 0;
	int stop = 0;
	
#pragma omp parallel for shared(stop, prosac_time, inlier_num_best, P_inlier)
	for (int RANSACnum = 0; RANSACnum < 4000; RANSACnum++){
		if (stop) { continue; }

		int index[16];
		//DLT: every time generate 6 pair
		for (int j = 0; j < 6; j++){
			int n = rand() % prosac_time;
			bool isRepeated = 0;
			for (int k = 0; k < j; k++){
				if (index[k] == n){
					isRepeated = 1;
					break;
				}
			}
			if (isRepeated){ --j; }
			else { index[j] = n; 
			if (n >= match_num - 1)
				cout << "n: " << n << "match: " << match_num << endl;
				stop++;
			}
		}
		if (prosac_time < match_num){
#pragma omp atomic
			prosac_time++;
		}
		//cout << " after select seed" << endl;

		std::vector<std::pair<cv::Vec2d, cv::Vec3d>> minimal_set;
		minimal_set.clear();
		if (bNormalized){
			for (int i = 0; i < 6; i++){
				minimal_set.push_back(match_2d_3d_normalized[index[i]]);
			}
		}
		else
		{
			for (int i = 0; i < 6; i++){
				minimal_set.push_back(match_2d_3d[index[i]]);
			}
		}
		//cout << " after select minimal set" << endl;
		//cout << " minimal set size: " << minimal_set.size() << endl;
		//compute P
		if (6 != minimal_set.size()){ cout << " minimal set size: " << minimal_set.size() << endl; }
		if (CM_Compute(minimal_set, P) == 0){ continue; }
		if (bNormalized){
			P = mat_2d_scaling_inv*P*mat_3d_scaling;
		}
		//cout << " after s CM_Compute" << endl;
		int inlier_num = 0;
		//count inlier number
		for (int i = 0; i < match_num; i++){
			if (ComputeReprojectionError(P, match_2d_3d[i]) < 10.0){
				inlier_num++;
			}
		}
		//cout << " after count inlier" << endl;

		//update the best inlier num
		if (inlier_num > inlier_num_best){
#pragma omp critical
			if (inlier_num > inlier_num_best){
				inlier_num_best = inlier_num;
				P_inlier = P;
			}
		}
		//cout << " after update best inlier" << endl;
		//stop?
		if (inlier_num > 100 || inlier_num > inlier_num_thres&&inlier_num > 12){
			stop++;
		}
	}

	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> inlier_match_list;
	vector<bool> bInlier_(match_num, 0);
	int inlier_num_last = 0, inlier_num_cur = 0;
	
	//refine the inlier
	while (1)
	{
		inlier_match_list.clear();
		if (bNormalized){
			for (int i = 0; i < match_num; i++){
				bInlier_[i] = 0;
				if (ComputeReprojectionError(P_inlier, match_2d_3d[i]) < 10.0){
					inlier_match_list.push_back(match_2d_3d_normalized[i]);
					bInlier_[i] = 1;
				}
			}
		}
		else{
			for (int i = 0; i < match_num; i++){
				bInlier_[i] = 0;
				if (ComputeReprojectionError(P_inlier, match_2d_3d[i]) < 10.0){
					inlier_match_list.push_back(match_2d_3d[i]);
					bInlier_[i] = 1;
				}
			}
		}
		
		inlier_num_cur = (int)inlier_match_list.size();
		//no better then stop
		if (inlier_num_cur <= inlier_num_last){ break; }
		
		//otherwise refine the R T
		if (0 == CM_Compute(inlier_match_list, P)){ break; }

		if (bNormalized){
			P = mat_2d_scaling_inv*P*mat_3d_scaling;
		}
		P_inlier = P;
		inlier_num_last = inlier_num_cur;
		binlier = bInlier_;
	}

	std::cout << "Inlier match num: " << inlier_num_last << std::endl;
	if (inlier_num_last < 6){
		std::cout << "not enough inlier" << std::endl;
		return 0;
	}

	P_inlier = P;
	cv::Matx33d KR, K_RQ, R_RQ;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++)
			KR(i, j) = P(i, j);
	}
	if (M3Det(KR) < 0){
		//cout << "||KR|| < 0, P: " << P << endl;
		//KR = -1.0*KR;
		P = -P;
		//P_inlier = -P_inlier;
		//cout << "P=-P, P: " << P << endl;
		//cout << "less than 0" << endl;
	}
	//cout << "KR :" <<endl<< KR << endl;
	//cv::RQDecomp3x3(KR, K_RQ, R_RQ);
	//P = (1.0 / K_RQ(2, 2))*P;
	//K_RQ = (1.0 / K_RQ(2, 2))*K_RQ;
	//cout << "K_RQ: " << endl << K_RQ << endl;
	//cout << "R_RQ: " << endl << R_RQ << endl;
	//cv::Matx31d T_RQ = K_RQ.inv()*(P.col(3));
	//cout << "T_RQ: " << T_RQ.t() << endl;

	//P = P_inlier;
	cv::Vec4d	camera_position;
	// The function computes a decomposition of a projection matrix into
	// a calibration  and a rotation matrix and the position of a camera.
	cv::decomposeProjectionMatrix(P, K_estimated, R, camera_position);
	//P = 1.0 / K_estimated(2, 2)*P;
	K_estimated = 1.0 / K_estimated(2,2)*K_estimated;
	//std::cout << "estimated K is: " <<endl<< K_estimated << std::endl;
	//cout << "determinent of K: " << M3Det(K_estimated) << endl;

	camera_position = (1.0 / camera_position[3])*camera_position;
	cv::Matx31d t;
	t(0, 0) = camera_position[0];
	t(1, 0) = camera_position[1];
	t(2, 0) = camera_position[2];
	// P = R*X+t  convert X in the world-coordinate into the P in the camera coordiante
	// for camera center P = 0 = R*X+t£¬ then X = -R*t is the center is world coordinate;
	t = -R*t;

	//cout << "translation: " << t.t() << endl;
	//t = K_estimated.inv()*P.col(3);

	T[0] = t(0, 0);
	T[1] = t(1, 0);
	T[2] = t(2, 0);
	//cout << "estimate t: " << T << endl;

	return inlier_num_last;
}

//calculated P is the right null space unit vector corresponding to the A's minial sigular value 
bool Geometry::CM_Compute(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	cv::Matx34d& P){
	//cout << "enter CM_Compute" << endl;

	int row_num = (int)match_2d_3d.size();
	if (row_num < 6){
		std::cout << "match_2d_3d num is less than 6. geometry.cpp line 442" << endl; 
		return 0;
	}

	cv::Mat A(row_num * 3, 12, CV_64FC1);

	for (int i = 0; i < row_num; i++){
		double x, y, X, Y, Z;
		x = match_2d_3d[i].first[0];
		y = match_2d_3d[i].first[1];
		X = match_2d_3d[i].second[0];
		Y = match_2d_3d[i].second[1];
		Z = match_2d_3d[i].second[2];
		
		double* ptrA = A.ptr<double>(3 * i);
		ptrA[0] = 0.0;	ptrA[1] = 0.0;	ptrA[2] = 0.0;	ptrA[3] = 0.0; 
		ptrA[4] = -X;	ptrA[5] = -Y;	ptrA[6] = -Z;	ptrA[7] = -1.0;
		ptrA[8] = y*X;	ptrA[9] = y*Y;	ptrA[10] = y*Z; ptrA[11] = y;
		
		ptrA = A.ptr<double>(3 * i + 1);
		ptrA[0] = X;	ptrA[1] = Y;	ptrA[2] = Z;	ptrA[3] = 1.0;
		ptrA[4] = 0.0;	ptrA[5] = 0.0;	ptrA[6] = 0.0;	ptrA[7] = 0.0;
		ptrA[8] = -x*X;	ptrA[9] = -x*Y;	ptrA[10] = -x*Z;ptrA[11] = -x;

		ptrA = A.ptr<double>(3 * i + 2);
		ptrA[0] = -y*X;	ptrA[1] = -y*Y;	ptrA[2] = -y*Z;	ptrA[3] = -y;
		ptrA[4] = x*X;	ptrA[5] = x*Y;	ptrA[6] = x*Z;	ptrA[7] = x;
		ptrA[8] = 0.0;	ptrA[9] = 0.0;	ptrA[10] = 0.0; ptrA[11] = 0.0;
	}
	//cout << " after construct A " << endl;

	cv::Mat	w, u, vt;
	cv::SVD::compute(A, w, u, vt);

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 4; j++){
			P(i, j) = vt.ptr<double>(11)[4 * i + j];
		}
	}
	//cout << "leave CM_Compute" << endl;
#ifdef TSET_GEOMETRY
	cout << "vt last row: " << vt.row(11) << endl;
	cout << "geo calculated P: " << P << endl;
#endif // TSET_GEOMETRY

	return 1;
}

//normalize when use DLT method
//mat_2d convert normalized 2d points to original pixel points, point2d_pixel = mat2d*point2d_normalized;
//mat_3d convert original 3d points to the normalized 3d points,  
bool Geometry::Normalize(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d_normalized,
	cv::Matx33d& mat_2d_inv, cv::Matx44d& mat_3d)
{
	cv::Vec2d pt1, center_2d(0.0, 0.0);
	cv::Vec3d pt2, center_3d(0.0, 0.0, 0.0);
	
	double n = 1.0;
	//find the point mass center
	for (size_t i = 0; i < match_2d_3d.size(); i++){
		center_2d = match_2d_3d[i].first*(1.0 / n) + center_2d*((n - 1.0) / n);
		center_3d = match_2d_3d[i].second*(1.0 / n) + center_3d*((n - 1.0) / n);
		n += 1.0;
	}

	match_2d_3d_normalized.clear();
	double scale1=0.0, scale2=0.0;
	n = 1.0;
	//normalize
	for (size_t i = 0; i < match_2d_3d.size(); i++){
		pt1 = match_2d_3d[i].first - center_2d;
		pt2 = match_2d_3d[i].second - center_3d;
		scale1 = scale1*((n - 1.0) / n) + cv::norm(pt1)*(1.0 / n);
		scale2 = scale2*((n - 1.0) / n) + cv::norm(pt2)*(1.0 / n);
		match_2d_3d_normalized.push_back(std::pair<cv::Vec2d, cv::Vec3d>(pt1, pt2));
		n += 1.0;
	}
	if (fabs(scale1) < 1e-12 || fabs(scale2) < 1e-12){
		return false;
	}
	scale1 = 1.41421356 / scale1;
	scale2 = 1.73205080 / scale2;

	for (size_t i = 0; i < match_2d_3d_normalized.size(); i++){
		match_2d_3d_normalized[i].first = match_2d_3d_normalized[i].first*scale1;
		match_2d_3d_normalized[i].second = match_2d_3d_normalized[i].second*scale2;
	}
	 
	auto &mat_2d = mat_2d_inv;
	mat_2d = mat_2d.zeros();
	mat_3d = mat_3d.zeros();

	mat_2d(0, 0) = 1.0 / scale1;	mat_2d(0, 1) = 0.0;			mat_2d(0, 2) = center_2d[0];
	mat_2d(1, 0) = 0.0;			mat_2d(1, 1) = 1.0 / scale1;	mat_2d(1, 2) = center_2d[1];
	mat_2d(2, 0) = 0.0;			mat_2d(2, 1) = 0.0;				mat_2d(2, 2) = 1.0;

	mat_3d(0, 0) = scale2;	mat_3d(0, 1) = 0.0;	mat_3d(0, 2) = 0.0;	mat_3d(0, 3) = -scale2*center_3d[0];
	mat_3d(1, 0) = 0.0;	mat_3d(1, 1) = scale2;	mat_3d(1, 2) = 0.0;	mat_3d(1, 3) = -scale2*center_3d[1];
	mat_3d(2, 0) = 0.0;	mat_3d(2, 1) = 0.0;	mat_3d(2, 2) = scale2;	mat_3d(2, 3) = -scale2*center_3d[2];
	mat_3d(3, 0) = 0.0;	mat_3d(3, 1) = 0.0;	mat_3d(3, 2) = 0.0;		mat_3d(3, 3) = 1.0;

	return 1;
}


//////////////////////////////////////////////////////////////////////////
//for test
const double uc = 320;
const double vc = 240;
const double fu = 800;
const double fv = 800;
const int n = 100;
const double noise = 3.0;

double rand(double min, double max)
{
	return min + (max - min) * double(rand()) / RAND_MAX;
}

void random_pose(double R[3][3], double t[3])
{
	const double range = 1;

	double phi = rand(0, range * 3.14159 * 2);
	double theta = rand(0, range * 3.14159);
	double psi = rand(0, range * 3.14159 * 2);

	R[0][0] = cos(psi) * cos(phi) - cos(theta) * sin(phi) * sin(psi);
	R[0][1] = cos(psi) * sin(phi) + cos(theta) * cos(phi) * sin(psi);
	R[0][2] = sin(psi) * sin(theta);

	R[1][0] = -sin(psi) * cos(phi) - cos(theta) * sin(phi) * cos(psi);
	R[1][1] = -sin(psi) * sin(phi) + cos(theta) * cos(phi) * cos(psi);
	R[1][2] = cos(psi) * sin(theta);

	R[2][0] = sin(theta) * sin(phi);
	R[2][1] = -sin(theta) * cos(phi);
	R[2][2] = cos(theta);

	t[0] = 0.0f;
	t[1] = 0.0f;
	t[2] = 6.0f;
}

void random_point(double & Xw, double & Yw, double & Zw)
{
	double theta = rand(0, 3.14159), phi = rand(0, 2 * 3.14159), R = rand(0, +2);

	Xw = sin(theta) * sin(phi) * R;
	Yw = -sin(theta) * cos(phi) * R;
	Zw = cos(theta) * R;
}

void project_with_noise(double R[3][3], double t[3],
	double Xw, double Yw, double Zw,
	double & u, double & v)
{
	
	double Xc = R[0][0] * Xw + R[0][1] * Yw + R[0][2] * Zw + t[0];
	double Yc = R[1][0] * Xw + R[1][1] * Yw + R[1][2] * Zw + t[1];
	double Zc = R[2][0] * Xw + R[2][1] * Yw + R[2][2] * Zw + t[2];

	double nu = rand(-noise, +noise);
	double nv = rand(-noise, +noise);
	u = uc + fu * Xc / Zc + nu;
	v = vc + fv* Yc / Zc + nv;
}

void Geometry::TestGeometry(){
	srand(time(0));
	cout << "test geometry function" << endl;

	SetIntrinsicParameter(fu, uc, vc);

	/********************************************************/
	epnp PnP;
	PnP.set_internal_parameters(uc, vc, fu, fv);
	PnP.set_maximum_number_of_correspondences(n);

	double R_true[3][3], t_true[3];
	random_pose(R_true, t_true);
	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> match_list;

	PnP.reset_correspondences();
	for (int i = 0; i < n; i++) {
		double Xw, Yw, Zw, u, v;
		random_point(Xw, Yw, Zw);
		project_with_noise(R_true, t_true, Xw, Yw, Zw, u, v);
		PnP.add_correspondence(Xw, Yw, Zw, u, v);
		match_list.push_back(make_pair(Vec2d(u, v), Vec3d(Xw, Yw, Zw)));
	}
	double R_est[3][3], t_est[3];
	double err2 = PnP.compute_pose(R_est, t_est); 
	cout << "epnp estimated R: " << endl;
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			cout<<R_est[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<"epnp estimated t: " << endl;
	cout<<t_est[0]<<" "<<t_est[1]<<" "<<t_est[2]<<endl; 

#if 0
	cout << "'True reprojection error':"
		<< PnP.reprojection_error(R_est, t_est) << endl;
	
	cv::Matx34d P;

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			P(i, j) = R_true[i][j];
		}
		P(i, 3) = t_true[i];
	}
	P = K*P;
	cout << " true P: " << P << endl;

	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			P(i, j) = R_est[i][j];
		}
		P(i, 3) = t_est[i];
	}
	P = K*P;
	cout << "calculated P: " << P << endl;

	double err = 0.0;
	for (int i = 0; i < match_list.size(); i++){
		err += sqrt(ComputeReprojectionError(P, match_list[i]));
	}
	err = err / match_list.size();
	cout << "geo calculated err: " << err << endl;
#endif
	/********************************************************/
#if 0
	CM_Compute(match_list, P);
	
	P = (1.0 / P(2, 3))*P;
	P = 6 * P;
	cout << "geo calculated P: " << P << endl;
#endif
	/********************************************************/
#if 0
	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> match_list_normalized;
	cv::Matx33d mat2d;
	cv::Matx44d mat3d;

	Normalize(match_list, match_list_normalized, mat2d, mat3d);
	cout << "original matched" << endl;
	for (auto& e : match_list){
		cout << e.first[0] << " " << e.first[1] << " "
			<< e.second[0] << " " << e.second[1] << " " << e.second[2] << endl;
	}

	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> match_n;
	for (auto& e : match_list_normalized){
		Vec3d v3(e.first[0], e.first[1], 1.0);
		v3 = mat2d*v3;
		Vec4d v4(e.second[0], e.second[1], e.second[2], 1.0);
		v4 = mat3d.inv()*v4;
		match_n.push_back(make_pair(Vec2d(v3[0], v3[1]), Vec3d(v4[0], v4[1], v4[2])));
		//match_n.push_back(make_pair(Vec2d(v3[0] / v3[2], v3[1] / v3[2]), Vec3d(v4[0] / v4[3], v4[1] / v4[3], v4[2] / v4[3])));
	}
	cout << " recover matched" << endl;
	for (auto& e : match_n){
		cout << e.first[0] << " " << e.first[1] << " "
			<< e.second[0] << " " << e.second[1] << " " << e.second[2] << endl;
	}
#endif
	/********************************************************/

	
	cv::Matx33d	K_dlt;
	cv::Matx33d	R_dlt;
	cv::Vec3d	t_dlt;
	vector<bool> bInlier;

	//for (int i = 0; i < 1000; i++)
	cout << "DLT inlier: " << ComputePoseDLT(match_list, R_dlt, t_dlt, bInlier, K_dlt) << endl;
	cout << "DLT R: " << R_dlt << endl;
	cout << "DLT t: " << t_dlt << endl;

	/*******************************************************/
#if 0
	cout << "epnp inlier: " << ComputePoseEPnP(match_list, R_dlt, t_dlt, bInlier) << endl;
	cout << "epnp R: " << R_dlt << endl;
	cout << "epnp t: " << t_dlt << endl;
#endif

}