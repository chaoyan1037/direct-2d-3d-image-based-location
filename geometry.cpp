
#include <iostream>
#include <fstream>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <algorithm>

#include <opencv/cv.h>
#include <opencv2/calib3d.hpp>
#include <opencv2/stitching.hpp>
//sparse bundler adjustment include path
#include "sba.h"
#include "TooN/TooN.h"
#include "TooN/se3.h"
#include "TooN/wls.h"

#include "MEstimator.h"
#include "TrackData_3D.h"
#include "geometry.h"
#include "Timer/timer.h"
#include "Epnp/epnp.h"
#include "sba_warper/sba_warper.h"

//estimate camera pose from 2d-3d correspondence
//both DLT method and ePnP method implemented

using std::cout;
using std::endl;

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

//epnp need the K parameter
//compute pose and return the number of inlier and inlier mask 
int Geometry::ComputePoseEPnP(){

	//init the rand number generate
	srand(int(time(0)));

	const int match_num = (int)match_2d_3d.size();
	
	std::cout << "match_num when compute pose is: " << match_num << std::endl;
	if (match_num < 12){ return 0; std::cout << "not enough match_num" << std::endl; }

	//original epnp class use new and delete , when use OpenMp, you must make sure
	//there is no copy or assign operation on epnp. so I modify the original epnp class
	//add the copy assign and copy constructor
	epnp PnP;
	//const double uc, const double vc, const double fu, const double fv
	PnP.set_internal_parameters(K(0, 2), K(1, 2), K(0, 0), K(1, 1));
	PnP.set_maximum_number_of_correspondences(match_num);

	cv::Matx34d P_inlier;
	int stop = 0;
	int inlier_num_best = 0;
	omp_set_num_threads(2);
#pragma omp parallel for shared(P_inlier, stop, inlier_num_best, match_num) firstprivate(PnP)
	for (int RANSACnum = 0; RANSACnum < 4000; RANSACnum++)
	{
		if (stop) {continue;}

		int index[6];
		int prosac_time = (10 + RANSACnum) < match_num ? (10 + RANSACnum) : (match_num);

		//int pid = omp_get_thread_num();
		//std::cout << " RANSAC start: " << pid << std::endl;

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

		//std::cout << " select seed: " << pid << std::endl;

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
		//std::cout << " set correspondence: " << pid << " "<< PnP.get_correspondence_number() << std::endl;
		double R_est[3][3], T_est[3];
		PnP.compute_pose(R_est, T_est);
		//std::cout << " compute_pose: " << pid << std::endl;
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				R(i, j) = R_est[i][j];
			}
			T[i] = T_est[i];
		}


		cv::Matx34d P;
		cv::Matx33d KR(K*R);
		cv::Vec3d	KT(K*T);
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				P(i, j) = KR(i, j);
			}
			P(i, 3) = KT[i];
		}
		//cout << P << endl;

		int inlier_num = 0;
		//count inlier number
		for (int i = 0; i < match_num; i++){
			if (ComputeReprojectionError(P, match_2d_3d[i]) < 10.0){
				inlier_num++;
			}
		}
		//std::cout << "reprojection error:" << pid << std::endl;

		//check better inlier P 
		if (inlier_num > inlier_num_best){
#pragma omp critical
			if (inlier_num > inlier_num_best)
			{
				inlier_num_best = inlier_num;
				P_inlier = P;
			}	
		}

		//check if stop RANSAC
		int inlier_num_thres = (match_num + 1) >> 1;
		if (inlier_num > 100 || inlier_num > inlier_num_thres&&inlier_num >= 12){
#pragma omp atomic
			++stop;
		}
		//std::cout << " one RANSAC end: " << pid <<" "<< RANSACnum << std::endl;
	}

	//std::cout << " all RANSAC end" << std::endl;
	std::vector<int> inlier_match_index_list;
	std::vector<bool> bInlier_(match_num, 0);
	int inlier_num_last = 0, inlier_num_cur = 0;
	double err2 = 0.0;

	binlier.assign(match_num, 0);
	//std::cout << " start refine" << std::endl;
	//refine the inlier
	while (1)
	{
		inlier_match_index_list.clear();
		//find inlier
		bInlier_.assign(match_num, 0);
		for (int i = 0; i < match_num; i++){
			if (ComputeReprojectionError(P_inlier, match_2d_3d[i]) < 10.0){
				inlier_match_index_list.push_back(i);
				bInlier_[i] = 1;
			}
		}
		inlier_num_cur = (int)inlier_match_index_list.size();
		//no better then stop
		if (inlier_num_cur <= inlier_num_last){ break;}

		//otherwise refine the R T
		assert(inlier_num_cur <= match_num);
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
		binlier.swap(bInlier_);
	}
	//std::cout << "epnp inlier match num: " << inlier_num_last << std::endl;
	//std::cout << "epnp reprojection error: " << err2 << std::endl;
	return inlier_num_last;
}

//Compute the pose use DLT
int Geometry::ComputePoseDLT(){
	//init the rand number generate
	srand(int(time(0)));

	const int match_num = (int)match_2d_3d.size();
	
	std::cout << "match_num when compute pose is: " << match_num << std::endl;
	if (match_num < 12){ return 0; std::cout << "not enough match_num " << std::endl; }

	cv::Matx34d P_inlier;

	bool bNormalized = Normalize();

	int inlier_num_best = 0;
	int stop = 0;
	
#pragma omp parallel for shared(stop, inlier_num_best, P_inlier, match_num)
	for (int RANSACnum = 0; RANSACnum < 4000; RANSACnum++){
		if (stop) { continue; }
		int index[6];
		int prosac_time = (10 + RANSACnum) < match_num ? (10 + RANSACnum) : (match_num);
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
			else { index[j] = n; }
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
		cv::Matx34d P;
		//cout << " after select minimal set" << endl;
		//cout << " minimal set size: " << minimal_set.size() << endl;
		//compute P
		//if (6 != minimal_set.size()){ cout << " minimal set size: " << minimal_set.size() << endl; }
		if (0 == CM_Compute(P, minimal_set)){ continue; }
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
		int inlier_num_thres = (match_num + 1) >> 1;
		if (inlier_num > 100 || inlier_num > inlier_num_thres&&inlier_num > 12){
#pragma omp atomic
			stop++;
		}
	}

	cv::Matx34d P;
	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> inlier_match_list;
	std::vector<bool> bInlier_(match_num, 0);
	int inlier_num_last = 0, inlier_num_cur = 0;
	binlier.assign(match_num, 0);

	//refine the inlier
	while (1)
	{
		inlier_match_list.clear();
		bInlier_.assign(match_num, 0);
		if (bNormalized){
			for (int i = 0; i < match_num; i++){
				if (ComputeReprojectionError(P_inlier, match_2d_3d[i]) < 10.0){
					inlier_match_list.push_back(match_2d_3d_normalized[i]);
					bInlier_[i] = 1;
				}
			}
		}
		else{
			for (int i = 0; i < match_num; i++){
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
		if (0 == CM_Compute(P, inlier_match_list)){ break; }

		if (bNormalized){
			P = mat_2d_scaling_inv*P*mat_3d_scaling;
		}
		P_inlier = P;
		inlier_num_last = inlier_num_cur;
		binlier.swap(bInlier_);
	}

	std::cout << "inlier match num: " <<inlier_num_last << std::endl;
	if (inlier_num_last < 6){
		std::cout << "not enough inlier, no pose calculated." << std::endl;
		return 0;
	}

	P_inlier = P;
	cv::Matx33d KR, K_RQ, R_RQ;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++)
			KR(i, j) = P(i, j);
	}
	if (M3Det(KR) < 0){
		P = -P;
	}

	cv::Vec4d	camera_position;
	// The function computes a decomposition of a projection matrix into
	// a calibration  and a rotation matrix and the position of a camera.
	cv::decomposeProjectionMatrix(P, K_estimated, R, camera_position);
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

	T[0] = t(0, 0);
	T[1] = t(1, 0);
	T[2] = t(2, 0);

	return inlier_num_last;
}

//calculated P is the right null space unit vector corresponding to the A's minial sigular value 
bool Geometry::CM_Compute(cv::Matx34d& P,
	const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d)
{
	//cout << "enter CM_Compute" << endl;

	int row_num = (int)match_2d_3d.size();
	if (row_num < 6){
		std::cout << "match_2d_3d num is less than 6. geometry.cpp line 436" << std::endl;
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
bool Geometry::Normalize()
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
	 
	auto &mat_2d = mat_2d_scaling_inv;
	auto &mat_3d = mat_3d_scaling;

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


//Compute pose use nonlinear optimization(Motion only bundle adjustment)
bool Geometry::RefinePoseSBA(const bool K_fixed)
{
	sba_warper_data sba;
	sba.clear();

	sba.pnp = 3;//3 parameters per 3d point
	sba.mnp = 2;//2 parameters per pixel point
	sba.cnp = 6; //ri, rj, rk, tx, ty, tz;
	sba.ncamera = 1;//refine only one image pose

	//fix intrinsics? then call SbaMotionOnly use cnp to indicate if fix intrinsics
	if (0 == K_fixed){
		sba.cnp += 5;//fu u0 v0 ar s
		sba.fix_K = 0;
		sba.K = 0;
	}
	else{
		sba.fix_K = 1;
		sba.K = new double[5];
		//initiate camera parameters
		sba.K[0] = K_estimated(0, 0); //fu
		sba.K[1] = K_estimated(0, 2); //u0
		sba.K[2] = K_estimated(1, 2); //v0
		sba.K[3] = K_estimated(1, 1) / K_estimated(0, 0); //ar = fv/fu
		sba.K[4] = K_estimated(0, 1); //skew
	}

	sba.n3dpoints = 0;
	sba.n2dpoints = 0;
	for (int i = 0; i < match_2d_3d.size(); i++){
		if (binlier[i]){
			++(sba.n3dpoints);
		}
	}
	sba.n2dpoints = sba.n3dpoints;
	
	sba.vmask			= new char[sba.ncamera*sba.n3dpoints];
	sba.para_camera		= new double[sba.ncamera*sba.cnp];//for now it always 11
	sba.para_3dpoints	= new double[sba.n3dpoints*sba.pnp];
	sba.para_2dpoints	= new double[sba.n2dpoints*sba.mnp];

	//TODO£º use try and catch
	if (!sba.vmask || !sba.para_camera || !sba.para_3dpoints || !sba.para_2dpoints)
	{
		std::cerr << "new error. geometry.cpp line 601" << std::endl;
	}
	int j = 0;
	//prepare 3d points and image points coordinates
	for (int i = 0; i < match_2d_3d.size(); i++){
		if (binlier[i]){
			sba.para_2dpoints[2 * j + 0] = match_2d_3d[i].first[0];//x_image_pixel
			sba.para_2dpoints[2 * j + 1] = match_2d_3d[i].first[1];//y_image_pixel
			sba.para_3dpoints[3 * j + 0] = match_2d_3d[i].second[0];//X_world
			sba.para_3dpoints[3 * j + 1] = match_2d_3d[i].second[1];//Y_world
			sba.para_3dpoints[3 * j + 2] = match_2d_3d[i].second[2];//Z_world
			sba.vmask[j] = 1;
			j++;
		}
	}

	//prepare frame parameter, e.g. 3 for rotation and 3 for translation
	//if(sba.cnp == 11) add another 5 caribration parameters
	//convert the rotation mat into Quaternion
	double quat[4];
	RotationToQuaterion(R, quat);
	if (0 == K_fixed){
		//initiate camera parameters
		sba.para_camera[0] = K_estimated(0, 0); //fu
		sba.para_camera[1] = K_estimated(0, 2); //u0
		sba.para_camera[2] = K_estimated(1, 2); //v0
		sba.para_camera[3] = K_estimated(1, 1) / K_estimated(0, 0); //ar = fv/fu
		sba.para_camera[4] = K_estimated(0, 1); //skew
	}

	sba.para_camera[sba.cnp - 6] = quat[1];
	sba.para_camera[sba.cnp - 5] = quat[2];
	sba.para_camera[sba.cnp - 4] = quat[3];
	sba.para_camera[sba.cnp - 3] = T[0];
	sba.para_camera[sba.cnp - 2] = T[1];
	sba.para_camera[sba.cnp - 1] = T[2];

	std::ofstream os("sba_debug1.txt", std::ios::out | std::ios::trunc);
	if (!os){
		std::cout << "sba_debug.txt file open fail." << std::endl;
	}
	sba.print(os);
	os.close();

	if ( 0 == SbaMotionOnly(sba) )
	{
		std::cout << " call SbaMotionOnly fail. " << std::endl;
		return 0; //fail
	}
	
	os.open("sba_debug2.txt", std::ios::out | std::ios::trunc);
	sba.print(os);
	os.close();

	quat[1] = sba.para_camera[sba.cnp - 6];
	quat[2] = sba.para_camera[sba.cnp - 5];
	quat[3] = sba.para_camera[sba.cnp - 4];
	T[0] = sba.para_camera[sba.cnp - 3];
	T[1] = sba.para_camera[sba.cnp - 2];
	T[2] = sba.para_camera[sba.cnp - 1];
	
	cv::Matx33d Rq;
	QuaternionToRotation(quat, Rq);
	//Rq is the rotation relative to the world coordinate
	R = Rq * R;

	return 1;
}

//refine pose use TooN
//need K_estimated from DLT to be fixed
bool Geometry::RefinePoseTooN()
{
	using namespace TooN;
	
	Matrix<3> R_est, K_est;
	Vector<3> T_est;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			R_est[i][j] = R(i, j);
			K_est[i][j] = K_estimated(i, j);
		}
		T_est[i] = T[i];
	}

	SE3<> ini;
	ini.get_rotation() = R_est;
	ini.get_translation() = T_est;
	
	trackdatalist_3D trackdatalist3d;
	trackdatalist3d.clear();
	for (int i = 0; i < match_2d_3d.size(); i++){
		if (!binlier[i]){
			continue;
		}
		
		Vector<2> p_image;
		p_image[0] = match_2d_3d[i].first[0];
		p_image[1] = match_2d_3d[i].first[1];
		Vector<3> p_world;
		p_world[0] = match_2d_3d[i].second[0];
		p_world[1] = match_2d_3d[i].second[1];
		p_world[2] = match_2d_3d[i].second[2];
		trackdata_3D* ptr = new trackdata_3D(p_world, p_image, 0);
		trackdatalist3d.list.push_back(ptr);
	}

	Vector<6> poseupdate;
	if (trackdatalist3d.list.size() < 10){
		for (int i = 0; i < trackdatalist3d.list.size(); i++){
			trackdatalist3d.list[i]->projection(ini, K_est);
		}
	}
	else{
		for (int iter = 0; iter < 20; iter++){
			bool bnonlinear = false;
			if (iter == 0 || iter == 4 || iter == 9 || iter == 13 || iter == 17 || iter == 19){
				bnonlinear = true;
			}
			std::vector<double> verrorsquared;
			WLS<6> wls;
			wls.add_prior(100.0);
			if (bnonlinear){
				for (int i = 0; i < trackdatalist3d.list.size(); i++)
				{
					trackdatalist3d.list[i]->projection(ini, K_est);
					trackdatalist3d.list[i]->CalcJacobian(K_est);
				}
			}
			else{
				for (int i = 0; i < trackdatalist3d.list.size(); i++){
					trackdatalist3d.list[i]->linearupdate(poseupdate);
				}
			}

			for (int i = 0; i < trackdatalist3d.list.size(); i++){
				Vector<2> err = trackdatalist3d.list[i]->measurement 
					- trackdatalist3d.list[i]->reprojection;
				double errsquared = err*err;
				verrorsquared.push_back(errsquared);
			}

			double dsigmasquared = Tukey::FindSigmaSquared(verrorsquared);
			if (iter>5) dsigmasquared = 49.0;

			for (int i = 0; i < trackdatalist3d.list.size(); i++){
				Vector<2> err = trackdatalist3d.list[i]->measurement
					- trackdatalist3d.list[i]->reprojection;
				double errsquared = err*err;
				double weight = Tukey::Weight(errsquared, dsigmasquared);
				wls.add_mJ(err[0], trackdatalist3d.list[i]->mJacobian[0], weight);
				wls.add_mJ(err[1], trackdatalist3d.list[i]->mJacobian[1], weight);
			}

			wls.compute();
			poseupdate = wls.get_mu();
			ini = SE3<>::exp(poseupdate)*ini;
		}
		for (int i = 0; i < trackdatalist3d.list.size(); i++){
			trackdatalist3d.list[i]->projection(ini, K_est);
		}
	}
	
	R_est = ini.get_rotation().get_matrix();
	T_est = ini.get_translation();

	cout << "after refine TooN: " << endl;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			cout << R_est[i][j] << " ";
		}
		cout << endl;
	}
	cout << T_est[0] << " " << T_est[1] << " ";
	cout << T_est[2] << endl; 
	
	return 1;
}


//unit quaternion are assumed
//q = w + xi + yj + zk
inline void Geometry::QuaternionToRotation(const double* quaterion, cv::Matx33d& R)const
{
	double w = quaterion[0];
	double x = quaterion[1];
	double y = quaterion[2];
	double z = quaterion[3];

	double wx = 2 * w*x, wy = 2 * w*y, wz = 2 * w*z;
	double xx = 2 * x*x, xy = 2 * x*y, xz = 2 * x*z;
	double yy = 2 * y*y, yz = 2 * y*z, zz = 2 * z*z;

	R(0, 0) = 1.0 - yy - zz;	R(0, 1) = xy - wz;			R(0, 2) = xz + wy;
	R(1, 0) = xy + wz;			R(1, 1) = 1.0 - xx - zz;	R(1, 2) = yz - wx;
	R(2, 0) = xz - wy;			R(2, 1) = yz + wx;			R(2, 2) = 1.0 - xx - yy;
}

//unit quaternion are assumed
//q = w + xi + yj + zk 
inline void Geometry::RotationToQuaterion(const cv::Matx33d& R, double* quaternion)const
{
	double& w = quaternion[0];
	double& x = quaternion[1];
	double& y = quaternion[2];
	double& z = quaternion[3];
	// This algorithm comes from  "Quaternion Calculus and Fast Animation",
	// Ken Shoemake, 1987 SIGGRAPH course notes
	double t = R(0, 0) + R(1, 1) + R(2, 2);
	if (t > 0){
		t = std::sqrt(t + 1.0);
		w = 0.5*t;
		t = 0.5 / t;
		x = (R(2, 1) - R(1, 2))*t;
		y = (R(0, 2) - R(2, 0))*t;
		z = (R(1, 0) - R(0, 1))*t;
	}
	else{
		int i = 0;
		if (R(1, 1) > R(0, 0)) 
			i = 1;
		if (R(2, 2) > R(i, i)) 
			i = 2;
		int j = (i + 1) % 3;
		int k = (j + 1) % 3;
		
		t = std::sqrt(1.0 + R(i, i) - R(j, j) - R(k, k));
		quaternion[i+1] = 0.5*t;
		t = 0.5 / t;
		w = (R(k, j) - R(j, k))*t;
		quaternion[j+1] = (R(j, i) + R(i, j))*t;
		quaternion[k+1] = (R(k, i) + R(i, k))*t;
		//in case that w is nagtive
		if (w < 0){
			w = -w; x = -x; y = -y; z = -z;
		}
	}
}

//////////////////////////////////////////////////////////////////////////
//for test
const double uc = 320;
const double vc = 240;
const double fu = 800;
const double fv = 800;
const int n = 100;
const double noise = 2.2;

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
	t[1] = 3.0f;
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

bool Orthogonal(const cv::Matx33d& R){
	double res = 0.0;
	for (int i = 0; i < 3; i++){
		res += R(0, i)*R(1, i);
	}
	if (int(res) != 0) return 0;
	
	res = 0.0;
	for (int i = 0; i < 3; i++){
		res += R(0, i)*R(2, i);
	}
	if (int(res) != 0) return 0;

	for (int i = 0; i < 3; i++){
		res += R(1, i)*R(2, i);
	}
	if (int(res) != 0) return 0;

	return 1;
}


void Geometry::TestGeometry(){
	srand((int)time(0));
	std::cout << "test geometry function" << std::endl;

	SetIntrinsicParameter(float(fu), int(uc), int(vc));

	/********************************************************/
	epnp PnP;
	PnP.set_internal_parameters(uc, vc, fu, fv);
	PnP.set_maximum_number_of_correspondences(n);

	double R_true[3][3], t_true[3];
	random_pose(R_true, t_true);
	
	std::cout << "ground truth R: " << std::endl;
	for (int i = 0; i < 3; i++){
		for (int j = 0; j < 3; j++){
			std::cout << R_true[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "ground truth t: " ;
	for (int i = 0; i < 3; i++)
		std::cout << t_true[i] << " ";
	cout << endl;

	auto& match_list = match_2d_3d;

	std::vector<cv::Point3d> list_points3d;
	std::vector<cv::Point2d> list_points2d;

	cv::Mat	points_3d(n, 3, CV_64FC1);
	cv::Mat points_2d(n, 2, CV_64FC1);

	PnP.reset_correspondences();
	for (int i = 0; i < n; i++) {
		double Xw, Yw, Zw, u, v;
		random_point(Xw, Yw, Zw);
		project_with_noise(R_true, t_true, Xw, Yw, Zw, u, v);
		if (!(u >= 0 && u <= 640 && v >= 0 && v <= 480)){ 
			i--; continue; 
		}
		PnP.add_correspondence(Xw, Yw, Zw, u, v);
		match_list.push_back(std::make_pair(cv::Vec2d(u, v), cv::Vec3d(Xw, Yw, Zw)));
		list_points2d.push_back(cv::Point2d(u, v));
		list_points3d.push_back(cv::Point3d(Xw, Yw, Zw));

		points_2d.ptr<double>(i)[0] = u;
		points_2d.ptr<double>(i)[1] = v;
		points_3d.ptr<double>(i)[0] = Xw;
		points_3d.ptr<double>(i)[1] = Yw;
		points_3d.ptr<double>(i)[2] = Zw;
		
	}
	double R_est[3][3], t_est[3];
	double err2 = PnP.compute_pose(R_est, t_est); 
	std::cout << "epnp estimated R: " << std::endl;
	for(int i=0; i<3; i++){
		for(int j=0; j<3; j++){
			std::cout << R_est[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << "epnp estimated t: " << std::endl;
	std::cout << t_est[0] << " " << t_est[1] << " " << t_est[2] << std::endl;

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
	CM_Compute(P, match_list);
	
	P = (1.0 / P(2, 3))*P;
	P = 6 * P;
	cout << "geo calculated P: " << P << endl;
#endif
	/********************************************************/

#if 0
	Normalize();
	cout << "original matched" << endl;
	for (auto& e : match_list){
		cout << e.first[0] << " " << e.first[1] << " "
			<< e.second[0] << " " << e.second[1] << " " << e.second[2] << endl;
	}

	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> match_n;
	for (auto& e : match_2d_3d_normalized){
		cv::Vec3d v3(e.first[0], e.first[1], 1.0);
		v3 = mat_2d_scaling_inv*v3;
		cv::Vec4d v4(e.second[0], e.second[1], e.second[2], 1.0);
		v4 = mat_3d_scaling.inv()*v4;
		match_n.push_back(std::make_pair(cv::Vec2d(v3[0], v3[1]), cv::Vec3d(v4[0], v4[1], v4[2])));
		//match_n.push_back(make_pair(Vec2d(v3[0] / v3[2], v3[1] / v3[2]), Vec3d(v4[0] / v4[3], v4[1] / v4[3], v4[2] / v4[3])));
	}
	cout << " recover matched" << endl;
	for (auto& e : match_n){
		cout << e.first[0] << " " << e.first[1] << " "
			<< e.second[0] << " " << e.second[1] << " " << e.second[2] << endl;
	}
#endif
	/********************************************************/

	cv::Matx33d	R_dlt;
	cv::Vec3d	t_dlt;

#if 1
	//for (int i = 0; i < 1000; i++)
	std::cout << "DLT inlier: " << ComputePoseDLT() << std::endl;
	GetRT(R_dlt, t_dlt);
	
	std::cout << "DLT R: " << std::endl << R << std::endl;
	std::cout << "DLT t: " << T << std::endl;

	std::cout << "refine_sba pose okay? " << RefinePoseSBA(0) << std::endl;
	std::cout << "RefinePoseSBA R: " << std::endl << R << std::endl;
	std::cout << "RefinePoseSBA t: " << T << std::endl;

	R = R_dlt; T = t_dlt;
	std::cout << "refine_sba pose okay? " << RefinePoseTooN() << std::endl;
	std::cout << "RefinePoseTooN R: " << std::endl << R << std::endl;
	std::cout << "RefinePoseTooN t: " << T << std::endl;

#endif

#if 0
	double quat[4] = { 0, 0, 0, 0 };
	RotationToQuaterion(R_dlt, quat);
	std::cout << "RotationToQuaterion Q: "
		<< quat[0] << " " << quat[1] << " "
		<< quat[2] << " " << quat[3] << std::endl;

	cv::Matx33d	R_quat;
	QuaternionToRotation(quat, R_quat);
	std::cout << "QuaternionToRotation R:" << std::endl << R_quat << std::endl;
	
#endif
	/*******************************************************/
#if 0

	//for (int i = 0; i < 1000; i++)
	std::cout << "epnp inlier: " << ComputePoseEPnP() << std::endl;
	GetRT(R_dlt, t_dlt);
	std::cout << "epnp R: " << std::endl << R_dlt << std::endl;
	std::cout << "epnp t: " << std::endl << t_dlt << std::endl;
	
	std::cout << "Detm(R): " << M3Det(R_dlt) << std::endl;
	std::cout << "Orthogonal: " << Orthogonal(R_dlt) << std::endl;

#endif

#if 0

	cv::Mat distCoeffs = cv::Mat::zeros(1, 4, CV_64FC1);
	cv::Mat Rmat, Rvec, Tvec;
	//inliers contain the indices of inliers in objectPoints and imagePoints .
	cv::Mat inliers;
	//cv::solvePnP(points_3d, points_2d, K, distCoeffs, Rvec, Tvec, false, CV_P3P);
	cv::solvePnPRansac(list_points3d, list_points2d, K, distCoeffs, Rvec, Tvec, false, 4000, 10.0, 0.99, inliers, CV_P3P);
	//cv::solvePnPRansac(points_3d, points_2d, K, distCoeffs, Rvec, Tvec, false, 4000, 10.0, 0.99, inliers, CV_P3P);
	cv::Rodrigues(Rvec, Rmat);
	std::cout << "solvePnPRansac inlier: " << inliers.rows << std::endl;
	std::cout << "R: " << std::endl << Rmat << std::endl;
	std::cout << "T: " << std::endl << Tvec << std::endl;

#endif

}