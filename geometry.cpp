
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <omp.h>

#include "geometry.h"
#include "Timer/timer.h"
#include "Epnp/epnp.h"

//estimate camera pose from 2d-3d correspondence
//both DLT method and ePnP method

using namespace std;
using namespace cv;


void Geometry::SetIntrinsicParameter(float f, int u, int v){
	K(0, 0) = f;	K(0, 1) = 0.0;	K(0, 2) = u;
	K(1, 0) = 0.0;	K(1, 1) = f;	K(1, 2) = v;
	K(2, 0) = 0.0;	K(2, 1) = 0.0;	K(2, 2) = 1.0;
	
	K_Inv = K.inv();
	/*cv::Matx33d	K1, K2;
	K1(0, 0) = 1.0/f;	K1(0, 1) = 0.0;		K1(0, 2) = 0.0;
	K1(1, 0) = 0.0;		K1(1, 1) = 1.0/f;	K1(1, 2) = 0.0;
	K1(2, 0) = 0.0;		K1(2, 1) = 0.0;		K1(2, 2) = 1.0;

	K2(0, 0) = 1.0;		K2(0, 1) = 0.0;		K2(0, 2) = -u;
	K2(1, 0) = 0.0;		K2(1, 1) = 1.0;		K2(1, 2) = -v;
	K2(2, 0) = 0.0;		K2(2, 1) = 0.0;		K2(2, 2) = 1.0;

	K_Inv = K1*K2;*/
}

void Geometry::SetK(const cv::Matx33d& Ki){
	K = Ki;
	K_Inv = K.inv();
}

//input projection matrix P and 2d image pixel points - 3d Euclidean coordinates match
//only for a single 2d-3d match pair
double Geometry::ComputeReprojectionError(const cv::Matx34d& P, const std::pair<cv::Vec2d, cv::Vec3d>& match)
{
	cv::Vec4d	point;
	point[0] = match.second[0];
	point[1] = match.second[1];
	point[2] = match.second[2];
	point[3] = 1;

	cv::Vec3d	reprojection = P*point;
	
	//it is near 0.0
	if (reprojection[2] > -0.000001 && reprojection[2] < 0.000001){
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
int Geometry::ComputePoseEPnP(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	cv::Matx33d &R, cv::Vec3d &T, std::vector<bool>& binlier){
	//init the rand number generate
	srand(int(time(0)));

	int inlier_num = 0;
	int inlier_num_best = 0;
	int index[12];
	int index_best[12];
	int match_num = (int)match_2d_3d.size();
	int inlier_num_thres = match_num >> 1;

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
		P.col(0) = KR.col(0);
		P.col(1) = KR.col(1);
		P.col(2) = KR.col(2);
		cout << P << endl;
		for(int i = 0; i < 3; i++){
			for(int j = 0; j < 3; j++){
				P(i, j) = KR(i, j);
			}
		}
		cout << P << endl;
		cv::Vec3d KT(K*T);
		P.col(3) = KT;
		cout << P << endl;
		P(0, 3) = KT[0];
		P(1, 3) = KT[1];
		P(2, 3) = KT[2];
		P(3, 3) = 1.0;
		cout << P << endl;

		inlier_num = 0;
		//check if it is an inlier
		for (auto _match : match_2d_3d){
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
	vector<bool> binlier_(match_num, 0);
	int inlier_num_last = 0, inlier_num_cur = 0;
	double err2 = 0.0;
	//refine the inlier
	while (1)
	{
		inlier_match_index_list.clear();
		//find inlier
		for (int i = 0; i < match_num; i++){
			binlier_[i] = 0;
			if (ComputeReprojectionError(P_inlier, match_2d_3d[i]) < 10.0){
				inlier_match_index_list.push_back(i);
				binlier_[i] = 1;
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
		P_inlier.col(0) = KR.col(0);
		P_inlier.col(1) = KR.col(1);
		P_inlier.col(2) = KR.col(2);

		cv::Vec3d KT(K*T);
		P_inlier(0, 3) = KT[0];
		P_inlier(1, 3) = KT[1];
		P_inlier(2, 3) = KT[2];
		P_inlier(3, 3) = 1.0;

		inlier_num_last = inlier_num_cur;
		binlier = binlier_;
	}
	std::cout << "Inlier match num: " << inlier_num_last << std::endl;
	std::cout << "Reprojection error: " << err2 << std::endl;
	return inlier_num_last;
}

//Compute the pose use DLT
int ComputePoseDLT(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	cv::Matx33d &R, cv::Vec3d &T, std::vector<bool>& binlier, cv::Matx33d & recovered_K);

void CM_Compute(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	cv::Matx34d& P);

//normalize when use DLT method
bool Normalize(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
	std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d_normalized,
	cv::Matx33d& mat_2d, cv::Matx44d& mat_3d);