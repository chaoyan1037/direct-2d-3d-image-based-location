#ifndef _GEOMETRY_H_
#define _GEOMETRY_H_

#include <vector>
#include <assert.h>

#include <opencv/cv.h>


class Geometry
{
public:
	Geometry(){}

	~Geometry(){}

	//reproject the 3d point to the image plane and compute error 
	double ComputeReprojectionError(const cv::Matx34d& P, 
		const std::pair<cv::Vec2d, cv::Vec3d>& match);

	//Compute the pose use ePnP algorithm, return inlier num
	int ComputePoseEPnP();

	//Compute the pose use DLT
	int ComputePoseDLT();

	bool CM_Compute(cv::Matx34d& P,
		const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d);

	//normalize when use DLT method
	bool Normalize();

	//refine pose use sparse bundle adjustment nonlinear optimization
	bool RefinePoseSBA(const bool K_fixed);

	//refine pose use TooN
	bool RefinePoseTooN();

	inline double M3Det(const cv::Matx33d &m) const
	{
		return	m(0,0) * (m(1,1) * m(2,2) - m(1,2) * m(2,1)) -
				m(0,1) * (m(1,0) * m(2,2) - m(1,2) * m(2,0)) +
				m(0,2) * (m(1,0) * m(2,1) - m(1,1) * m(2,0));
	}
	inline double M3Error(const cv::Matx33d &m) const
	{
		return (1-m(0, 0))*(1-m(0, 0)) + m(0, 1)*m(0, 1) + m(0, 2)*m(0, 2)
			+ m(1, 0)*m(1, 0) + (1 - m(1, 1))*(1 - m(1, 1)) + m(1, 2)*m(1, 2)
			+ m(2, 0)*m(2, 0) + m(2, 1)*m(2, 1) + (1 - m(2, 2))*(1 - m(2, 2));
	}

	/*cv::Matx33d M3Mult(const cv::Matx33d& m1, const cv::Matx33d& m2)
	{
		cv::Matx33d res;
		for (int i = 0; i < 3; i++){
			for (int j = 0; j < 3; j++){
				res(i, j) = m1(i, 0)*m2(0, j) + m1(i, 1)*m2(1, j) + m1(i, 2)*m2(2, j);
			}
		}
		return res;
	}*/

	/*inline cv::Matx33d Inv(const cv::Matx33d &m){
		cv::Matx33d madjoin;
		double delta = M3Det(m);
		assert(delta<-0.000001 || delta>0.000001);

		madjoin(0, 0) = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
		madjoin(0, 1) = m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2);
		madjoin(0, 2) = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);
		madjoin(1, 0) = m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2);
		madjoin(1, 1) = m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
		madjoin(1, 2) = m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1);
		madjoin(2, 0) = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
		madjoin(2, 1) = m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2);
		madjoin(2, 2) = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

		madjoin *= (1.0/delta);

		return madjoin.t();
	}*/

	void SetIntrinsicParameter(float f, int u, int v);
	void SetK(const cv::Matx33d& Ki);

	//get estimated R and T
	void GetRT(cv::Matx33d& R_est, cv::Vec3d& T_est) const{
		R_est = R; T_est = T;
	}

	//save the matched 3d feature(x, y) and 3d point(x, y, z)
	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> match_2d_3d;
	
	//indicate whether match in match_2d_3d vector is inlier
	std::vector<bool> binlier;

	//just for test the geometry module
	void TestGeometry();

private:
	/** Convert the quaternion to a 3x3 rotation matrix. 
	  The quaternion is required to
	* be normalized, otherwise the result is undefined.**/
	inline void QuaternionToRotation(const double* quaterion,
		cv::Matx33d& R) const;

	inline void RotationToQuaterion(const cv::Matx33d& R, 
		double* quaterion) const;

	//the real K for epnp
	cv::Matx33d K;

	cv::Matx33d K_Inv;

	//estimated R
	cv::Matx33d R;

	//estimated T
	cv::Vec3d T;

	//the estimated K for DLT
	cv::Matx33d K_estimated;

	std::vector<std::pair<cv::Vec2d, cv::Vec3d>> match_2d_3d_normalized;
	// x = P * X; mat2ds * x = P_scaled * mat3ds * X
	// x = mat2ds_inv * P_scaled *mat3ds * X = P * X
	// P = mat2ds_inv * P_scaled *mat3ds 
	cv::Matx33d	mat_2d_scaling_inv;

	cv::Matx44d mat_3d_scaling;
};




#endif