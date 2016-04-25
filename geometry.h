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
	int ComputePoseEPnP(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
		cv::Matx33d &R, cv::Vec3d &T, std::vector<bool>& binlier);

	//Compute the pose use DLT
	int ComputePoseDLT(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
		cv::Matx33d &R, cv::Vec3d &T, std::vector<bool>& binlier, 
		cv::Matx33d &recovered_K);

	bool CM_Compute(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d, 
		cv::Matx34d& P);

	//normalize when use DLT method
	bool Normalize(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d, 
		std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d_normalized,
		cv::Matx33d& mat_2d_inv, cv::Matx44d& mat_3d);

	//Compute pose use nonlinear optimization
	int RefinePoseSBA(const std::vector<std::pair<cv::Vec2d, cv::Vec3d>>& match_2d_3d,
		cv::Matx33d &R_initial, cv::Vec3d &T_initial, const std::vector<bool>& binlier,
		const bool K_fixed, const cv::Matx33d & K_estimated);

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

	cv::Matx33d K;
	cv::Matx33d K_Inv;
	void SetIntrinsicParameter(float f, int u, int v);
	void SetK(const cv::Matx33d& Ki);

	//just for test the geometry module
	void TestGeometry();

private:
	/** Convert the quaternion to a 3x3 rotation matrix. The quaternion is required to
	* be normalized, otherwise the result is undefined.**/
	inline void QuaternionToRotation(const double* quaterion, cv::Matx33d& R) const;
	
	
	inline void RotationToQuaterion(const cv::Matx33d& R, double* quaterion) const;
};




#endif