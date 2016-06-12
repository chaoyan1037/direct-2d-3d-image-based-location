
#ifndef _SBA_WARPER_H
#define _SBA_WARPER_H

#include <iostream>
#include <ostream>
#include <fstream>
#include <string>

#define FULLQUATSZ     4

 /* in sba_imgproj.cpp */
 void calcImgProj(double a[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]); 
 void calcImgProjFullR(double a[5], double qr0[4], double t[3], double M[3], double n[2]);
 void calcImgProjJacKRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11], double jacmS[2][3]);
 void calcImgProjJacKRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKRT[2][11]);
 void calcImgProjJacS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);
 void calcImgProjJacRTS(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6], double jacmS[2][3]);
 void calcImgProjJacRT(double a[5], double qr0[4], double v[3], double t[3], double M[3], double jacmRT[2][6]);
 void calcDistImgProj(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double n[2]);
 void calcDistImgProjFullR(double a[5], double kc[5], double qr0[4], double t[3], double M[3], double n[2]);
 void calcDistImgProjJacKDRTS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16], double jacmS[2][3]);
 void calcDistImgProjJacKDRT(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmKDRT[2][16]);
 void calcDistImgProjJacS(double a[5], double kc[5], double qr0[4], double v[3], double t[3], double M[3], double jacmS[2][3]);


/* pointers to additional data, used for computed image projections and their jacobians */
struct globs_
{
	globs_(){
		rot0params = 0;
		intrcalib = 0;
		ptparams = 0;
		camparams = 0;
	}
	~globs_(){
		if (rot0params) delete[]	rot0params;
		if (intrcalib)	delete[]	intrcalib;
		if (ptparams)	delete[]	ptparams;
		if (camparams)	delete[]	camparams;
	}

	void print(std::ostream& os = std::cout);

	double *rot0params; /* initial rotation parameters, combined with a local rotation parameterization */
	double *intrcalib; /* the 5 intrinsic calibration parameters in the order [fu, u0, v0, ar, skew],
					   * where ar is the aspect ratio fv/fu.
					   * Used only when calibration is fixed for all cameras;
					   * otherwise, it is null and the intrinsic parameters are
					   * included in the set of motion parameters for each camera
					   */
	int nccalib; /* number of calibration parameters that must be kept constant.
				 * 0: all parameters are free
				 * 1: skew is fixed to its initial value, all other parameters vary (i.e. fu, u0, v0, ar)
				 * 2: skew and aspect ratio are fixed to their initial values, all other parameters vary (i.e. fu, u0, v0)
				 * 3: meaningless
				 * 4: skew, aspect ratio and principal point are fixed to their initial values, only the focal length varies (i.e. fu)
				 * 5: all intrinsics are kept fixed to their initial values
				 * >5: meaningless
				 * Used only when calibration varies among cameras
				 */

	int ncdist; /* number of distortion parameters in Bouguet's model that must be kept constant.
				* 0: all parameters are free
				* 1: 6th order radial distortion term (kc[4]) is fixed
				* 2: 6th order radial distortion and one of the tangential distortion terms (kc[3]) are fixed
				* 3: 6th order radial distortion and both tangential distortion terms (kc[3], kc[2]) are fixed [i.e., only 2nd & 4th order radial dist.]
				* 4: 4th & 6th order radial distortion terms and both tangential distortion ones are fixed [i.e., only 2nd order radial dist.]
				* 5: all distortion parameters are kept fixed to their initial values
				* >5: meaningless
				* Used only when calibration varies among cameras and distortion is to be estimated
				*/
	int cnp; /* number of parameters for ONE camera; e.g. 6 for Euclidean cameras */
	int pnp; /* number of parameters for ONE point; e.g. 3 for Euclidean points */
	int mnp; /* number of parameters for EACH measurement; usually 2 */

	double *ptparams; /* needed only when bundle adjusting for camera parameters only */
	double *camparams; /* needed only when bundle adjusting for structure parameters only */

	int ncamera;
	int n3dpoints;
	int n2dpoints;
};

struct sba_warper_data
{
	sba_warper_data();

	~sba_warper_data();
	
	//copy constructor
	sba_warper_data(const sba_warper_data&s);

	//copy assignment
	sba_warper_data& operator=(const sba_warper_data&s);

	// swap function
	void swap(sba_warper_data&s);

	void clear();

	// print the sba data to debug
	void print(std::ostream& os = std::cout);

	bool fix_K;
	double* K;
	double* para_camera;
	int ncamera, cnp;
	char* vmask;
	double* para_3dpoints;
	int n3dpoints, pnp;
	double* para_2dpoints;
	int n2dpoints, mnp;
};

//assume para_camera: f d and 3 for rotation and 3 for translation
bool SbaMotionOnly(sba_warper_data& sba);




#endif