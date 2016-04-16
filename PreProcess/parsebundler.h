#ifndef _PARSEBUNDLER_H_
#define _PARSEBUNDLER_H_

/************************************************************************/
/* parse bundler.out file, get the view  and 3d point information*/
/* for each 3D point, find the list of 2d feature points */
/************************************************************************/
#include <set>
#include <fstream>
#include <vector>
#include <omp.h>


#include "bundlercamera.h"
#include "picture.h"

/* definition of a 3D point consist of x,y,z and rgb */
class POINT3D
{
public:
	
	POINT3D()
	{
		x = y = z = 0.0;
		r = g = b = 0;
	}

	POINT3D(float x_, float y_, float z_) :x(x_), y(y_), z(z_)
	{
		r = g = b = 0; 
	}

	POINT3D(float x_, float y_, float z_, 
		unsigned char r_, unsigned char g_, unsigned char b_) 
		:x(x_), y(y_), z(z_), r(r_), g(g_), b(b_){}

	POINT3D(const POINT3D& point)
	{
		x = point.x;y = point.y;z = point.z;
		r = point.r;g = point.g;b = point.b;
	}

	POINT3D& operator=(const POINT3D& point)
	{
		x = point.x; y = point.y; z = point.z;
		r = point.r; g = point.g; b = point.b;
		return (*this);
	}

	void Set3DCoordinate(float x_, float y_, float z_)
	{
		x = x_; y = y_; z = z_;
	}

	void Set3DPointColor(unsigned char r_, unsigned char g_, unsigned char b_)
	{
		r = r_; g = g_; b = b_;
	}

	float x, y, z;//coordinates
	unsigned char r, g, b;//point colors
};

/*  a view of a 3D point as defined by Bundler 
	contain the index of the camera the point was seen in it
	the key point index in the .key file of that that camera
	the x and y coordinate of that key point in a reference frame
	the coordinate center is the center of the image
	scale and orientation of that key point
*/
struct VIEW
{
	size_t camera;//the index of camera the feature was detected in
	size_t key;//index of this feature in the corresponding .key file
	
	float x, y;//coordinates the center is the image center
	float scale;
	float orientation;

	VIEW()
	{
		camera = key = 0;
		x = y = 0.0;
		scale = orientation = 0.0;
	}
};

/* store all the information of a 3D point,include a view list
The descriptor for view view_list[i] is stored in descriptors[128*i]...descriptors[128*i+127]
*/
class FEATURE_3D_INFO
{
public:

	POINT3D mPoint;
	std::vector<VIEW> mView_list;
	//each single descriptor save as char*
	std::vector<SIFT_Descriptor> mDescriptor;

	//constructor
	FEATURE_3D_INFO()
	{
		mView_list.clear();
		mDescriptor.clear();
	}

	//copy constructor
	FEATURE_3D_INFO(const FEATURE_3D_INFO& feature_3d)
	{
		ClearData();

		mView_list.resize(feature_3d.mView_list.size());
		mDescriptor.resize(feature_3d.mDescriptor.size());

#pragma omp parallel for
		for (int i = 0; i < feature_3d.mView_list.size(); ++i)
		{
			mView_list[i] = feature_3d.mView_list[i];
		}

#pragma omp parallel for
		for (int i = 0; i < feature_3d.mDescriptor.size(); i++)
		{
			mDescriptor[i] = feature_3d.mDescriptor[i];
		}

		mPoint = feature_3d.mPoint;
	}

	//destructor
	~FEATURE_3D_INFO(){ClearData();}

	//clear all the data
	void ClearData()
	{
		mDescriptor.clear();
		mView_list.clear();
	}

};

/* 
	class to parse bundler file
	also able to load the .key files of all cameras
*/

class PARSE_BUNDLER
{
public:
	//constructor
	PARSE_BUNDLER();

	~PARSE_BUNDLER();

	//set the bundler file name
	void SetBundleFileName(const std::string &s){
		mBundle_file = s;
	}

	//get the number of 3d points in reconstruction
	size_t GetNumPoints() const{
		return mNumbPoints;
	}

	//get the number of cameras
	size_t GetNumCameras() const{
		return mNumCameras;
	}

	//clear all the data
	void  ClearData(){
		mCameras.clear();
		mFeature_infos.clear();
		mNumbPoints = 0;
		mNumCameras = 0;
	}

	//parse the bundler file
	//input parameter: the file name of the output file(bundler.out) and
	//the filename of the list of images(usually list.txt)
	bool ParseBundlerFile();

	//load the .key info from ALL_CAMERA
	bool LoadCameraInfo( ALL_PICTURES& all_pics);

	//return the 3d point feature info
	const std::vector< FEATURE_3D_INFO >& GetFeature3DInfo() const
	{
		return mFeature_infos;
	}

private:

	std::string						mBundle_file;
	std::vector< BUNDLER_CAMERA >	mCameras;
	std::vector< FEATURE_3D_INFO >	mFeature_infos;
	size_t mNumbPoints, mNumCameras;
};

#endif