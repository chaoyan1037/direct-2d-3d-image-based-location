#ifndef  _PICTURE_H_
#define  _PICTURE_H_

#include<string>
#include<vector>
#include<algorithm>
#include<opencv2/opencv.hpp>

#include "bundlercamera.h"
/*
*	PICTURE class to store key point and descriptor
*	for a picture from .key file
*
*/

#ifndef _MSC_VER
#define NOEXCEPT noexcept
#else
#define NOEXCEPT
#endif

//calculate the distance between the two sift descriptor
int CalculateSIFTDistanceSquared(const unsigned char* d1, const unsigned char* d2);

struct SIFT_KeyPoint
{
	//origin is the left-up corner
	//in .key file  y  x  scale  orientation
	float y;
	float x; //x is col
	float scale;
	float orientation;//(in radians from - PI to PI)
};


struct SIFT_Descriptor
{
	unsigned char* ptrDesc;
	static int legth;
	void ClearDesc(){
		if (ptrDesc != nullptr){
			delete[] ptrDesc;
			ptrDesc = nullptr;
		}
	}
	SIFT_Descriptor(){
		ptrDesc = nullptr;
	}
	~SIFT_Descriptor(){
		ClearDesc();
	}
	
	//copy constructor
	SIFT_Descriptor(const SIFT_Descriptor&d){
		if (d.ptrDesc != 0){
			ptrDesc = new unsigned char[legth];
			std::copy(d.ptrDesc, d.ptrDesc + d.legth, ptrDesc);
		}
	}

	//copy assignment
	SIFT_Descriptor& operator=(const SIFT_Descriptor&d){
		//std::cout << " sift desc copy assignment" << std::endl;
		if (this != &d){
			ClearDesc();
			if (d.ptrDesc != 0){
				ptrDesc = new unsigned char[legth];
				std::copy(d.ptrDesc, d.ptrDesc + d.legth, ptrDesc);
			}
		}
		return(*this);
	}

	//move constructor
	SIFT_Descriptor(SIFT_Descriptor&&d) NOEXCEPT
		:ptrDesc(d.ptrDesc)
	{
		d.ptrDesc = nullptr;
	}
	
	//move assignment 
	SIFT_Descriptor& operator=(SIFT_Descriptor&&d) NOEXCEPT
	{
		//std::cout << " sift desc move assignment" << std::endl;
		if (this != &d){
			ClearDesc();
			ptrDesc = d.ptrDesc;
			d.ptrDesc = nullptr;
		}
		return (*this);
	}
};


//store a picture's key points and descriptor
class PICTURE
{
public:
	//default constructor
	PICTURE();

	~PICTURE();

	//copy constructor
	//PICTURE(const PICTURE &pic);

	//copy assignment constructor
	//PICTURE& operator=(const PICTURE &pic);

	//load descriptor from the .key file
	bool LoadKeyPointAndDes(const std::string& des_filename, bool bHave_image_size=false);
	
	//set size
	void SetImageSize(const int height, const int width);
	//get size
	void GetImageSize(int& height, int& width) const;

	//the num of feature points of the picture
	size_t PointsNum() const
	{ return mFeature_points.size(); } ;

	//clear all the data
	void ClearData();

	//return the key points
	std::vector<SIFT_KeyPoint>& GetFeaturePoint(){
		return mFeature_points;
	}
	const std::vector<SIFT_KeyPoint>& GetFeaturePoint()const{
		return mFeature_points;
	}

	//return the descriptor, store as chars
	std::vector<SIFT_Descriptor>& GetDescriptor() {
		return mDescriptors;
	}
	const std::vector<SIFT_Descriptor>& GetDescriptor() const{
		return mDescriptors;
	}

private:
	int		mImageHeight;
	int		mImgaeWidth;
	int		mKeypoint_num;//total num of descriptors
	int		mDes_length;//128 for sift
	//origin is the left-up corner
	//for query image, make the center as origin with known image size
	//for data base image, just leave it without using x, y
	std::vector<SIFT_KeyPoint>		mFeature_points;
	std::vector<SIFT_Descriptor>	mDescriptors;

};

//contain pictures and cameras
class ALL_PICTURES
{
public:
	//default constructor, set empty
	ALL_PICTURES();

	//set only key file path and list file path 
	ALL_PICTURES(const std::string& key, const std::string &list);

	//set key file path, image file path and list file path 
	ALL_PICTURES(const std::string& key, const std::string& image, const std::string &list);

	//destructor
	~ALL_PICTURES();

	// load all picture keys from the list, clear if exist
	// for query image, also load image size
	bool LoadPicturesKeyFile();

	//load the camera pose ground truth
	bool LoadCamerasPose(const std::string& s);


	//clear all picture
	void ClearPicsCameras(){
		mPictures.clear();
		mCameras.clear();
	}

	void ClearPics(){mPictures.clear();}
	void ClearCameras(){mCameras.clear();}

	//set the string contents
	void SetParameters(const std::string& model_path, const std::string &list);

	//if it is for query image, make flag true
	void SetQueryFlag(const bool flag);
	const bool RetQueryFlag() const;

	//return all pictures
	std::vector<PICTURE>& GetAllPictures(){
		return mPictures;
	}
	const std::vector<PICTURE>& GetAllPictures()const{
		return mPictures;
	}

	//return all pictures
	std::vector<BUNDLER_CAMERA>& GetAllCameras(){
		return mCameras;
	}
	const std::vector<BUNDLER_CAMERA>& GetAllCameras() const{
		return mCameras;
	}

	//for query images, return all focal
	std::vector<double> GetAllQueryFocal(){
		return mQueryfocal;
	}
	const std::vector<double> GetAllQueryFocal() const{
		return mQueryfocal;
	}

private:
	
	std::string						mKeyfilepath;//the image key file path
	std::string						mImagepath;//the images path
	std::string						mPicturelistfile;//a txt file store the picture name
	std::vector< PICTURE >			mPictures;//vector store all pictures database or query
	std::vector< BUNDLER_CAMERA >	mCameras;

	bool							mIsqueryimage;//indicate if is query image
	std::vector<double>				mQueryfocal;//the focal length of query image
};


#endif // !_CAMERA_H_
