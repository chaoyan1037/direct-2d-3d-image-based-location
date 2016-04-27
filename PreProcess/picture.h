#ifndef  _PICTURE_H_
#define  _PICTURE_H_

#include<string>
#include<vector>
#include<algorithm>
#include<opencv2/opencv.hpp>

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
		legth = 128;
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
	PICTURE(){};
	~PICTURE(){ ClearData(); };

	//copy constructor
	//PICTURE(const PICTURE &pic);

	//copy assignment constructor
	//PICTURE& operator=(const PICTURE &pic);

	//load descriptor from the .key file
	bool LoadKeyPointAndDes(std::string des_filename);
	
	//the num of feature points of the picture
	size_t PointsNum() const
	{ return mFeature_points.size(); } ;

	//clear all the data
	void ClearData();

	//return the key points
	const std::vector<SIFT_KeyPoint>& GetFeaturePoint()const{
		return mFeature_points;
	}

	//return the descriptor, store as chars
	const std::vector<SIFT_Descriptor>& GetDescriptor() const{
		return mDescriptors;
	}

private:

	size_t							 mKeypoint_num;//total num of descriptors
	size_t							 mDes_length;//128 for sift
	std::vector<SIFT_KeyPoint>		 mFeature_points;//origin is the left-up corner
	std::vector<SIFT_Descriptor>	 mDescriptors;
};

class ALL_PICTURES
{
public:

	ALL_PICTURES() :mDBpath(""), mPictureListFile(""){};

	ALL_PICTURES(const std::string& model_path, const std::string &list) 
		:mDBpath(model_path), mPictureListFile(list){};

	~ALL_PICTURES(){};

	//load all picture from the list
	bool LoadAllPictures();

	//clear all picture
	void ClearAllPictures(){
		mAll_pictures.clear();
	}

	//set the string contents
	bool SetParameters(const std::string& model_path, const std::string &list);

	//return all pictures
	const std::vector<PICTURE>& GetAllPictures() const
	{
		return mAll_pictures;
	}

private:

	std::string						mDBpath;//the model database path
	std::string						mPictureListFile;//a txt file store the picture name
	std::vector<PICTURE>			mAll_pictures;//vector store all pictures database or query
};


#endif // !_CAMERA_H_
