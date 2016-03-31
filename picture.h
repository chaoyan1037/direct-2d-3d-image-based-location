#ifndef  _PICTURE_H_
#define  _PICTURE_H_

#include<string>
#include<opencv2/opencv.hpp>

/*
*	PICTURE class to store key point and descriptor
*	for a picture from .key file
*
*/

struct SIFT_KeyPoint
{
	float x;//origin is the left-up corner
	float y;
	float scale;
	float orientation;//(in radians from - PI to PI)
};

//store a picture's key points and descriptor
class PICTURE
{
public:
	PICTURE(){};
	~PICTURE(){ ClearData(); };

	//load descriptor from the .key file
	bool LoadKeyPointAndDes(std::string des_filename);
	
	//the num of feature points of the picture
	int PointsNum(){ return mFeature_points.size(); };

	//clear all the data
	void ClearData();

	//return the key points
	std::vector<SIFT_KeyPoint>& GetFeaturePoint();

	//return the descriptor, store as chars
	std::vector<unsigned char*>& GetDescriptor();

private:

	int								 mKeypoint_num;//total num of descriptors
	int								 mDes_length;//128 for sift
	std::vector<SIFT_KeyPoint>		 mFeature_points;//origin is the left-up corner
	std::vector<unsigned char*>		 mDescriptors;

};

class ALL_PICTURES
{
public:
	ALL_PICTURES(const std::string& model_path, const std::string &list) 
		:mPictureListFile(list), mDBpath(model_path){};

	~ALL_PICTURES(){};

	//load all picture from the list
	bool LoadAllPictures();

	//clear all picture
	bool ClearAllPictures();

	//return all pictures
	std::vector<PICTURE>& GetAllPictures();

private:

	std::string						mDBpath;//the model database path
	std::string						mPictureListFile;//a txt file store the picture name
	std::vector<PICTURE>			mAll_pictures;//vector store all pictures database or query
};


#endif // !_CAMERA_H_
