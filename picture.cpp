#include "picture.h"

#include <fstream>
#include <sstream>
#include <memory>
#include <omp.h>
#include <windows.h>

using namespace std;

/****** for class PICTURE ****/
//clear all the data, delete the pointer content
void PICTURE::ClearData()
{
#pragma omp parallel for
	for (int i = 0; i < mKeypoint_num; ++i){
		if (!mDescriptors[i]) delete[] mDescriptors[i];
		mDescriptors[i] = 0;
	}
	mDescriptors.clear();
	mFeature_points.clear();
}

//load  key points and descriptors  
bool PICTURE::LoadKeyPointAndDes(std::string des_filename)
{
	ifstream infile(des_filename, ios::in);
	if (!infile.is_open()){
		cout << " key file open fail" <<des_filename <<endl;
		return 0;
	}

	infile >> mKeypoint_num >> mDes_length;

	int cnt = mKeypoint_num;
	while (cnt--)
	{
		SIFT_KeyPoint sift_keypt;
		infile >> sift_keypt.x >> sift_keypt.y 
			>> sift_keypt.scale >> sift_keypt.orientation;

		mFeature_points.push_back(sift_keypt);

		int des_temp;
		unsigned char*ptr_des = new unsigned char[mDes_length];
		//read the descriptor
		for (int i = 0; i < mDes_length; ++i)
		{
			infile >> des_temp;
			ptr_des[i] = (unsigned char)des_temp;
		}
		mDescriptors.push_back(ptr_des);
	}

	infile.close();
	return 1;
}

//return the key points
std::vector<SIFT_KeyPoint>& PICTURE::GetFeaturePoint()
{
	return mFeature_points;
}

//return the descriptor, store as chars
std::vector<unsigned char*>& PICTURE::GetDescriptor()
{
	return mDescriptors;
}




/****** for Class ALL_PICTURES ***/
//load all picture from the list
bool ALL_PICTURES::LoadAllPictures()
{
	ifstream infile(mDBpath +'/'+ mPictureListFile, std::ios::in);
	if (!infile.is_open()){
		cout << "Open list file fail: " << mPictureListFile << endl;
		return 0;
	}

	vector<string> pic_filename;
	string  line_in_file; 
	while ( getline(infile, line_in_file) )
	{
		istringstream words_in_line(line_in_file);
		string picture_filename;
		words_in_line >> picture_filename;

		picture_filename.erase(picture_filename.end() - 3, picture_filename.end());
		picture_filename += "key";

		pic_filename.push_back(picture_filename);
	}
	//clear the pictures and then load pictures;
	mAll_pictures.clear();
	mAll_pictures.resize(pic_filename.size());
	
	double time1 = (double)GetTickCount();

#pragma omp parallel for
	for (int i = 0; i < pic_filename.size(); i++)
	{
		//load a picture
		PICTURE pic;
		pic.LoadKeyPointAndDes(mDBpath + "/" + pic_filename[i]);

		//save the pic into the vector
		mAll_pictures[i]=pic;
	}

	time1 = (double)GetTickCount() - time1;
	cout << "load picture time: " << time1 << endl;
	return 1;
}

//clear all picture
bool ALL_PICTURES::ClearAllPictures()
{
	mAll_pictures.clear();
	return 1;
}

std::vector<PICTURE>& ALL_PICTURES::GetAllPictures()
{
	return mAll_pictures;
}

//set the string contents
bool ALL_PICTURES::SetParameters(const std::string& model_path, const std::string &list)
{
	mDBpath = model_path;
	mPictureListFile = list;
	return 1;
}