#include "picture.h"

#include <fstream>
#include <sstream>
#include <memory>
#include <omp.h>

#include "Timer/timer.h"

int SIFT_Descriptor::legth = 128;

using namespace std;

//calculate the distance between the two sift descriptor
int CalculateSIFTDistanceSquared(const unsigned char* d1, const unsigned char* d2)
{
	int dif, distsq = 0;
	for (int i = 0; i < 128; i++){
		dif = d1[i] - d2[i];
		distsq += dif*dif;
	}
	return distsq;
}

/****** for class PICTURE ****/

//clear all the data, delete the pointer content
void PICTURE::ClearData()
{
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
	assert(mKeypoint_num >= 0 && mDes_length >= 0);

	mFeature_points.resize(mKeypoint_num);
	mDescriptors.resize(mKeypoint_num);

	//assert(mDes_length == mDescriptors[0].legth);

	for (int cnt = 0; cnt < mKeypoint_num; cnt++)
	{
		auto& sift_keypt = mFeature_points[cnt];
		// y x  scale  orientation
		infile >> sift_keypt.y >> sift_keypt.x 
			>> sift_keypt.scale >> sift_keypt.orientation;

		//directly operate on the desc
		//do not use temp variable and then push_back it into the vector
		//since program end the temp variable will release the newed memory 
		auto& sift_desc = mDescriptors[cnt];

		sift_desc.ptrDesc = new unsigned char[sift_desc.legth];
		if (sift_desc.ptrDesc == nullptr || mDes_length != sift_desc.legth){
			std::cerr << "new error(picture.cpp, line 56)" << std::endl;
			return 0;
		}

		//read the descriptor
		int des_temp = 0;
		for (int i = 0; i < sift_desc.legth; ++i)
		{
			infile >> des_temp;
			sift_desc.ptrDesc[i] = (unsigned char)des_temp;
		}
	}

	infile.close();

	return 1;
}



/****** for Class ALL_PICTURES ***/
//load all picture from the list
bool ALL_PICTURES::LoadAllPictures()
{
	//first clear all pictures if already loaded
	ClearPics();

	ifstream infile(mDBpath +'/'+ mPictureListFile, std::ios::in);
	if (!infile.is_open()){
		cout << "Open list file fail: " << mPictureListFile << endl;
		return 0;
	}

	vector<string> pic_keyfilename;
	string  line_in_file; 
	while ( getline(infile, line_in_file) )
	{
		istringstream words_in_line(line_in_file);
		string picture_filename;
		words_in_line >> picture_filename;

		picture_filename.erase(picture_filename.begin());
		picture_filename.erase(picture_filename.begin());
		picture_filename.erase(picture_filename.end() - 3, picture_filename.end());
		picture_filename += "key";

		pic_keyfilename.push_back(picture_filename);
	}
	//clear the pictures and then load pictures;
	mPictures.clear();
	mPictures.resize(pic_keyfilename.size());
	

#pragma omp parallel for
	for (int i = 0; i < pic_keyfilename.size(); i++)
	{
		//load a picture
		mPictures[i].LoadKeyPointAndDes(mDBpath + "/" + pic_keyfilename[i]);
	}

	return 1;
}



//set the string contents
bool ALL_PICTURES::SetParameters(const std::string& model_path, const std::string &list)
{
	mDBpath = model_path;
	mPictureListFile = list;
	return 1;
}