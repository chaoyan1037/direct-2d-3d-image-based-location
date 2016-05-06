#include "picture.h"

#include <fstream>
#include <sstream>
#include <memory>
#include <omp.h>

#include "timer/timer.h"
#include "exif_reader/exif_reader.h"

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
//default constructor
PICTURE::PICTURE(): mDes_length(0), mKeypoint_num(0), mImageHeight(0), mImgaeWidth(0){
	
}

PICTURE::~PICTURE(){ 
	ClearData(); 
};

//set and get image size
void PICTURE::SetImageSize(const size_t height, const size_t width)
{
	mImgaeWidth = width;
	mImageHeight = height;
}

//get size
void PICTURE::GetImageSize(size_t& height, size_t& width) const
{
	height = mImageHeight;
	width = mImgaeWidth;
}

//clear all the data, delete the pointer content
void PICTURE::ClearData()
{
	mDescriptors.clear();
	mFeature_points.clear();
}

//load  key points and descriptors  
bool PICTURE::LoadKeyPointAndDes(const std::string& des_filename, bool bCenter_image)
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

		if (bCenter_image){
			assert(mImageHeight>0 && mImgaeWidth>0);
			// center the keypoints around the center of the image
			// first we need to get the dimensions of the image
			sift_keypt.x -= (mImgaeWidth - 1.0) / 2.0f;
			sift_keypt.y = (mImageHeight - 1.0) / 2.0f - sift_keypt.y;
		}

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
//set all empty
ALL_PICTURES::ALL_PICTURES() :mKeyfilepath(""), mImagepath(""), mPicturelistfile(""){

}

//set key and image have the same path
ALL_PICTURES::ALL_PICTURES(const std::string& key, const std::string &list)
: mKeyfilepath(key), mImagepath(key), mPicturelistfile(list){

}

//set separated path 
ALL_PICTURES::ALL_PICTURES(const std::string& key, const std::string& image, const std::string &list)
: mKeyfilepath(key), mImagepath(image), mPicturelistfile(list){

}

//destructor
ALL_PICTURES::~ALL_PICTURES(){

}

//if there already exists pictures, clear them then reload
bool ALL_PICTURES::LoadPicturesKeyFile()
{
	Timer timer;
	timer.Start();

	//first clear all pictures if already loaded
	ClearPics();

	ifstream infile(mKeyfilepath + '/' + mPicturelistfile, std::ios::in);
	if (!infile.is_open()){
		cout << "Open list file fail: " << mPicturelistfile << endl;
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
		
		pic_keyfilename.push_back(picture_filename);
	}
	infile.close();

	//clear the pictures and then load pictures;
	mPictures.clear();
	mPictures.resize(pic_keyfilename.size());
	
	//for query images, load the image size;
	if (mIsqueryimage){
		// first we need to get the dimensions of the image
		// then center the keypoints around the center of the image
		std::string img_filename;
		for (size_t i = 0; i < mPictures.size(); i++)
		{
			img_filename = mKeyfilepath + '/' + pic_keyfilename[i];
			exif_reader::open_exif(img_filename.c_str());
			//int img_width, img_height;
			//img_width = exif_reader::get_image_width();
			//img_height = exif_reader::get_image_height();
			//std::cout << "image size: " << img_width << " , " << img_height << std::endl;
			mPictures[i].SetImageSize(exif_reader::get_image_height(),
				exif_reader::get_image_width());
			exif_reader::close_exif();
		}
	}
	
#pragma omp parallel for
	for (int i = 0; i < pic_keyfilename.size(); i++)
	{
		pic_keyfilename[i].replace(pic_keyfilename[i].end() - 3, pic_keyfilename[i].end(), "key");
		//load a picture
		mPictures[i].LoadKeyPointAndDes(mKeyfilepath + "/" + pic_keyfilename[i], mIsqueryimage);
	}

	timer.Stop();
	cout << "load picture keys time: " << timer.GetElapsedTimeAsString() << endl;
	return 1;
}

//if it is for query image, make flag true
void ALL_PICTURES::SetQueryFlag(const bool flag)
{
	mIsqueryimage = flag;
}

const bool ALL_PICTURES::RetQueryFlag() const
{
	return mIsqueryimage;
}

//set the string contents
void ALL_PICTURES::SetParameters(const std::string& model_path, const std::string &list)
{
	mKeyfilepath = model_path;
	mPicturelistfile = list;
}

//load the camera pose ground truth from bundler.query.out
bool ALL_PICTURES::LoadCamerasPose(const std::string& s)
{
	Timer timer;
	timer.Start();

	std::ifstream instream(s, std::ios::in);
	if (!instream.is_open()){
		std::cout << "open bundler fail: " << s << std::endl;
		return 0;
	}
	string line;
	getline(instream, line);//header

	int num_cam = 0, num_points = 0;
	instream >> num_cam >> num_points;
	mCameras.resize(num_cam);

	for (size_t i = 0; i < num_cam; i++){
		instream >> mCameras[i].focal_length
			>> mCameras[i].k1 >> mCameras[i].k2;

		instream >> mCameras[i].rotation(0, 0)
			>> mCameras[i].rotation(0, 1)
			>> mCameras[i].rotation(0, 2)
			>> mCameras[i].rotation(1, 0)
			>> mCameras[i].rotation(1, 1)
			>> mCameras[i].rotation(1, 2)
			>> mCameras[i].rotation(2, 0)
			>> mCameras[i].rotation(2, 1)
			>> mCameras[i].rotation(2, 2);

		instream >> mCameras[i].translation(0)
			>> mCameras[i].translation(1)
			>> mCameras[i].translation(2);
	}

	instream.close();

	timer.Stop();
	cout << "load camera true pose time: " << timer.GetElapsedTimeAsString() << endl;
	return 1;
}