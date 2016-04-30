
#include "parsebundler.h"

#include <string.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

//using namespace std;
using std::ifstream;
using std::ofstream;
using std::istringstream;
using std::cout;
using std::endl;
using std::string;


PARSE_BUNDLER::PARSE_BUNDLER()
{
	mNumbPoints = 0;
	mNumCameras = 0;
}

PARSE_BUNDLER::~PARSE_BUNDLER()
{
	mAll_pic_cameras.ClearPicsCameras();
	mFeature_infos.clear();
}

//parse the bundler file
/*Each camera entry contains the estimated camera intrinsics and extrinsics,
  and has the form:
<f> <k1> <k2>   [the focal length, followed by two radial distortion coeffs]
<R>             [a 3x3 matrix representing the camera rotation]
<t>             [a 3-vector describing the camera translation]

Each point entry has the form
<position>      [a 3-vector describing the 3D position of the point]
<color>         [a 3-vector describing the RGB color of the point]
<view list>     [a list of views the point is visible in]

The view list begins with the length of the list 
(i.e., the number of cameras the point is visible in).
The list is then given as a list of quadruplets <camera> <key> <x> <y>
The pixel positions are floating point numbers in a coordinate system 
where the origin is the center of the image, the x-axis increases 
to the right, and the y-axis increases towards the top of the image. 
Thus, (-w/2, -h/2) is the lower-left corner of the image, 
and (w/2, h/2) is the top-right corner (where w and h are the width 
and height of the image).
*/

bool PARSE_BUNDLER::ParseBundlerFile()
{
	std::ifstream instream(mBundle_file, std::ios::in);
	if (!instream.is_open()){
		std::cout << "open bundler fail: " << mBundle_file << std::endl;
		return 0;
	}

	string line_in_file;
	getline(instream, line_in_file);//header

	instream >> mNumCameras >> mNumbPoints;

	auto & mCameras = mAll_pic_cameras.GetAllCameras();
	//mCameras.clear();
	mCameras.resize(mNumCameras);
	//mFeature_infos.clear();
	mFeature_infos.resize(mNumbPoints);

	//load the camera parameters 
	for (int i = 0; i < mNumCameras; i++)
	{
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
		
		//mCameras[i].id = i;
	}
	//return 1;
	//load the points
	int r, g, b;
	for (int i = 0; i < mNumbPoints; i++)
	{
		instream >> mFeature_infos[i].mPoint.x
			>> mFeature_infos[i].mPoint.y
			>> mFeature_infos[i].mPoint.z
			>> r >> g >> b;
		mFeature_infos[i].mPoint.r = (int)r;
		mFeature_infos[i].mPoint.g = (int)g;
		mFeature_infos[i].mPoint.b = (int)b;
		
		int view_lenth = 0;
		instream >> view_lenth;
		mFeature_infos[i].mView_list.resize(view_lenth);
		mFeature_infos[i].mDescriptor.resize(view_lenth);

		for (int j = 0; j < view_lenth; j++)
		{
			instream >> mFeature_infos[i].mView_list[j].camera
				>> mFeature_infos[i].mView_list[j].key
				>> mFeature_infos[i].mView_list[j].x
				>> mFeature_infos[i].mView_list[j].y;
		}
	}

	instream.close();
	return 1;
}

//load the .key info 
bool PARSE_BUNDLER::LoadCameraInfo()
{
	auto& picture = mAll_pic_cameras.GetAllPictures();
	//make sure that database #pictures equal to #cameras
	assert(picture.size() == mAll_pic_cameras.GetAllCameras().size());

#pragma omp parallel for
	for (int i = 0; i < mNumbPoints; ++i)
	{
		for (int j = 0; j < mFeature_infos[i].mView_list.size(); ++j)
		{
			auto & view = mFeature_infos[i].mView_list[j];
			auto & keypoint_vec = picture[mFeature_infos[i].mView_list[j].camera].GetFeaturePoint();

			//do not use SIFT x, y coordinates, of which the origin is the left-up corner
			//use the bundler x, y coordinates, of which the origin is the center of image
			view.scale			= keypoint_vec[view.key].scale;
			view.orientation	= keypoint_vec[view.key].orientation;

			//mDescriptor: the jth descriptor
			auto & descriptor = picture[mFeature_infos[i].mView_list[j].camera].GetDescriptor();

			mFeature_infos[i].mDescriptor[j] = std::move(descriptor[view.key]);
		}
	}

	return 1;
}


//after load original bundler file, mask the query image
void PARSE_BUNDLER::FindQueryPicture(const std::string& s)
{
	ifstream is(s, std::ios::in);
	if (is.is_open() == 0){
		std::cerr << "query_picture_list.txt open fail: "<< s << endl;
		return;
	}

	mPic_query_mask.clear();
	string line;
	while (is >> line)
	{
		//stringstream istrstream;
		mPic_query_mask.push_back( line[0] == 'q' );
	}

}

//
void PARSE_BUNDLER::WriteQueryBundler(const std::string& s) const
{
	ofstream os(s, std::ios::out | std::ios::trunc);
	if (0 == os.is_open()){
		std::cerr << " open query bundle file fail: " << s << endl;
		return;
	}

	auto & mCameras = mAll_pic_cameras.GetAllCameras();

	os << "# Bundle file v0.3" << endl;

	size_t num_cam = 0;
	//count to number of cameras
	for (size_t i = 0; i < mPic_query_mask.size(); i++){
		if (1 == mPic_query_mask[i])
			num_cam++;
	}
	os << num_cam << " " << int(0) << endl;

	for (size_t i = 0; i < mPic_query_mask.size(); i++){
		if (1 == mPic_query_mask[i]){
			os  << mCameras[i].focal_length 
				<< mCameras[i].k1 << mCameras[i].k2 
				<< endl;
			os  << mCameras[i].rotation(0, 0)
				<< mCameras[i].rotation(0, 1)
				<< mCameras[i].rotation(0, 2) 
				<< endl;
			os  << mCameras[i].rotation(1, 0)
				<< mCameras[i].rotation(1, 1)
				<< mCameras[i].rotation(1, 2) 
				<< endl;
			os  << mCameras[i].rotation(2, 0)
				<< mCameras[i].rotation(2, 1)
				<< mCameras[i].rotation(2, 2) 
				<< endl;
			os  << mCameras[i].translation(0)
				<< mCameras[i].translation(1)
				<< mCameras[i].translation(2) 
				<< endl;
		}
	}
	os.close();
}