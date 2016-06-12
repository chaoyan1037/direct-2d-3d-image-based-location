
#ifndef _VISUAL_WORDS_H_
#define _VISUAL_WORDS_H_

/*
*	class to assign features to visual words using FLANN
*	can also be used for simple nearest neighbor search.
*/
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "preprocess/parsebundler.h"

const double PI = 3.14159268979;

class VISUALWORDS_HANDLER
{
public:
	
	//default constructor
	VISUALWORDS_HANDLER();

	~VISUALWORDS_HANDLER();
	
	//get the number of database total visual words
	const int GetNumVisualWords() const;

	//load the db visual words (100k)
	bool LoadDBVisualWords();

	//build the index of db visual words
	bool BuildIndex();

	//knn search  k=2;
	bool KnnSearch(const std::vector<SIFT_Descriptor>& query_des,
		cv::Mat& indices, cv::Mat& dists, int knn = 2);

private:

	/***************setting parameters**************/

	// visual words file(include path)
	std::string mVisualwords_file;

	// number of visual words(default 100k)
	int mNum_visualwords;

	/***************contained data******************/

	// openCV flannIndex
	cv::flann::Index mVW_index;

	//database visual words(100k * 128float , sift) 
	cv::Mat	mDB_visualwords;
};

//struct to store data for a query picture
struct  LOCATE_RESULT
{
	LOCATE_RESULT() :have_intrinsics(0),located_image(0),
		num_putative_match(0),num_inlier_match(0), 
		time_findcorresp(0.0), time_computepose(0.0),
		error_rotation(0.0), error_center(0.0)
	{
		K = K.zeros();
		rotation = rotation.zeros();
		center[0] = 0.0; center[1] = 0.0; center[2] = 0.0;
	}

	bool	have_intrinsics;
	bool	located_image;

	size_t	num_putative_match;
	size_t	num_inlier_match;

	double	time_findcorresp;//ms
	double	time_computepose;

	double	error_rotation;
	double	error_center;

	// if have intrinsics, then use epnp, else use DLT and estimate K
	cv::Matx33d	K; 
	cv::Matx33d	rotation;
	cv::Vec3d	center;

	//feature 3d point correspondence
	//int: the index of feature; int: the index of 3d point
	std::vector< std::pair<size_t, size_t> > mFeature_3d_point_correspondence;

  // putative correspondence
  std::map< size_t, std::vector<int> >  mFeat_3DPt_corres_record;
};

class VISUALWORDS_3DPOINT_HANDLER
{
//this should be private, public for debug
private:
//public:
	//find 2d-3d corresponding in Function LocateSinglePicture()
	bool FindCorrespondence(const size_t loca_res_index, const PICTURE& picture);

	//Do query for a single picture
	bool LocateSinglePicture(const size_t loca_res_index, const PICTURE& picture);

	//build the visual words's index of 3d point 
	bool BuildIndex3DPoints();

	//save and load the index
	//format:
	//num_of_records
	//visual_words_index pair(i1, j1)... pair(in, jn)
	//...
	bool SaveIndex3DPoints(const std::string& s) const;
	bool LoadIndex3DPoints(const std::string& s);

	//save the localization result
	//0  no, 1: RT, 2 RTK
	void SaveLocalizationResult(const std::string& s, 
		const ALL_PICTURES& pic_cam_query, const int iReportRTK = 0) const;

  // save the correspondence record
  void SaveCorresRecord( const std::string& strCorresRecord ) const;

	//ratio test threshold to accept a 3d point as match
	//i.e. find two two possible 3d points by compare with 
	//its descriptors and find the most closest descriptor 
	//as the represent of this 3d point. then do ratio test
	float  mFeature_3d_point_correspondence_ratio_test_thres;

	//threshold: the max number of matched feature and 3d point
	//when reach this threshold then stop to calculate camera pose
	int mMaxNumberCorrespondence;

	//threshold: the minimal number of matched correspondence
	//if there are not enough 2d-3d correspondence, localize fail.
	int mMinNumberCorrespondence;

	//the method to represent the 3d point
	//0: integer mean per visual words
	//1: use all descriptors
	int	mPoint3D_method;

public:

	VISUALWORDS_3DPOINT_HANDLER(const std::string &bundle_path, 
		const std::string &list_txt,
		const std::string &bundle_file);

	~VISUALWORDS_3DPOINT_HANDLER();

	//inti, load database image, 3d points, and visual words
	bool Init();

	//locate all query images, call private LocateSinglePicture()
	void LocatePictures(const ALL_PICTURES& pic_query);

  // test the performance of the CasHash
  void TestCasHash( const ALL_PICTURES& pic_query );

	//3d point each visual words contained
	//size is the num of visual words
	//pair int:the id of 3d point, int: the index of feat in mAll_descriptor
	std::vector< std::set< std::pair<size_t, size_t> > > mVisualwords_index_3d;
	std::vector< POINT3D >			    mPoint3D;
	std::vector< SIFT_Descriptor >	mAll_descriptor;

	//PARSE_BUNDLER contain database image, used to init the 3d point
	PARSE_BUNDLER					    mParse_bundler;
	VISUALWORDS_HANDLER				mVW_handler;

	//localization result
	size_t							          mNum_totalimage;
	size_t							          mNum_locatedimage;
	std::vector<LOCATE_RESULT>		mLocate_result;

};



#endif