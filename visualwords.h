#ifndef _VISUAL_WORDS_H_
#define _VISUAL_WORDS_H_

/*
*	class to assign features to visual words using FLANN
*	can also be used for simple nearest neighbor search.
*/
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "PreProcess/parsebundler.h"

const double PI = 3.14159268979;

class VISUALWORDS_HANDLER
{
public:
	
	//default constructor
	VISUALWORDS_HANDLER() :
		mVisualwords_file("generic_vocabulary_100k/visual_words_sift_100k.cluster"),
		mNum_visualwords(100000)
	{
		;
	}

	~VISUALWORDS_HANDLER(){};
	
	//get the number of database total visual words
	const int GetNumVisualWords() const;

	//load the db visual words (100k)
	bool LoadDBVisualWords();

	//build the index of db visual words
	bool BuildIndex();

	//knn search  k=2;
	bool KnnSearch(const std::vector<SIFT_Descriptor>& query_des,
		cv::Mat& indices, cv::Mat& dists, int knn = 2) ;

private:

	/***************setting parameters**************/

	//visual words file(include path)
	std::string mVisualwords_file;

	//number of visual words(default 100k)
	int mNum_visualwords;

	/***************contained data******************/

	//openCV flannIndex
	cv::flann::Index mVW_index;

	//database visual words(100k * 128float , sift) 
	cv::Mat	mDB_visualwords;
};

struct  LOCATE_RESULT
{
	LOCATE_RESULT():located_image(0),
	num_putative_match(0),num_inlier_match(0), 
	time_findcorresp(0.0), time_computepose(0.0),
	error_rotation(0.0), error_translation(0.0){}

	bool	located_image;

	size_t	num_putative_match;
	size_t	num_inlier_match;

	double	time_findcorresp;//ms
	double	time_computepose;

	double	error_rotation;
	double	error_translation;

	BUNDLER_CAMERA cam_pose_est;
};

class VISUALWORDS_3DPOINT_HANDLER
{
//this should be private, public for debug
private:
//public:
	//find 2d-3d corresponding in Function LocateSinglePicture()
	bool FindCorrespondence(const PICTURE& picture);

	//Do query for a single picture
	bool LocateSinglePicture(const PICTURE& picture, LOCATE_RESULT& result);

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
	void SaveLocalizationResult(const std::string& s) const;

	//ratio test threshold to accept a 3d point as match
	//i.e. find two two possible 3d points by compare with 
	//its descriptors and find the most closest descriptor 
	//as the represent of this 3d point. then do ratio test
	float  mFeature_3d_point_correspondence_ratio_test_thres;

	//whether do ratio test when find feature's visual word?
	//in general, it is unnecessary since we just find the nearest one.
	bool mFeature_visual_word_correspondence_ratio_test;
	//if do ratio test, the ratio test threshold
	float mFeature_visual_word_correspondence_ratio_test_thres;

	//feature mask 1:there is 3d point correspondence  
	std::vector< bool > mFeature_3d_point_correspondence_mask;

	//feature 3d point correspondence
	//int: the index of feature; int: the index of 3d point
	std::vector< std::pair<int, int> > mFeature_3d_point_correspondence;

	//threshold: the max number of matched feature and 3d point
	//when reach this threshold then stop to calculate camera pose
	int mMaxNumberCorrespondence;

	//threshold: the minimal number of matched correspondence
	//if there are not enough 2d-3d correspondence, localize fail.
	int mMinNumberCorrespondence;

public:

	VISUALWORDS_3DPOINT_HANDLER(const std::string &bundle_path, 
		const std::string &list_txt,
		const std::string &bundle_file);

	~VISUALWORDS_3DPOINT_HANDLER();

	//inti, load database image, 3d points, and visual words
	bool Init();

	//locate all query images, call private LocateSinglePicture()
	void LocatePictures(const ALL_PICTURES& pic_query);

	//std::set< std::pair<int, int>, ViewListIsShort> mViewListSet;
	//self define compare function, compare which view list is shorter
	static struct  compareFunc
	{
		bool operator()(const std::pair<int, int> & p1, const std::pair<int, int> & p2) const
		{
			return p1.second < p2.second;
		}
	};

	//3d point each visual words contain
	//pair int:the id of 3d point, int: the length of the view list of this 3d point
	std::vector<std::set<std::pair<int, int>, compareFunc>> mVisualwords_index_3d;

	//PARSE_BUNDLER contain database image, used to init the 3d point
	PARSE_BUNDLER			mParse_bundler;
	VISUALWORDS_HANDLER		mVW_handler;

	//localization result
	size_t							mNum_totalimage;
	size_t							mNum_locatedimage;
	std::vector<LOCATE_RESULT>		mLocate_result;
};

#endif