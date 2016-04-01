#ifndef _VISUAL_WORDS_
#define _VISUAL_WORDS_

/*
*	class to assign features to visual words using FLANN
*	can also be used for simple nearest neighbor search.
*/

#include <string>
#include <opencv2/opencv.hpp>

#include "parsebundler.h"


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
	int GetNumVisualWords();

	//load the db visual words (100k)
	bool LoadDBVisualWords();

	//build the index of db visual words
	bool BuildIndex();

	//knn search  k=2;
	bool KnnSearch(const std::vector<unsigned char*>& query_des,
		cv::Mat& indices, cv::Mat& dists, int knn = 2);

private:

	/***************setting parameters**************/

	//visual words file(include path)
	std::string mVisualwords_file;

	//number of visual words(default 100k)
	int mNum_visualwords;

	/***************contained data******************/

	//opencv flannIndex
	cv::flann::Index mVW_index;

	//database visual words(100k * 128float , sift) 
	cv::Mat	mDB_visualwords;
	
	
};

class VISUALWORDS_3DPOINT_HANDLER
{
public:
	VISUALWORDS_3DPOINT_HANDLER(const std::string &bundle_path, 
		const std::string &list_txt,
		const std::string &bundle_file)
	{
		mPic_db.SetParameters(bundle_path, list_txt);
		mParse_bundler.SetBundleFileName(bundle_file);
	};

	~VISUALWORDS_3DPOINT_HANDLER(){};

	void Init();

	//build the visual words's index of 3d point 
	bool BuildIndex3DPoints();

	//visual words list to rank the 3d point according to the length of its view list
	bool ViewListIsShort(const std::pair<int, int> & p1, const std::pair<int, int> & p2){
		return p1.second < p2.second;
	}
	std::set< std::pair<int, int>, decltype(ViewListIsShort)*> mViewListSet(decltype(ViewListIsShort)*);

	//3d point each visual words containing
	//pair int:the id of 3d point, int:  the length of its view_list
	std::vector < std::set< std::pair<int, int>, decltype(ViewListIsShort)*> * > mVisualwords_index_3d;

	ALL_PICTURES			mPic_db;
	PARSE_BUNDLER			mParse_bundler;

	VISUALWORDS_HANDLER		mVW_handler;

};

#endif