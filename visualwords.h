#ifndef _VISUAL_WORDS_
#define _VISUAL_WORDS_

/*
*	class to assign features to visual words using FLANN
*	can also be used for simple nearest neighbor search.
*/

#include<string>
#include<opencv2/opencv.hpp>

#include "parsebundler.h"

class VISUALWORDS_HANDLER
{
public:
	
	VISUALWORDS_HANDLER();
	~VISUALWORDS_HANDLER(){};

	//load the db visual words (100k)
	bool LoadDBVisualWords();

	//build the index of db visual words
	bool BuildIndex();

	//build the visual words's index of 3d point 
	bool BuildIndex3DPoints(const std::vector< FEATURE_3D_INFO >& feat_3d_infos);

	//knn search  k=2;
	bool KnnSearch(std::vector<unsigned char*>& query_des, 
		cv::Mat& indices, cv::Mat& dists, int knn);

private:

	/***************setting parameters**************/

	//visual words file(include path)
	std::string mVisualwords_file;

	//number of visual words(default 100k)
	int mNum_visualwords;

	//knn k=2
	int mK_nearest_neighbor;

	/***************contained data******************/

	//opencv flannIndex
	cv::flann::Index mVW_index;

	//database visual words(100k * 128float , sift) 
	cv::Mat	mDB_visualwords;

	//3d point each visual words containing
	std::vector<std::vector<int>> mIndex_3d_feature_info;
};


#endif