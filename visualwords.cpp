#include <fstream>
#include <windows.h>

#include"visualwords.h"

using namespace std;

//default constructor
VISUALWORDS_HANDLER::VISUALWORDS_HANDLER():
	mVisualwords_file("generic_vocabulary_100k/visual_words_sift_100k.cluster"),
	mNum_visualwords(100000),
	mK_nearest_neighbor(2)
{
	;
}

//load the 100k database visual words, save in the Mat
bool VISUALWORDS_HANDLER::LoadDBVisualWords()
{
	cv::Mat visual_words(mNum_visualwords, 128, CV_32FC1);
	ifstream instream(mVisualwords_file, std::ios::in);
	if (!instream.is_open()) {
		cout << "visual words open fail: " << mVisualwords_file << endl;
		return 0;
	}

	for (int i = 0; i < mNum_visualwords; i++)
	{
		for (int j = 0; j < 128; j++)
		{
			instream >> visual_words.ptr<float>(i)[j];
		}
	}

	mDB_visualwords = visual_words;
	return 1;
}

//build the index of db visual words
//visual words 100k, FLANN_DIST_L2
bool VISUALWORDS_HANDLER::BuildIndex()
{
	if (mDB_visualwords.empty())
	{
		cout << "visual words database is empty." << endl;
		return 0;
	}
	mVW_index.build(mDB_visualwords, cv::flann::KDTreeIndexParams(4), cvflann::FLANN_DIST_L2);
	return 1;
}

//do knn search  k=2;
bool VISUALWORDS_HANDLER::KnnSearch(vector<unsigned char*>& query_des,
	cv::Mat& indices, cv::Mat& dists, int knn)
{
	//convert the query format into cv::Mat 
	cv::Mat	query_des_mat(query_des.size(), 128, CV_8UC1);
	double time1 = (double)GetTickCount();
#pragma omp parallel for
	for (int i = 0; i < query_des.size(); i++)
	{
		memcpy_s(query_des_mat.ptr<unsigned char>(i), 128, query_des[i], 128);
	}
	time1 = (double)GetTickCount() - time1;
	cout << "visual words knn search time: " << time1 << endl;

	mVW_index.knnSearch(query_des_mat, indices, dists,
		knn, cv::flann::SearchParams(64));//check 64
	
	return 1;
}

//build the visual words's index of 3d point 
bool VISUALWORDS_HANDLER::BuildIndex3DPoints(const std::vector< FEATURE_3D_INFO >& feat_3d_infos)
{
	mIndex_3d_feature_info.clear();
	mIndex_3d_feature_info.resize(mNum_visualwords);

#pragma omp parallel for
	for (int i = 0; i < feat_3d_infos.size(); i++)
	{
		cv::Mat indices, dists;
		mVW_index.knnSearch(feat_3d_infos[i].mDescriptor, indices, dists,
			1, cv::flann::SearchParams(64));//check 64
		//for each visual words, add the current 3d point index
		for (int j = 0; j < indices.rows; j++)
		{	

			mIndex_3d_feature_info[].push_back(i);
		}
	}
}