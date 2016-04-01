#include <fstream>
#include <windows.h>

#include"visualwords.h"

using namespace std;

//get the number of database total visual words
int VISUALWORDS_HANDLER::GetNumVisualWords()
{
	return mNum_visualwords;
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
bool VISUALWORDS_HANDLER::KnnSearch(const vector<unsigned char*>& query_des,
	cv::Mat& indices, cv::Mat& dists, int knn)
{
	//convert the query format into cv::Mat 
	cv::Mat	query_des_mat(query_des.size(), 128, CV_32FC1);
	double time1 = (double)GetTickCount();
#pragma omp parallel for
	for (int i = 0; i < query_des.size(); i++)
	{
		for (int j = 0; j < 128; j++)
		{
			query_des_mat.ptr<float>(i)[j] = query_des[i][j];
		}
		
	}
	time1 = (double)GetTickCount() - time1;
	cout << "visual words knn search time: " << time1 << endl;

	mVW_index.knnSearch(query_des_mat, indices, dists,
		knn, cv::flann::SearchParams(64));//check 64
	
	return 1;
}


//build the visual words's index of 3d point 
bool VISUALWORDS_3DPOINT_HANDLER::BuildIndex3DPoints()
{
	mVisualwords_index_3d.clear();
	mVisualwords_index_3d.resize(mVW_handler.GetNumVisualWords());

//#pragma omp parallel for
	for (int i = 0; i < mParse_bundler.mFeature_infos.size(); i++)
	{
		cv::Mat indices, dists;
		mVW_handler.KnnSearch(mParse_bundler.mFeature_infos[i].mDescriptor, indices, dists, 1);
		//for each visual words, add the current 3d point index
		for (int j = 0; j < indices.rows; j++)
		{
			int vw_index_id = indices.ptr<int>(j)[0];
			assert(vw_index_id <= mVW_handler.GetNumVisualWords());
			//mVisualwords_index_3d[vw_index_id].insert(make_pair(i, (int)mParse_bundler.mFeature_infos[i].mView_list.size()));
		}
	}

	return 1;
}

void VISUALWORDS_3DPOINT_HANDLER::Init()
{
	//load the database pictures
	mPic_db.LoadAllPictures();

	mParse_bundler.ParseBundlerFile();
	mParse_bundler.LoadCameraInfo(mPic_db);

	mVW_handler.LoadDBVisualWords();
	mVW_handler.BuildIndex();

	BuildIndex3DPoints();
}