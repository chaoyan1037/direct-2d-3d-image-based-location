#include <fstream>
#include <limits>
#include <algorithm>
#include <windows.h>

#include "visualwords.h"
#include "picture.h"

using namespace std;


//get the number of database total visual words
int VISUALWORDS_HANDLER::GetNumVisualWords() const
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
	cv::Mat	query_des_mat((int)query_des.size(), 128, CV_32FC1);
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
	//cout << "visual words knn search time: " << time1 << endl;
	
	mVW_index.knnSearch(query_des_mat, indices, dists,
		knn, cv::flann::SearchParams(64, 0.0f, true));//check 64
	
	return 1;
}


//build the visual words's index of 3d point 
bool VISUALWORDS_3DPOINT_HANDLER::BuildIndex3DPoints()
{
	int num_visualwords = mVW_handler.GetNumVisualWords();
	mVisualwords_index_3d.clear();
	mVisualwords_index_3d.resize(num_visualwords);

	const std::vector< FEATURE_3D_INFO > &feature_info = mParse_bundler.GetFeature3DInfo();

#pragma omp parallel for
	for (int i = 0; i < feature_info.size(); i++)
	{
		cv::Mat indices, dists;
		mVW_handler.KnnSearch(feature_info[i].mDescriptor, indices, dists, 1);
		//for each visual words, add the current 3d point index
		for (int j = 0; j < indices.rows; j++)
		{
			int vw_index_id = indices.ptr<int>(j)[0];
			assert(vw_index_id < num_visualwords);
			mVisualwords_index_3d[vw_index_id].insert(make_pair(i, (int)feature_info[i].mView_list.size()));
		}
	}

	return 1;
}

/********** class VISUALWORDS_3DPOINT_HANDLER**************/
//constructor
VISUALWORDS_3DPOINT_HANDLER::VISUALWORDS_3DPOINT_HANDLER(const std::string &bundle_path,
	const std::string &list_txt,
	const std::string &bundle_file)
{
	mPic_db.SetParameters(bundle_path, list_txt);
	mParse_bundler.SetBundleFileName(bundle_file);
	
	mFeature_visual_word_correspondence_ratio_test = false;
	mFeature_visual_word_correspondence_ratio_test_thres = 0.7f;

	mMaxNumberCorrespndence = 100;

}

//initiation work, load the picture and 3d points and build the index
bool VISUALWORDS_3DPOINT_HANDLER::Init()
{
	//load the database pictures
	mPic_db.LoadAllPictures();

	mParse_bundler.ParseBundlerFile();
	mParse_bundler.LoadCameraInfo(mPic_db);
	
	//after load bundler and execute LoadCameraInfo, release the mPic_db
	mPic_db.ClearAllPictures();

	mVW_handler.LoadDBVisualWords();
	mVW_handler.BuildIndex();

	BuildIndex3DPoints();

	return 1;
}


//Do query for a single picture
bool VISUALWORDS_3DPOINT_HANDLER::LocateSinglePicture(const PICTURE& picture)
{
	mFeature_3d_point_correspondence.clear();
	mFeature_3d_point_correspondence_mask.clear();
	mFeature_3d_point_correspondence_mask.resize(picture.GetDescriptor().size(), true);

	const std::vector<unsigned char*>&pic_feat_desc = picture.GetDescriptor();

	cv::Mat indices, dists;
	if (mFeature_visual_word_correspondence_ratio_test){
		mVW_handler.KnnSearch(pic_feat_desc, indices, dists, 2);
		//do ratio test dists[0] smaller than dists[1]
		for (int i = 0; i < indices.rows; i++){
			//find those false match
			if (dists.ptr<float>(i)[0] > 
				dists.ptr<float>(i)[1] * mFeature_visual_word_correspondence_ratio_test_thres)
			{
				mFeature_3d_point_correspondence_mask[i] = false;
			}
		}
	}
	else { mVW_handler.KnnSearch(pic_feat_desc, indices, dists, 1); }

	assert(pic_feat_desc.size() == indices.rows);

	//feature matched 3d point
	std::vector<int> feat_matched_3d_point(pic_feat_desc.size(), -1);

	//if #matched feature exceeds the threshold then stop find matched 3d point
	int cnt_matched_feature = 0;

//#pragma  omp parallel for
	//for matched visual words, find feature's matched 3d points
	for (int i = 0; i < indices.rows && mFeature_3d_point_correspondence_mask[i]; i++)
	{
		//first let the mask be false
		mFeature_3d_point_correspondence_mask[i] = false;

		//the squared distance of current feature to 3d point's feature
		int min_distance_squared[2] = { INT_MAX };
		int min_distance_3d_point_index[2] = { -1 };

		int vw_index_id = indices.ptr<int>(i)[0];
		
		//for each visual words find all 3d point pair<int, int>
		for (auto pair_3d_point : mVisualwords_index_3d[vw_index_id])
		{
			int index_3d_point = pair_3d_point.first;
			const FEATURE_3D_INFO &feat_3d_info = mParse_bundler.GetFeature3DInfo()[index_3d_point];
			const std::vector<unsigned char*>& _3d_point_feat_desc = feat_3d_info.mDescriptor;
			
			//for each 3d point, find all its descriptors
			//find one smallest distance represent this 3d point
			//only record the squared distance, since the index is index_3d_point
			int min_distance_squared_each_3d_point = INT_MAX;
			for (int j = 0; j < _3d_point_feat_desc.size(); j++)
			{
				int distsq_temp = CalculateSIFTDistanceSquared(pic_feat_desc[i], _3d_point_feat_desc[j]);
				min_distance_squared_each_3d_point = std::min(distsq_temp, min_distance_squared_each_3d_point);
			}

			//after get this 3d point smallest distance
			//compare this 3d point to current smallest distances
			//and always keep the smallest distances 3d point
			if (min_distance_squared_each_3d_point < min_distance_squared[1])
			{
				min_distance_squared[1] = min_distance_squared_each_3d_point;
				min_distance_3d_point_index[1] = index_3d_point;

				if (min_distance_squared[1] < min_distance_squared[0]){
					std::swap(min_distance_squared[0], min_distance_squared[1]);
					std::swap(min_distance_3d_point_index[0], min_distance_3d_point_index[1]);
				}
			}
		}

		//after find two putative matched 3d points do ratio test     
		//mFeature_3d_point_correspondence_ratio_test_thres
		//Check whether closest distance is less than 0.7 of second.
		if (10 * 10 * min_distance_squared[0] < 7 * 7 * min_distance_squared[1])
		{
			mFeature_3d_point_correspondence_mask[i] = true;
			mFeature_3d_point_correspondence.push_back(make_pair(i, min_distance_3d_point_index[0]));
			//check if there are enough correspondence and stop now
			if (++cnt_matched_feature >= mMaxNumberCorrespndence){
				return 1;
			}
		}

	}

	return 1;
}
