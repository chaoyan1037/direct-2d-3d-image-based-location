#include <fstream>
#include <algorithm>
#include <vector>

#include <opencv/cv.h>

#include "visualwords.h"
#include "PreProcess/picture.h"
#include "Timer/timer.h"
#include "geometry.h"


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
	//use only one kd-tree
	mVW_index.build(mDB_visualwords, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);
	return 1;
}

//do knn search  k=2;
bool VISUALWORDS_HANDLER::KnnSearch(const vector<SIFT_Descriptor>& query_des,
	cv::Mat& indices, cv::Mat& dists, int knn)
{
	//convert the query format into cv::Mat 
	cv::Mat	query_des_mat((int)query_des.size(), 128, CV_32FC1);
	double time1 = (double)GetTickCount();
#pragma omp parallel for
	for (int i = 0; i < query_des.size(); i++)
	{
		for (int j = 0; j < query_des[i].legth; j++)
		{
			query_des_mat.ptr<float>(i)[j] = query_des[i].ptrDesc[j];
		}
		
	}
	time1 = (double)GetTickCount() - time1;
	//cout << "visual words knn search time: " << time1 << endl;
	
	mVW_index.knnSearch(query_des_mat, indices, dists, knn, cv::flann::SearchParams(10));//path number
	
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
	auto& mPicCam_db = mParse_bundler.GetAllPicturesAndCameras();
	mPicCam_db.SetParameters(bundle_path, list_txt);
	mParse_bundler.SetBundleFileName(bundle_file);
	
	mFeature_visual_word_correspondence_ratio_test = false;
	mFeature_visual_word_correspondence_ratio_test_thres = 0.7f;

	mMaxNumberCorrespondence = 100;
	mMinNumberCorrespondence = 12;
}

//initiation work, load the picture and 3d points and build the index
bool VISUALWORDS_3DPOINT_HANDLER::Init()
{
	Timer timer;
	timer.Start();
	//load the database pictures
	auto& mPicCam_db = mParse_bundler.GetAllPicturesAndCameras();
	mPicCam_db.LoadAllPictures();
	timer.Stop();
	std::cout << "Load database pictures time: " << timer.GetElapsedTimeAsString() << std::endl;

	timer.ReStart();
	mParse_bundler.ParseBundlerFile();
	mParse_bundler.LoadCameraInfo();
	timer.Stop();
	std::cout << "Parse Bundler file time: " << timer.GetElapsedTimeAsString() << std::endl;

	//after load bundler and execute LoadCameraInfo, release the pictures
	mPicCam_db.ClearPics();

	timer.ReStart();
	mVW_handler.LoadDBVisualWords();
	timer.Stop();
	std::cout << "Load visual words time: " << timer.GetElapsedTimeAsString() << std::endl;
	
	timer.ReStart();
	mVW_handler.BuildIndex();
	BuildIndex3DPoints();
	timer.Stop();
	std::cout << "Build visual words index time: " << timer.GetElapsedTimeAsString() << std::endl;

	return 1;
}


//Do query for a single picture and find correspondence between 
//its 2d features and database 3d points
//success return 1, else return 0
bool VISUALWORDS_3DPOINT_HANDLER::FindCorrespondence(const PICTURE& picture)
{
	auto& pic_feat_desc = picture.GetDescriptor();

	mFeature_3d_point_correspondence.clear();
	mFeature_3d_point_correspondence_mask.clear();
	mFeature_3d_point_correspondence_mask.resize(pic_feat_desc.size(), true);

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

	if (pic_feat_desc.size() != indices.rows){
		std::cerr << "error:visualwords.cpp line 166" << std::endl;
		abort();
	}

	//feature matched 3d point
	std::vector<int> feat_matched_3d_point(pic_feat_desc.size(), -1);

	//if #matched feature exceeds the threshold then stop find matched 3d point
	int cnt_matched_feature = 0;

#pragma omp parallel for shared(cnt_matched_feature)
	//for matched visual words, find feature's matched 3d points
	for (int i = 0; i < indices.rows; i++)
	{
		if (!mFeature_3d_point_correspondence_mask[i] || cnt_matched_feature > mMaxNumberCorrespondence) continue;

		//first let the mask be false
		mFeature_3d_point_correspondence_mask[i] = false;

		//the squared distance of current feature to 3d point's feature
		int min_distance_squared[2] = { 100000000, 100000000 };
		int min_distance_3d_point_index[2] = {-1, -1};

		int vw_index_id = indices.ptr<int>(i)[0];
		
		//for each visual words find all 3d point pair<int, int>
		for (auto pair_3d_point : mVisualwords_index_3d[vw_index_id])
		{
			int index_3d_point = pair_3d_point.first;
			const FEATURE_3D_INFO &feat_3d_info = mParse_bundler.GetFeature3DInfo()[index_3d_point];
			const std::vector<SIFT_Descriptor>& _3d_point_feat_desc = feat_3d_info.mDescriptor;
			
			//for each 3d point, find all its descriptors
			//find one smallest distance represent this 3d point
			//only record the squared distance, since the index is index_3d_point
			int min_distance_squared_each_3d_point = 100000000;
			for (int j = 0; j < _3d_point_feat_desc.size(); j++)
			{
				using std::min;
				int distsq_temp = CalculateSIFTDistanceSquared(pic_feat_desc[i].ptrDesc, _3d_point_feat_desc[j].ptrDesc);
				min_distance_squared_each_3d_point = min(distsq_temp, min_distance_squared_each_3d_point);
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
		if (min_distance_3d_point_index[1] > 0 && 10 * 10 * min_distance_squared[0] < 7 * 7 * min_distance_squared[1])
		{
			mFeature_3d_point_correspondence_mask[i] = true;
			mFeature_3d_point_correspondence.push_back(make_pair(i, min_distance_3d_point_index[0]));
			//check if there are enough correspondence and stop now
			#pragma omp atomic 
			++cnt_matched_feature;
		}

	}

	//no enough correspondence, location fail
	if (cnt_matched_feature < mMinNumberCorrespondence){
		mFeature_3d_point_correspondence.clear();
		mFeature_3d_point_correspondence_mask.clear();
		return 0;
	}
	else return 1;
}

//the public function to locate a single picture
bool VISUALWORDS_3DPOINT_HANDLER::LocateSinglePicture(const PICTURE& picture){
	//0: can not find enough 2d-3d correspondence
	if (0 == FindCorrespondence(picture)){
		std::cout << "not enough putative matches" << std::endl;
		return 0;
	}

	Geometry geo;
	//geo.match_2d_3d
	auto& mat_2d_3d = geo.ReturnMatch_2d_3d();
	for (auto pa : mFeature_3d_point_correspondence){
		auto& pt_2d = picture.GetFeaturePoint()[pa.first];
		auto& pt_3d = mParse_bundler.GetFeature3DInfo()[pa.second].mPoint;
		mat_2d_3d.push_back(std::make_pair(
			cv::Vec2d(pt_2d.x, pt_2d.y),
			cv::Vec3d(pt_3d.x, pt_3d.y, pt_3d.z)));
	}

	//if there are no intrinsics then use DLT
	geo.ComputePoseDLT();
	
	return 1;
}