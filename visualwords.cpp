#include <fstream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <assert.h>
#include <opencv/cv.h>

#include "visualwords.h"
#include "preprocess/picture.h"
#include "timer/timer.h"
#include "geometry.h"
#include "global.h"

using std::ifstream;
using std::ofstream;
using std::endl;

using global::cout;

/************ class VISUALWORDS_HANDLER ***************/
//default constructor
VISUALWORDS_HANDLER::VISUALWORDS_HANDLER() 
:mVisualwords_file("generic_vocabulary_100k/visual_words_sift_100k.cluster"),
mNum_visualwords(100000)
{
	
}

VISUALWORDS_HANDLER::~VISUALWORDS_HANDLER()
{

}

//get the number of database total visual words
const int VISUALWORDS_HANDLER::GetNumVisualWords() const
{
	return mNum_visualwords;
}

//load the 100k database visual words, save in the Mat
bool VISUALWORDS_HANDLER::LoadDBVisualWords()
{
	cv::Mat visual_words(mNum_visualwords, 128, CV_32FC1);
	ifstream instream(mVisualwords_file, std::ios::in);
	if (!instream.is_open()) {
		global::cout << "visual words open fail: " << mVisualwords_file << endl;
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
		global::cout << "visual words database is empty." << endl;
		return 0;
	}
	//use only one kd-tree
	mVW_index.build(mDB_visualwords, cv::flann::KDTreeIndexParams(1), cvflann::FLANN_DIST_L2);
	return 1;
}

//do knn search  k=2;
bool VISUALWORDS_HANDLER::KnnSearch(const std::vector<SIFT_Descriptor>& query_des,
	cv::Mat& indices, cv::Mat& dists, int knn)
{
	//convert the query format into cv::Mat 
	cv::Mat	query_des_mat((int)query_des.size(), 128, CV_32FC1);
	double time1 = (double)GetTickCount();
//#pragma omp parallel for
	for (int i = 0; i < query_des.size(); i++)
	{
		for (int j = 0; j < query_des[i].legth; j++)
		{
			query_des_mat.ptr<float>(i)[j] = query_des[i].ptrDesc[j];
		}
	}
	time1 = (double)GetTickCount() - time1;
	//global::cout << "visual words knn search time: " << time1 << endl;
	
	mVW_index.knnSearch(query_des_mat, indices, dists, knn, cv::flann::SearchParams(10));//path number
	
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

//destructor
VISUALWORDS_3DPOINT_HANDLER::~VISUALWORDS_3DPOINT_HANDLER()
{

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
			mVisualwords_index_3d[vw_index_id].insert(std::make_pair(i, (int)feature_info[i].mView_list.size()));
		}
	}

	return 1;
}

//after build the index, then save it into file.
//format:
//num_of_records
//visual_words_index pair(i1, j1)... pair(in, jn)
//...
bool VISUALWORDS_3DPOINT_HANDLER::SaveIndex3DPoints(const std::string&s) const
{
	std::ofstream of(s, std::ios::out | std::ios::trunc);
	if (0 == of.is_open()){
		global::cout << " open index_3d_points file fail: " << s << std::endl;
		return 0;
	}

	size_t num_non_empty = 0;
	size_t num_visualwords = mVisualwords_index_3d.size();
	for (size_t i = 0; i < num_visualwords; i++){
		if (0 == mVisualwords_index_3d[i].empty()){
			num_non_empty++;
		}
	}

	of << num_non_empty << std::endl;
	for (size_t i = 0; i < num_visualwords; i++){
		if (0 == mVisualwords_index_3d[i].empty()){
			of << i << " ";
			for (auto pairii : mVisualwords_index_3d[i]){
				of << pairii.first << " " << pairii.second << " ";
			}
			of << std::endl;
		}
	}

	of.close();
	return 1;
}

bool VISUALWORDS_3DPOINT_HANDLER::LoadIndex3DPoints(const std::string& s)
{
	std::ifstream is(s, std::ios::in);
	if (0 == is.is_open()){
		global::cout << " open index_3d_points file fail: " << s << std::endl;
		return 0;
	}
	size_t num_VM_records = 0;
	is >> num_VM_records;
	assert(mVW_handler.GetNumVisualWords() >= num_VM_records);
	mVisualwords_index_3d.resize(mVW_handler.GetNumVisualWords());

	//std::vector<std::set<std::pair<int, int>, compareFunc>> mVisualwords_index_3d;
	std::string line;
	size_t VW_index = 0;
	for (size_t i = 0; i < num_VM_records; i++){
		is >> VW_index;
		getline(is, line);
		std::istringstream istream(line);
		int num_1 = 0, num_2 = 0;
		while (istream>>num_1)
		{
			if (istream >> num_2){
				mVisualwords_index_3d[VW_index].insert(std::make_pair(num_1, num_2));
			}
			else return 0;
		}
	}
	is.close();
	return 1;
}


//initiation work, load the picture and 3d points and build the index
bool VISUALWORDS_3DPOINT_HANDLER::Init()
{
	Timer timer;
	timer.Start();
	mParse_bundler.ParseBundlerFile();

	//load the desc directly from parsed_bundler.txt
	ifstream is_desc("parsed_bundler.txt", std::ios::_Nocreate);
	if (is_desc.is_open() && mParse_bundler.LoadFeature3DInfro("parsed_bundler.txt")){
		global::cout << "load parsed_bundler.txt" << endl;
		is_desc.close();
	}
	else{
		//load the database pictures
		auto& mPicCam_db = mParse_bundler.GetAllPicturesAndCameras();
		mPicCam_db.LoadPicturesKeyFile();
		mParse_bundler.LoadCameraInfo();
		//after load bundler and execute LoadCameraInfo, release the pictures
		mPicCam_db.ClearPics();
		mParse_bundler.SaveFeature3DInfro("parsed_bundler.txt");
		global::cout << "load .key files and save parsed_bundler.txt" << endl;
	}

	timer.Stop();
	global::cout << "Parse Bundler file time: " << timer.GetElapsedTimeAsString() << std::endl;


	timer.Start();
	mVW_handler.LoadDBVisualWords();
	timer.Stop();
	global::cout << "Load visual words time: " << timer.GetElapsedTimeAsString() << std::endl;
	
	timer.Start();
	mVW_handler.BuildIndex();

	ifstream is("index_3dpoints.txt", std::ios::_Nocreate);
	if (1 == is.is_open() && LoadIndex3DPoints("index_3dpoints.txt"))
	{
		global::cout << "load index_3dpoints.txt" << endl;
		is.close();
	}
	else{
		BuildIndex3DPoints();
		SaveIndex3DPoints("index_3dpoints.txt");
		global::cout << "build index and save index_3dpoints.txt" << endl;
	}
	timer.Stop();
	global::cout << "Build visual words index time: " << timer.GetElapsedTimeAsString() << std::endl;

	return 1;
}


//Do query for a single picture and find correspondence between 
//its 2d features and database 3d points
//success return 1, else return 0
bool VISUALWORDS_3DPOINT_HANDLER::FindCorrespondence(const PICTURE& picture)
{
	auto& pic_feat_desc = picture.GetDescriptor();

	mFeature_3d_point_correspondence.clear();
	mFeature_3d_point_correspondence_mask.assign(pic_feat_desc.size(), true);

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
		std::cerr << "error:visualwords.cpp line 281" << std::endl;
		return false;
	}
	global::cout << "start find feature's corresp" << std::endl;

	//feature matched 3d point
	std::vector<int> feat_matched_3d_point(pic_feat_desc.size(), -1);

	//if #matched feature exceeds the threshold then stop find matched 3d point
	int cnt_matched_feature = 0;

//#pragma omp parallel for shared(cnt_matched_feature)
	//for matched visual words, find feature's matched 3d points
	for (int i = 0; i < indices.rows; i++)
	{
		if (false == mFeature_3d_point_correspondence_mask[i] || cnt_matched_feature > mMaxNumberCorrespondence) continue;

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
			const std::vector<SIFT_Descriptor>& point_3d_feat_desc = feat_3d_info.mDescriptor;
			
			//for each 3d point, find all its descriptors
			//find one smallest distance represent this 3d point
			//only record the squared distance, since the index is index_3d_point
			int min_distance_squared_each_3d_point = 100000000;
			for (int j = 0; j < point_3d_feat_desc.size(); j++)
			{
				using std::min;
				int distsq_temp = CalculateSIFTDistanceSquared(pic_feat_desc[i].ptrDesc, point_3d_feat_desc[j].ptrDesc);
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
		if (min_distance_3d_point_index[1] > 0 && 
			10 * 10 * min_distance_squared[0] < 8 * 8 * min_distance_squared[1])
		{
			mFeature_3d_point_correspondence_mask[i] = true;
			mFeature_3d_point_correspondence.push_back(std::make_pair(i, min_distance_3d_point_index[0]));
			//check if there are enough correspondence and stop now
//#pragma omp atomic 
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
bool VISUALWORDS_3DPOINT_HANDLER::LocateSinglePicture(const PICTURE& picture, LOCATE_RESULT& result)
{
	Timer timer;
	timer.Start();
	//0: can not find enough 2d-3d correspondence
	global::cout << "start find corresp" << std::endl;
	if (0 == FindCorrespondence(picture)){
		global::cout << "not enough putative matches" << std::endl;
		timer.Stop();
		result.time_findcorresp = timer.GetElapsedTimeMilliSecond();
		return 0;
	}
	timer.Stop();
	result.time_findcorresp = timer.GetElapsedTimeMilliSecond();
	
	global::cout << "after find corresp" << std::endl;

	timer.ReStart();
	
	double sign = 1.0;
	//if have intrinsics, use epnp, need reverse the sign of 3d points coordinate
	if (true == result.have_intrinsics){
		sign = -1.0;
	}

	Geometry geo;
	//geo.match_2d_3d
	auto& mat_2d_3d = geo.ReturnMatch_2d_3d();
	mat_2d_3d.reserve(200);
	for (auto pa : mFeature_3d_point_correspondence)
	{
		auto& pt_2d = picture.GetFeaturePoint()[pa.first];
		auto& pt_3d = mParse_bundler.GetFeature3DInfo()[pa.second].mPoint;
		mat_2d_3d.push_back(std::make_pair(
			cv::Vec2d(pt_2d.x, pt_2d.y),
			cv::Vec3d(sign*pt_3d.x, pt_3d.y, sign*pt_3d.z)));
	}
	global::cout << "geo" << std::endl;

	result.num_putative_match = mat_2d_3d.size();

	//if have intrinsics then use epnp
	if (true == result.have_intrinsics){
		global::cout << "compute pose epnp" << std::endl;
		geo.SetK(result.K);
		//geo.SetIntrinsicParameter(result.K(0, 0), 0, 0);
		result.num_inlier_match = geo.ComputePoseEPnP();
		geo.GetRT(result.rotation, result.translation);
		//transform back
		result.rotation(0, 1) = -result.rotation(0, 1);
		result.rotation(1, 0) = -result.rotation(1, 0);
		result.rotation(1, 2) = -result.rotation(1, 2);
		result.rotation(2, 1) = -result.rotation(2, 1);
		
	}
	else{
		global::cout << "compute pose dlt" << std::endl;
		result.num_inlier_match = geo.ComputePoseDLT();
		geo.GetRT(result.rotation, result.translation);
		result.rotation(0, 0) = -result.rotation(0, 0);
		result.rotation(0, 1) = -result.rotation(0, 1);
		result.rotation(0, 2) = -result.rotation(0, 2);
		result.rotation(2, 0) = -result.rotation(2, 0);
		result.rotation(2, 1) = -result.rotation(2, 1);
		result.rotation(2, 2) = -result.rotation(2, 2);

		//only for DLT, get the estimated K
		geo.GetK_est(result.K);
	}
	result.translation[0] = -result.translation[0];
	result.translation[2] = -result.translation[2];

	timer.Stop();
	global::cout << " locate single image time: " << timer.GetElapsedTimeMilliSecond() << endl;
	result.time_computepose = timer.GetElapsedTimeMilliSecond() - result.time_findcorresp;
	
	if (0 == result.num_inlier_match){ return 0; }

	result.located_image = true;	
	return 1;
}

void VISUALWORDS_3DPOINT_HANDLER::LocatePictures(const ALL_PICTURES& pic_cam_query)
{
	Timer timer;
	timer.Start();

	auto & pic_query		= pic_cam_query.GetAllPictures();
	auto & cam_query_true	= pic_cam_query.GetAllCameras();
	auto & pic_query_focal	= pic_cam_query.GetAllQueryFocal();

	assert(pic_cam_query.RetQueryFlag());
	assert(pic_query.size() == cam_query_true.size());
	assert(pic_query.size() == pic_query_focal.size());

	mNum_totalimage = pic_query.size();
	mNum_locatedimage = 0;
	mLocate_result.resize(mNum_totalimage);
	
	//can not use openMP parallel here, since VISUALWORDS_3DPOINT_HANDLER is shared
	for (int i = 0; i < pic_query.size(); i++)
	{
		global::cout << "start locating image " << i << endl;
		if (pic_query_focal[i] > 0){
			mLocate_result[i].have_intrinsics = true;
			mLocate_result[i].K(0, 0) = pic_query_focal[i];
			mLocate_result[i].K(1, 1) = pic_query_focal[i];
			int h = 0, w = 0;
			pic_query[i].GetImageSize(h, w);
			mLocate_result[i].K(0, 2) = (w - 1) >> 1;
			mLocate_result[i].K(1, 2) = (h - 1) >> 1;
			mLocate_result[i].K(2, 2) = 1.0;
		}
		LocateSinglePicture(pic_query[i], mLocate_result[i]);
		if (0 == mLocate_result[i].located_image) continue;
		global::cout << "success locating image " << i << endl;

		mNum_locatedimage++;

		double quat[4];
		//calculate the rotation err
		cv::Matx33d R_diff = (cam_query_true[i].rotation * mLocate_result[i].rotation.t());
		RotationToQuaterion(R_diff, quat);
		global::cout << "after rotation to quat " << endl;

		double cos_half_phi, sin_half_phi;
		cos_half_phi = quat[0];
		sin_half_phi = std::sqrt(quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3]);

		//atan2(sin_half_phi, cos_half_phi) is phi/2, so it is 2 * 180 /PI
		mLocate_result[i].error_rotation = atan2(sin_half_phi, cos_half_phi) * 360.0 / PI;
		global::cout << "after atan2 " << endl;

		cv::Vec3d err_trans = cam_query_true[i].translation - mLocate_result[i].translation;
		mLocate_result[i].error_translation = std::sqrt(err_trans[0] * err_trans[0] 
			+ err_trans[1] * err_trans[1] + err_trans[2] * err_trans[2]);

		global::cout << "end locating image " << i << endl;
	}

	global::cout << "start write result " << endl;
	//save the result into a txt file
	SaveLocalizationResult("result.txt", 2);

	timer.Stop();
	global::cout << "locate all images time: " << timer.GetElapsedTimeAsString() << std::endl;
}

//save the localization result
//0  no, 1: RT, 2 RTK
//format: total num of images  located num of images
//each line is the result information of a image

void VISUALWORDS_3DPOINT_HANDLER::SaveLocalizationResult(const std::string& s, const int iReportRT) const
{
	std::ofstream os(s, std::ios::trunc);
	if (false == os.is_open()){
		global::cout << "open result file fail: " << s << std::endl;
		return;
	}

	auto& cam_true = mParse_bundler.GetAllPicturesAndCameras().GetAllCameras();
	
	os << mNum_totalimage << " " << mNum_locatedimage << endl;
	for (size_t i = 0; i < mLocate_result.size(); i++)
	{
		auto & res = mLocate_result[i];
		os	<< res.located_image << " "
			<< res.have_intrinsics << " "
			<< res.num_putative_match << " "
			<< res.num_inlier_match << " "
			<< res.time_findcorresp << " "
			<< res.time_computepose << " "
			<< res.error_rotation << " "
			<< res.error_translation << " "
			<< std::endl;
		if (iReportRT){
			for (int j = 0; j < 3; j++){
				os	<< res.rotation(j, 0) << " "
					<< res.rotation(j, 1) << " "
					<< res.rotation(j, 2) << " "
					<< res.translation(j) << "    |    ";
				
				os	<< cam_true[i].rotation(j, 0) << " "
					<< cam_true[i].rotation(j, 1) << " "
					<< cam_true[i].rotation(j, 2) << " "
					<< cam_true[i].translation(j) << std::endl;
			}
		}
		if (2 == iReportRT){
			for (int m = 0; m < 3; m++){
				for (int n = 0; n < 3; n++){
					os << res.K(m, n) << " ";
				}
			}
			os << std::endl;
		}
		os << std::endl;
	}

	os.close();
}
