#include <fstream>
#include <algorithm>
#include <vector>
#include <set>
#include <map>
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

//self define compare function for use in the FindCorrespondence()
static bool compareFunc(const std::pair<size_t, size_t> & p1, const std::pair<size_t, size_t> & p2){
	return p1.second < p2.second;
};

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

	mMaxNumberCorrespondence = 120;
	mMinNumberCorrespondence = 12;

	//integer mean per visual words
	mPoint3D_method = 0;
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
	
	mPoint3D.resize( mParse_bundler.GetNumPoints() );
	size_t num_all_desc = 0;
	//calculate the total num of descriptors
	for (size_t i = 0; i < feature_info.size(); i++){
		num_all_desc += feature_info[i].mView_list.size();
		mPoint3D[i] = feature_info[i].mPoint;
	}

	mAll_descriptor.resize(num_all_desc);
	
	size_t desc_index = 0;
	for (int i = 0; i < feature_info.size(); i++)
	{
		cv::Mat indices, dists;
		mVW_handler.KnnSearch(feature_info[i].mDescriptor, indices, dists, 1);
		//for each visual words, add the current 3d point index
		for (int j = 0; j < indices.rows; j++)
		{
			int vw_index_id = indices.ptr<int>(j)[0];
			mVisualwords_index_3d[vw_index_id].insert(std::make_pair(i, desc_index));
			mAll_descriptor[desc_index++] = std::move(feature_info[i].mDescriptor[j]);
		}
	}
	global::cout << "Build index 3d points, total num of desc: " << num_all_desc << endl;
	global::cout << "After build index 3d points, release mParse_bundler" << endl;
	mParse_bundler.ClearData();

	// after build the index, process visual words and 3d points according to the mPoint3D_method
	std::vector< std::set< std::pair<size_t, size_t> > > temp_vw_index_3d(num_visualwords);
	std::vector< SIFT_Descriptor > temp_all_desc;
	size_t id_total_desc = 0;

	for (size_t i = 0; i < mVisualwords_index_3d.size(); i++){
		// get the number of point of that visual words, which coincides with the number of 
		// descriptors available for that visual words.
		size_t num_point_i = mVisualwords_index_3d[i].size();
		
		// compute for each visual word the mean descriptors, round it to the next integer and store it
		if (0 == mPoint3D_method) //integer mean per visual words
		{
			// first determine the number of activated 3d points
			std::set< size_t > activated_3d_points;
			activated_3d_points.clear();
			
			// traverse the set
			for (auto pair_temp : mVisualwords_index_3d[i]){
				activated_3d_points.insert(pair_temp.first);
			}

			size_t num_vw_3d_desc = 0;
			std::vector<float> visual_word_desc(SIFT_Descriptor::legth, 0.0);
			// now we compute the mean descriptors belonging to the visual words
			for (auto id_3d_point : activated_3d_points){
				num_vw_3d_desc = 0;
				visual_word_desc.assign(SIFT_Descriptor::legth, 0.0);
				for (auto pair_temp : mVisualwords_index_3d[i]){
					if (id_3d_point == pair_temp.first){
						auto ptr_desc = mAll_descriptor[pair_temp.second].ptrDesc;
						for (size_t k = 0; k < SIFT_Descriptor::legth; k++){
							visual_word_desc[k] += (float)ptr_desc[k];
						}
						num_vw_3d_desc++;
					}
				}

				std::vector<float> visual_word_desc_mean(SIFT_Descriptor::legth, 0.0);
				SIFT_Descriptor temp_sift;
				temp_sift.ptrDesc = new unsigned char[SIFT_Descriptor::legth];
				if (!temp_sift.ptrDesc){
					global::cout << "new error. visualwords.cpp line203" << endl;
					return 0;
				}
				for (size_t k = 0; k < SIFT_Descriptor::legth; k++){
					visual_word_desc_mean[k] = visual_word_desc[k] / num_vw_3d_desc;
					// round to the nearest integer values
					float bottom = visual_word_desc_mean[k] - std::floor(visual_word_desc_mean[k]);
					float top = std::ceil(visual_word_desc_mean[k]) - visual_word_desc_mean[k];
					if (bottom < top){
						temp_sift.ptrDesc[k] = (unsigned char)std::floor(visual_word_desc_mean[k]);
					}
					else  temp_sift.ptrDesc[k] = (unsigned char)std::ceil(visual_word_desc_mean[k]);
				}

				temp_vw_index_3d[i].insert(std::make_pair(id_3d_point, id_total_desc));
				// store the descriptor
				temp_all_desc.push_back(std::move(temp_sift));
				id_total_desc++;
			}
		}
	}
	//std::vector< std::set< std::pair<size_t, size_t> > >(temp_vw_index_3d).swap(mVisualwords_index_3d);
	//std::vector< SIFT_Descriptor >(temp_all_desc).swap(mAll_descriptor);
	mVisualwords_index_3d.swap(temp_vw_index_3d);
	mAll_descriptor.swap(temp_all_desc);
	mAll_descriptor.shrink_to_fit();
	global::cout << "Finish process visual words and 3d points , total num of desc: " << id_total_desc << endl;
	
	return 1;
}

//after build the index, then save it into file.
//format:
//#visual_words #3Dpoints  #total descriptor
//#3d points assigned to this vw, pair(i1, j1)... pair(in, jn) for each visual word
//...
//3d points(x, y, z)
//...
//descriptors
//...
bool VISUALWORDS_3DPOINT_HANDLER::SaveIndex3DPoints(const std::string&s) const
{
	std::ofstream of(s, std::ios::out | std::ios::trunc);
	if (0 == of.is_open()){
		global::cout << " open index_3d_points file fail: " << s << std::endl;
		return 0;
	}

	of	<< mVisualwords_index_3d.size() << " "
		<< mPoint3D.size() << " "
		<< mAll_descriptor.size() << std::endl;
	
	//save the index of visual words to 3d points
	for (size_t i = 0; i < mVisualwords_index_3d.size(); i++){
		//first save the num of 3d points in this visual words
		of << mVisualwords_index_3d[i].size() << " ";
		for (auto& set_member : mVisualwords_index_3d[i]){
			of << set_member.first << " " << set_member.second << " ";
		}
		of << std::endl;
	}

	//save the 3d points
	for (auto& point3d : mPoint3D){
		of << point3d.x << " " << point3d.y << " " << point3d.z <<std::endl;
	}
	
	//save all the descriptors
	for (size_t j = 0; j < mAll_descriptor.size(); j++){
		for (size_t k = 0; k < mAll_descriptor[j].legth; k++){
			of << int(mAll_descriptor[j].ptrDesc[k]) << " ";
		}
		of << std::endl;
	}

	of.close();
	return 1;
}

bool VISUALWORDS_3DPOINT_HANDLER::LoadIndex3DPoints(const std::string& s)
{
	std::ifstream is("index_3dpoints.txt", std::ios::_Nocreate);
	if (0 == is.is_open()){
		global::cout << " no index_3d_points file, then reload: "<< std::endl;
		return 0;
	}

	size_t num_VW = 0, num_3Dpoint = 0, num_desc = 0;
	is >> num_VW >> num_3Dpoint >> num_desc;
	if (num_VW != mVW_handler.GetNumVisualWords()){
		global::cout << "loaded VW num is not equal to the #visual words" << endl;
		return 0;
	}

	mVisualwords_index_3d.resize(num_VW);
	mPoint3D.resize(num_3Dpoint);
	mAll_descriptor.resize(num_desc);

	//std::vector<std::set<std::pair<int, int>>> mVisualwords_index_3d;
	size_t num_pts_VW = 0;
	for (size_t i = 0; i < num_VW; i++){
		is >> num_pts_VW;
		std::pair<size_t, size_t> pair_temp;
		while (num_pts_VW--){
			is >> pair_temp.first >> pair_temp.second;
			mVisualwords_index_3d[i].insert(pair_temp);
		}
	}

	//load the 3d points
	for (size_t i = 0; i < num_3Dpoint; i++){
		is >> mPoint3D[i].x >> mPoint3D[i].y >> mPoint3D[i].z;
	}

	//load descriptors
	int temp = 0;
	for (size_t i = 0; i < num_desc; i++){
		mAll_descriptor[i].ptrDesc = new unsigned char[SIFT_Descriptor::legth];
		if (!mAll_descriptor[i].ptrDesc){
			global::cout << "new failed. visualwords.cpp line 324. try again." << endl;
			i--;
			continue;
		}
		for (size_t k = 0; k < mAll_descriptor[i].legth; k++){
			is >> temp;
			mAll_descriptor[i].ptrDesc[k]=unsigned char(temp);
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
	mVW_handler.LoadDBVisualWords();
	mVW_handler.BuildIndex();
	timer.Stop();
	global::cout << "Load and build visual words time: " << timer.GetElapsedTimeAsString() << std::endl;

	//if the index file exists, directly load it.
	if (LoadIndex3DPoints("index_3dpoints.txt"))
	{
		global::cout << "load index_3dpoints.txt" << endl;
	}
	else{
		//if parsed_bundler.txt exists, directly load it.
		timer.Start();
		if (mParse_bundler.LoadFeature3DInfro("parsed_bundler.txt")){
			global::cout << "load parsed_bundler file" << endl;
		}
		else{
			mParse_bundler.ParseBundlerFile();
			//load the database pictures
			auto& mPicCam_db = mParse_bundler.GetAllPicturesAndCameras();
			mPicCam_db.LoadPicturesKeyFile();
			mParse_bundler.LoadCameraInfo();
			//after load bundler and execute LoadCameraInfo, release the pictures
			mPicCam_db.ClearPics();
			mParse_bundler.SaveFeature3DInfro("parsed_bundler.txt");
			global::cout << "load *.key files and save parsed_bundler.txt" << endl;
		}
		timer.Stop();
		global::cout << "Parse Bundler file time: " << timer.GetElapsedTimeAsString() << std::endl;

		//then build the index 3d points
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
bool VISUALWORDS_3DPOINT_HANDLER::FindCorrespondence(const size_t loca_res_index, const PICTURE& picture)
{
	auto& pic_feat_desc			= picture.GetDescriptor();
	auto& correspdence_2d_3d	= mLocate_result[loca_res_index].mFeature_3d_point_correspondence;
	
	cv::Mat indices, dists;
	mVW_handler.KnnSearch(pic_feat_desc, indices, dists, 1);

	if (pic_feat_desc.size() != indices.rows){
		global::cout << "error:visualwords.cpp line 399" << std::endl;
		return false;
	}


	//calculate the feature's assigned index of visual words.
	std::vector< size_t > assigned_2d_vw( pic_feat_desc.size() );
	//corres_2d_vw.reserve(pic_feat_desc.size());
	for ( size_t i = 0; i < pic_feat_desc.size(); i++ ){
		assigned_2d_vw[i] = indices.ptr<int>(i)[0];
	}


	//sort the 2d_vw correspondence, according to the num of 3d points the vw contained 
	std::vector< std::pair< size_t, size_t > > priorities(pic_feat_desc.size());
	for (size_t j = 0; j < priorities.size(); j++){
		priorities[j].first = j;
		priorities[j].second = mVisualwords_index_3d[assigned_2d_vw[j]].size();
	}
	std::sort(priorities.begin(), priorities.end(), compareFunc);


	global::cout << "start find feature's correspondence" << std::endl;
	
	// we store for each 3D point the corresponding 2D feature as well as 
	// the squared distance this is needed in case that two 2D features 
	// are assigned to one 3D point, because we only want to keep the 
	// correspondence to the 2D point with the most similar descriptor
	// i.e. the smallest Euclidean distance in descriptor space
	std::map< size_t, std::pair< size_t, int > > corr_3D_to_2D;
	corr_3D_to_2D.clear();

	//for matched visual words, find feature's matched 3d points
	for (size_t i = 0; i < priorities.size(); i++)
	{
		if (corr_3D_to_2D.size() >= mMaxNumberCorrespondence) break;
		
		size_t feat_index	= priorities[i].first;
		size_t vw_index		= assigned_2d_vw[feat_index];

		//the squared distance of current feature to 3d point's feature
		int min_distance_squared[2] = { 100000000, 100000000 };
		int min_distance_3d_point_index[2] = {-1, -1};
		
		//for each visual words find all 3d point pair<int, int>
		for (auto pair_3d_point : mVisualwords_index_3d[vw_index])
		{
			int point_3d_index = pair_3d_point.first;
			int point_desc_index = pair_3d_point.second;
		
			int distsq_temp = CalculateSIFTDistanceSquared(pic_feat_desc[feat_index].ptrDesc, mAll_descriptor[point_desc_index].ptrDesc);
			
			//after get this 3d point smallest distance
			//compare this 3d point to current smallest distances
			//and always keep the smallest distances 3d point
			if (distsq_temp < min_distance_squared[1])
			{
				min_distance_squared[1] = distsq_temp;
				min_distance_3d_point_index[1] = point_3d_index;

				if (min_distance_squared[1] < min_distance_squared[0]){
					std::swap(min_distance_squared[0], min_distance_squared[1]);
					std::swap(min_distance_3d_point_index[0], min_distance_3d_point_index[1]);
				}
			}
		}

		//after find two putative matched 3d points do ratio test     
		//mFeature_3d_point_correspondence_ratio_test_thres
		//Check whether closest distance is less than 0.7 of second.
		if (min_distance_3d_point_index[0] >= 0 
			&& 10 * 10 * min_distance_squared[0] < 7 * 7 * min_distance_squared[1]
			&& min_distance_squared[0] <= 12800)
		{
			//check if the found 3D is in the map corr_3D_to_2D
			auto map_it_3D = corr_3D_to_2D.find(min_distance_3d_point_index[0]);
			
			//this 3D is already in the map
			if (map_it_3D != corr_3D_to_2D.end()){
				if (map_it_3D->second.second > min_distance_squared[0])
				{
					map_it_3D->second.first = feat_index;
					map_it_3D->second.second = min_distance_squared[0];
				}
			}
			else{
				corr_3D_to_2D.insert(std::make_pair(min_distance_3d_point_index[0], 
					std::make_pair(feat_index, min_distance_squared[0])));
			}
		}
	}

	//no enough correspondence, then fail
	if (corr_3D_to_2D.size() < mMinNumberCorrespondence){
		correspdence_2d_3d.clear();
		global::cout << "not enough 2d-3d correspondence found." << endl;
		return 0;
	}
	
	// store the ids of the 2D features and the 3D points3D
	// first the 2D, then the 3D point
	for (auto map_it_3D = corr_3D_to_2D.cbegin(); map_it_3D != corr_3D_to_2D.cend(); ++map_it_3D)
	{
		correspdence_2d_3d.push_back(std::make_pair(map_it_3D->second.first, map_it_3D->first));
	}
	global::cout << "2d-3d correspondence found: " << correspdence_2d_3d .size() << endl;
	
	return 1;
}

//the public function to locate a single picture
bool VISUALWORDS_3DPOINT_HANDLER::LocateSinglePicture(const size_t loca_res_index, const PICTURE& picture)
{
	Timer timer;
	timer.Start();
	auto& result = mLocate_result[loca_res_index];
	result.located_image = false;
	//0: can not find enough 2d-3d correspondence
	global::cout << "start find correspondence" << std::endl;
	if (0 == FindCorrespondence(loca_res_index, picture))
	{
		timer.Stop();
		result.time_findcorresp = timer.GetElapsedTimeMilliSecond();
		return 0;
	}
	timer.Stop();
	result.time_findcorresp = timer.GetElapsedTimeMilliSecond();


	static int test_cnt = 20;
	if (test_cnt < 20){
		global::cout << result.have_intrinsics << " "
			<< result.K(0, 0) << " " << result.K(1, 1) << " "
			<< result.K(0, 2) << " " << result.K(1, 2) << " "
			<< result.mFeature_3d_point_correspondence.size();
	}


	timer.ReStart();
	global::cout << "geo " << std::endl;
	Geometry geo;
	//geo.match_2d_3d
	auto& mat_2d_3d = geo.ReturnMatch_2d_3d();
	mat_2d_3d.reserve(mMaxNumberCorrespondence);

	int height = 0, width = 0;
	picture.GetImageSize(height, width);
	assert(height && width);

	auto& pic_point_2d = picture.GetFeaturePoint();

	for (auto pa : result.mFeature_3d_point_correspondence)
	{
		auto& pt_2d = pic_point_2d[pa.first];
		auto& pt_3d = mPoint3D[pa.second];
		mat_2d_3d.push_back(std::make_pair(
			cv::Vec2d(width - pt_2d.x, height - pt_2d.y),
			cv::Vec3d(-pt_3d.x, pt_3d.y, -pt_3d.z)));
	
		//for debug convenience, save ten query images
		if (test_cnt < 20){
			global::cout << pt_2d.x << " " << pt_2d.y << " "
				<< pt_3d.x << " " << pt_3d.y << " " << pt_3d.z << " ";
		}
	}

	//if (test_cnt < 20) test_cnt++; global::cout << endl;

	result.num_putative_match = mat_2d_3d.size();

	//if have intrinsics then use epnp
	if (true == result.have_intrinsics){
		global::cout << "epnp compute pose" << std::endl;
		geo.SetK(result.K);
		//geo.SetIntrinsicParameter(result.K(0, 0), 0, 0);
		result.num_inlier_match = geo.ComputePoseEPnP();

		timer.Stop();
		global::cout << "epnp locate single image time: " << timer.GetElapsedTimeMilliSecond() << endl;
		result.time_computepose = timer.GetElapsedTimeMilliSecond() - result.time_findcorresp;

		if (0 == result.num_inlier_match){ return 0; }

		geo.GetRT(result.rotation, result.translation);
		
	}
	else{
		global::cout << "dlt compute pose" << std::endl;
		result.num_inlier_match = geo.ComputePoseDLT();

		timer.Stop();
		global::cout << "dlt locate single image time: " << timer.GetElapsedTimeMilliSecond() << endl;
		result.time_computepose = timer.GetElapsedTimeMilliSecond() - result.time_findcorresp;

		if (0 == result.num_inlier_match){ return 0; }

		geo.GetRT(result.rotation, result.translation);

		//only for DLT, get the estimated K
		geo.GetK_est(result.K);
	}

	//transform back
	result.rotation(0, 1) = -result.rotation(0, 1);
	result.rotation(1, 0) = -result.rotation(1, 0);
	result.rotation(1, 2) = -result.rotation(1, 2);
	result.rotation(2, 1) = -result.rotation(2, 1);
	result.translation[0] = -result.translation[0];
	result.translation[2] = -result.translation[2];
	result.located_image = true;	

	return 1;
}

void VISUALWORDS_3DPOINT_HANDLER::LocatePictures(const ALL_PICTURES& pic_cam_query)
{
	Timer timer, timer_sigle;
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

	//TODO: use openMP 
	for (int i = 0; i < pic_query.size(); i++)
	{
		int h = 0, w = 0;
		global::cout << endl <<"start locating image " << i << endl;
		if (pic_query_focal[i] > 0){
			mLocate_result[i].have_intrinsics = true;
			mLocate_result[i].K(0, 0) = pic_query_focal[i];
			mLocate_result[i].K(1, 1) = pic_query_focal[i];
			
			pic_query[i].GetImageSize(h, w);
			mLocate_result[i].K(0, 2) = (w - 1) >> 1;
			mLocate_result[i].K(1, 2) = (h - 1) >> 1;
			mLocate_result[i].K(2, 2) = 1.0;
		}

		timer_sigle.Start();
		LocateSinglePicture(i, pic_query[i]);
		timer_sigle.Stop();
		global::cout << "locate single image time: " <<timer_sigle.GetElapsedTimeAsString()<< endl;

		if (0 == mLocate_result[i].located_image){
			global::cout << "fail to locate image " << i << endl;
			continue;
		} 

		global::cout << "successfully locate image " << i << endl;

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
	SaveLocalizationResult("result.txt", pic_cam_query, 2);

	timer.Stop();
	global::cout << "locate all images time: " << timer.GetElapsedTimeAsString() << std::endl;
}

//save the localization result
//0  no, 1: RT, 2 RTK
//format: total num of images  located num of images
//each line is the result information of a image

void VISUALWORDS_3DPOINT_HANDLER::SaveLocalizationResult(const std::string& s, 
	const ALL_PICTURES& pic_cam_query, const int iReportRT) const
{
	std::ofstream os(s, std::ios::trunc);
	if (false == os.is_open()){
		global::cout << "open result file fail: " << s << std::endl;
		return;
	}

	auto& cam_true = pic_cam_query.GetAllCameras();
	
	os << mNum_totalimage << " " << mNum_locatedimage << endl;
	for (size_t i = 0; i < mLocate_result.size(); i++)
	{
		auto & res = mLocate_result[i];
		os	<< "Y: " << res.located_image << " "
			<< "f: " << res.have_intrinsics << " "
			<< "match: "<<res.num_putative_match << " "
			<< "inlier_mat: "<<res.num_inlier_match << " "
			<< "t_corres: "<<res.time_findcorresp << " "
			<< "t_compute: "<<res.time_computepose << " "
			<< "err_r: "<<res.error_rotation << " "
			<< "err_c: "<<res.error_translation << " "
			<< std::endl;

		if (iReportRT){
			for (int j = 0; j < 3; j++){
				os	<< res.rotation(j, 0) << " "
					<< res.rotation(j, 1) << " "
					<< res.rotation(j, 2) << " "
					<< res.translation(j) << "   | truth   ";
				
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
