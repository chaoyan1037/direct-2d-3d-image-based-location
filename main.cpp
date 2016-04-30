//main function for image location
#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include <opencv/cv.h>

#include "geometry.h"
#include "visualwords.h"
#include "PreProcess/picture.h"
#include "PreProcess/bundlercamera.h"
#include "PreProcess/parsebundler.h"


using namespace cv;
using namespace std;



int main(int * argc, char** argv)
{
//pre-proscess the original bundler.out
//to get the query pictures pose ground truth
#if 0
	PARSE_BUNDLER parse_bundler("E:/Dubrovnik6K/bundle/bundle.orig.out");
	parse_bundler.ParseBundlerFile();
	parse_bundler.FindQueryPicture("E:/Dubrovnik6K/bundle/list.orig.txt");
	parse_bundler.WriteQueryBundler("E:/Dubrovnik6K/bundle/bundle.query.out");
	return 1;
#endif
	/*	VISUALWORDS_3DPOINT_HANDLER(const std::string &bundle_path,
		const std::string &list_txt,
		const std::string &bundle_file)
	*/
	VISUALWORDS_3DPOINT_HANDLER vw_3d_point_handler("D:/bundlerSIFT/examples/statue",
		"list.txt", "D:/bundlerSIFT/examples/statue/bundle/bundle.db.out");
	vw_3d_point_handler.Init();


	//define query images:
	ALL_PICTURES pic_query("D:/bundlerSIFT/examples/statue/", "list_query.txt");
	pic_query.LoadAllPictures();
	//load the query images pose ground truth
	pic_query.LoadCamerasPose("D:/bundlerSIFT/examples/statue/bundle/bundle.query.out");

	//define the estimated camera pose
	std::vector< BUNDLER_CAMERA >	camera_pose_estimated(pic_query.GetAllPictures().size());
	std::vector< bool >				camera_pose_mask(pic_query.GetAllPictures().size(), false);

	vw_3d_point_handler.LocatePictures(pic_query.GetAllPictures(), camera_pose_estimated, camera_pose_mask);


	return 1;
}