//main function for image location
#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include <opencv/cv.h>

#include "geometry.h"
#include "visualwords.h"
#include "preprocess/picture.h"
#include "preprocess/bundlercamera.h"
#include "preprocess/parsebundler.h"


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
	ALL_PICTURES pic_query("D:/bundlerSIFT/examples/statue", "list_query.txt");
	pic_query.SetQueryFlag(true);
	pic_query.LoadPicturesKeyFile();

	//load the query images pose ground truth
	pic_query.LoadCamerasPose("D:/bundlerSIFT/examples/statue/bundle/bundle.query.out");

	vw_3d_point_handler.LocatePictures(pic_query);

	return 1;
}