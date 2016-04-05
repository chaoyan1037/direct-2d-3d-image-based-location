//mian function for image location
#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>

#include "picture.h"
#include "visualwords.h"
#include "bundlercamera.h"
#include "parsebundler.h"

using namespace cv;
using namespace std;



int main(int * argc, char** argv)
{
	/*	VISUALWORDS_3DPOINT_HANDLER(const std::string &bundle_path,
		const std::string &list_txt,
		const std::string &bundle_file)
	*/
	VISUALWORDS_3DPOINT_HANDLER vw_3d_point_handler("D:/bundlerSIFT/examples/statue/",
		"list.txt", "D:/bundlerSIFT/examples/statue/bundle/bundle.out");
	vw_3d_point_handler.Init();


	//define query images:
	ALL_PICTURES pic_query("D:/bundlerSIFT/examples/statue/", "list_query.txt");
	pic_query.LoadAllPictures();



	//Mat indices, dists;
	//vw_handler.KnnSearch(query_des, indices, dists);
	
	return 1;
}