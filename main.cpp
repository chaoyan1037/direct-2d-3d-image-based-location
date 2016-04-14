//main function for image location
#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include <vector>
#include <opencv/cv.h>

#include "geometry.h"
#include "picture.h"
#include "visualwords.h"
#include "bundlercamera.h"
#include "parsebundler.h"


using namespace cv;
using namespace std;



int main(int * argc, char** argv)
{
	Matx34d P;
	P = P.randn(1.0, 1.0);
	std::pair<cv::Vec2d, cv::Vec3d> match;
	match.first = Vec2d(200.0, 400.0);
	match.second = Vec3d(1.0, 2.0, 3.0);

	Geometry geo;
	cout<<geo.ComputeReprojectionError(P, match);
	return 1;
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

	vw_3d_point_handler.LocateSinglePicture(pic_query.GetAllPictures()[0]);

	//Mat indices, dists;
	//vw_handler.KnnSearch(query_des, indices, dists);
	
	return 1;
}