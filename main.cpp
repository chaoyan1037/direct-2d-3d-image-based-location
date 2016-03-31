//mian function for image location
#include<fstream>
#include<iostream>

#include<opencv2/opencv.hpp>

#include "picture.h"
#include "visualwords.h"
#include "bundlercamera.h"
#include "parsebundler.h"

using namespace cv;
using namespace std;

int main(int * argc, char** argv)
{
	cv::Mat mat(10, 1, CV_8UC1);
	mat.row(5) = 1;
	cout << mat;
	return 1;

	ALL_PICTURES pic_db("E:/Dubrovnik6K", "list.db.txt");
	pic_db.LoadAllPictures();

	PARSE_BUNDLER parse_bundler;
	parse_bundler.ParseBundlerFile("E:/Dubrovnik6K/bundle/bundle.db.out");
	parse_bundler.LoadCameraInfo(pic_db);

	PICTURE pic;
	pic.LoadKeyPointAndDes("E:/Dubrovnik6K/query/_sml_2738520609.key");
	
	//load visual words and build the index
	VISUALWORDS_HANDLER vw_handler;
	vw_handler.LoadDBVisualWords();
	vw_handler.BuildIndex();
	
	//vector<unsigned char*>  query_des;
	//for (int i = 0; i < 1; i++)
	//{
	//	query_des.push_back(pic.GetDescriptor()[i]);
	//}

	//Mat indices, dists;
	//vw_handler.KnnSearch(query_des, indices, dists);
	
	return 1;
}