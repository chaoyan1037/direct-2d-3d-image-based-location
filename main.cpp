//main function for image location
#include <fstream>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

#include <opencv/cv.h>

#include "global.h"
#include "geometry.h"
#include "visualwords.h"
#include "preprocess/picture.h"
#include "preprocess/bundlercamera.h"
#include "preprocess/parsebundler.h"


int main(int * argc, char** argv)
{
	if (false == global::OpenRunningTimeFile()){
		return false;
	}

//pre-proscess the original bundler.out
//to get the query pictures pose ground truth
#if 0
	PARSE_BUNDLER parse_bundler("E:/Dubrovnik6K/bundle/bundle.orig.out");
	parse_bundler.ParseBundlerFile();

  parse_bundler.FindQueryPicture("E:/Dubrovnik6K/bundle/list.orig.txt");
  parse_bundler.WriteQueryBundler( "E:/Dubrovnik6K/bundle/bundle.query.point.out", true );
	return 1;
#endif

	//test geometry
#if 0
	Test();
	//Geometry geo;
	//geo.TestGeometry();
	return 1;
#endif

 // modify the original bundler, add feature point x, y 
#if 0
  VISUALWORDS_3DPOINT_HANDLER vw_3d_pointhandler( "E:/Dubrovnik6K",
    "list.query.txt", "E:/Dubrovnik6K/bundle/bundle.query.cam.out" );
  vw_3d_pointhandler.Init();
  // when write bundle, should not load desc, and not release pictures
  vw_3d_pointhandler.mParse_bundler.WriteBundlerWithXY( "E:/Dubrovnik6K/bundle/bundle.query.xy.out" );

  return 1;
#endif

#if 0
	/*	VISUALWORDS_3DPOINT_HANDLER(const std::string &bundle_path,
		const std::string &list_txt,
		const std::string &bundle_file)
	*/
	VISUALWORDS_3DPOINT_HANDLER vw_3d_point_handler("E:/statue",
		"list.db.txt", "E:/statue/bundle/bundle.db.out");
	vw_3d_point_handler.Init();
	
	//define query images:
 	ALL_PICTURES pic_query("E:/statue", "list.query.txt");
	pic_query.SetQueryFlag(true);
	pic_query.LoadPicturesKeyFile();

	//load the query images pose ground truth
	pic_query.LoadCamerasPose("E:/statue/bundle/bundle.query.out");

	vw_3d_point_handler.LocatePictures(pic_query);

#else

	VISUALWORDS_3DPOINT_HANDLER vw_3d_point_handler("E:/Dubrovnik6K",
		"list.db.txt", "E:/Dubrovnik6K/bundle/bundle.db.out");
	vw_3d_point_handler.Init();

	//define query images:
	ALL_PICTURES pic_query("E:/Dubrovnik6K", "list.query.txt");
	pic_query.SetQueryFlag(true);
	pic_query.LoadPicturesKeyFile(true);

	//load the query images pose ground truth
	pic_query.LoadCamerasPose("E:/Dubrovnik6K/bundle/bundle.query.out");
  //test the performance of CasHash
	vw_3d_point_handler.LocatePictures(pic_query);

#endif


	if (false == global::CloseRunningTimeFile()){
		return false;
	}
	return 1;
}