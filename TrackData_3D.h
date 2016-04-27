#ifndef TRACKDATA_3D_H
#define TRACKDATA_3D_H

#include <TooN/TooN.h>
#include <TooN/se3.h>
#include <vector>


using namespace std;
using namespace TooN;


struct trackdata_3D{
	Vector<3> mappoint;
	Vector<3> p_cam;
	Vector<2> measurement;
	Vector<2> reprojection;
	Matrix<2,6> mJacobian;//Jacobian wrt camera position
	//MapPoint* mappointpointer;
	double errorsquared;
	int idx;
	trackdata_3D(Vector<3> p_world,Vector<2> p_image,int i)
		:mappoint(p_world),measurement(p_image),idx(i)
	{


	}
	trackdata_3D(){


	}
	
	trackdata_3D &operator =(const trackdata_3D & trackdata)
	{
		mappoint=trackdata.mappoint;
		p_cam=trackdata.p_cam;
		measurement=trackdata.measurement;
		reprojection=trackdata.reprojection;
		mJacobian=trackdata.mJacobian;
		//mappointpointer=trackdata.mappointpointer;
		errorsquared=trackdata.errorsquared;
		idx=trackdata.idx;
		return *this;
	}

	// Project point into image given certain pose and camera.
	inline void projection(SE3<> camerapose,Matrix<3> K){
	    p_cam=camerapose*mappoint;
	    Vector<3> vImage=K*p_cam;
	    if(vImage[2]==0)
	        vImage[2]=0.0000001;
	    vImage=vImage/vImage[2];
	    reprojection=makeVector(vImage[0],vImage[1]);
	    errorsquared=(measurement-reprojection)*(measurement-reprojection);
	}

	// Jacobian of projection W.R.T. the camera position
	// I.e. if  p_cam = SE3Old * p_world,
	//         SE3New = SE3Motion * SE3Old
	inline void CalcJacobian(Matrix<3> K)
	{
	    double doneovercameraz = 1.0 / p_cam[2];
	    for(int m=0; m<6; m++){
	        const Vector<4> v4Motion = SE3<>::generator_field(m, makeVector(p_cam[0],p_cam[1],p_cam[2],1));
	        Vector<2> v2CamFrameMotion;
	        v2CamFrameMotion[0] =K[0][0]*(v4Motion[0] - p_cam[0] * v4Motion[2] * doneovercameraz) * doneovercameraz;
	        v2CamFrameMotion[1] =K[1][1]*(v4Motion[1] - p_cam[1] * v4Motion[2] * doneovercameraz) * doneovercameraz;
	        mJacobian.T()[m] = v2CamFrameMotion;
	    };
	}
	// Sometimes in tracker instead of reprojecting, just update the error linearly!
	inline void linearupdate(const Vector<6> & v)
	{
	    reprojection+=mJacobian*v;
	    errorsquared=(measurement-reprojection)*(measurement-reprojection);
	}

};

struct trackdatalist_3D{
	trackdatalist_3D(){
		clear();
		list.clear();
	}
	~trackdatalist_3D(){
		clear();
		list.clear();
	}
    vector<trackdata_3D *> list;
    void clear()
    {
        for(int i=0;i<list.size();i++)
            if(list[i])
                delete list[i];
        list.clear();
    }
    trackdatalist_3D & operator=(const trackdatalist_3D & trackdatalist)
	{
        clear();
        for(int i=0;i<trackdatalist.list.size();i++){
            trackdata_3D * temp=new trackdata_3D();
            *temp=*(trackdatalist.list[i]);
            list.push_back(temp);
        }
    }
};



#endif // TRACKDATA_3D_H
