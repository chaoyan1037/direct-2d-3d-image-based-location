# direct-2d-3d-image-based-location
use the direct 2d-3d match method to do image-base location.

original paper: 
Sattler, Torsten, Bastian Leibe, and Leif Kobbelt. 
"Fast image-based localization using direct 2d-to-3d matching." 
Computer Vision (ICCV), 2011 IEEE International Conference on. IEEE, 2011.


I use 5-point epnp algorithm(Epnp: An accurate o(n) solution to the pnp problem.IJCV2009)
to estimate camera pose when intrinsics are availabel, and will get a quite accurate result.
Then use 6-point DLT methods to estimate pose when there are no intrinsics, and the result
is much less accurate. So nonlinear optimization is utilized after DLT estimating pose, Also,
I implemented two methods to do this task. First, sba 1.6(http://users.ics.forth.gr/~lourakis/sba/)
is used to do bundle adjustment. Second, use nonlinear optimization as PTAM do. This nonlinear
optimization is optional, because I find most time the result improve just a little.

Notes:

The program was developed in the  VS2013 on a 64-bit Windows 10 machine.
visual studio project files are provided, you can directly open the .vcxproj file.
Also, You may new the visual studio project yourself.

The program use the flann kd-tree in OpenCV(I use OpenCV3.1, 2.4.8 should be fine, 
but I did not test that)
Please first install OpenCV library on your PC.

Besides, the program use TooN to do nonlinear optimization as the PTAM. 
You can download TooN here http://www.edwardrosten.com/cvd/toon/html-user/
Just extract the TooN under the source file and rename it as "TooN". 

To accelerate the processing, simple OpenMP instructors is used.
To enable OpenMP sopport, choose visual studio project properties -> Configuration Properties
 -> C/C++ -> Language, choose the Open MP Support Yes!
 
I use the 5-point epnp algorithm to calculate camera pose when intrinsics are availabnle.
The original epnp implementation do not have copy constructor or copy assignment function,
this will cause error when use OpenMP, so I implement copy constructor and copy assignment.
So I upload my modified epnp algorithm source files. 


TODO:
In the feature I will try to use binary descriptors instead of SIFT.
This is chanllanging since binary descriptors are less robust.



