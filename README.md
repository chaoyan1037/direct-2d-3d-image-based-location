# direct-2d-3d-image-base-location
use the direct 2d-3d match method to do image-base location

Notes:
The program was developed in the  VS2013 on a 64-bit Windows 10 machine.
visual studio project files are provided, you can directly open the .vcxproj file.
Also, You may new the visual studio project yourself.

The program use the flann kd-tree in OpenCV(I use OpenCV3.1, 2.4.8 should be fine, but I did not test that)
Please first install OpenCV library on your PC.

I use the 5-point epnp algorithm to calculate camera pose when intrinsics are availabnle.
The original epnp implementation do not have copy constructor or copy assignment function,
this will cause error when use OpenMP, so I implement copy constructor and copy assignment.

To accelerate the processing, simple OpenMP instructors is used.
To enable OpenMP sopport, choose visual studio project properties -> Configuration Properties
 -> C/C++ -> Language, choose the Open MP Support Yes!
 
 
 
 
original paper: 
Sattler, Torsten, Bastian Leibe, and Leif Kobbelt. 
"Fast image-based localization using direct 2d-to-3d matching." 
Computer Vision (ICCV), 2011 IEEE International Conference on. IEEE, 2011.
