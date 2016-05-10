#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <fstream>
#include <iostream>

/************************************************************************/
/* define the global settings                                           */
/************************************************************************/

#define USE_STD_COUT

namespace global{

#ifdef USE_STD_COUT

	extern std::ostream& cout;

#else

	extern std::ofstream cout;

#endif //  USE_STD_COUT

	bool OpenRunningTimeFile();
	bool CloseRunningTimeFile();
}

#endif