#include "global.h"
#include <iostream>



namespace global{

#ifdef  USE_STD_COUT
	std::ostream& cout=std::cout;
	bool OpenRunningTimeFile(){ return true; }
	bool CloseRunningTimeFile(){ return true; }

#else
	std::ofstream cout;
	bool OpenRunningTimeFile(){
		cout.open("runtimefile.txt", std::ios::out | std::ios::trunc);
		if (!cout.is_open()){
			std::cerr << "open runtime file fail." << std::endl;
			return false;
		}
		return true;
	}

	bool CloseRunningTimeFile(){
		if (!cout.is_open()){
			std::cerr << "runtime file is not open." << std::endl;
			return false;
		}
		else cout.close();
		return true;
	}
#endif
}


