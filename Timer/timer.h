#ifndef _TIMER_H_ 
#define _TIMER_H_

/**
*    Class to get timing results (basic stopwatch).
*    Reports time in seconds, accurate up to microseconds
*	 now it works only on windows...
*
*    Based on the timer implementation of Darko Pavic.
*/

#ifdef WIN32
#include<windows.h>
#endif

#include <string>

class  Timer
{
public:
	 Timer();
	~ Timer();

	//initialize timer(set the elapsed time to 0)
	void Init();

	//start the timer(also set the elapsed time to 0)
	void Start();

	//Restart the tiemr(do not reset the elapsed time)
	void ReStart();

	//stop the timer 
	void Stop();

	//get the elapsed time in seconds or milliseconds
	double GetElapsedTimeSecond();
	double GetElapsedTimeMilliSecond();

	//get the elapsed time in seconds as a string
	std::string GetElapsedTimeAsString();

private:

	double	mTime_start;
	double	mTime_end;

	double  mElapsed_time;//in milli seconds;
};



#endif