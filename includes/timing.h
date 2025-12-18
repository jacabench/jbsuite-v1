/**
*   JACABench: JACA Benchmark Suite
*  	December 2025
*/
#ifndef TIMING_H
#define TIMING_H

/* Timing options
*	TIMING:
*	0: none timing measurements
*	1: for considering timing measurements
*/
#ifndef TIMING
#define TIMING 0
#endif

#if TIMING == 1 // for timing the execution
	#include "timer.h"
	
	#define __INIT_TIMING() Timer *timer = timer_init();
	
	#define __START_TIMING() timer_start(timer);

	#define __END_TIMING() timer_stop(timer);

	#define __REPORT_TIMING_US() \
		const double time = timer_get_us(timer); \
		timer = timer_destroy(timer); \
		printf("\nExecution time:  %.4f us\n", time); \
		
	#define __REPORT_TIMING_MS() \
		const double time = timer_get_ms(timer); \
		timer = timer_destroy(timer); \
		printf("\nExecution time:  %.4f ms\n", time); \
		
	#define __REPORT_TIMING_S() \
		const double time = timer_get_s(timer); \
		timer = timer_destroy(timer); \
		printf("\nExecution time:  %.4f s\n", time); \
	
#else // nothing to be done
	#define __INIT_TIMING() 
	#define __START_TIMING() 
	#define __END_TIMING() 
	#define __REPORT_TIMING_S()
	#define __REPORT_TIMING_MS()
	#define __REPORT_TIMING_US()
#endif

#endif // TIMING_H
