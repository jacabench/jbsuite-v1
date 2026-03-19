#ifndef KNN_PARAMS_H //KNN_PARAMS_H
#define KNN_PARAMS_H

#include <float.h>
#include <stdint.h>

#ifndef SCENARIO_FEATURES
#define SCENARIO_FEATURES 2  // 1 = WISDM_Act_v1.1 features; 2 = PAMAP2 features
#endif

#if SCENARIO_FEATURES == 1
#define WSIZE 200
#ifndef NUM_FEATURES
	#define NUM_FEATURES 43
#endif
#define OVERLAP (WSIZE/2) //(WSIZE/4) // min is 0 and max is W_SIZE-1
#define NUM_DATA_SAMPLE 3
#define NUM_CLASSES 6
#define ORIG_NUM_TRAINING_SAMPLES 738442
#define ORIG_NUM_TESTING_SAMPLES 316472

#elif SCENARIO_FEATURES == 2
#define WSIZE 300
#ifndef NUM_FEATURES
	#define NUM_FEATURES 90
#endif
#define OVERLAP (WSIZE/10) //(WSIZE/4) // min is 0 and max is W_SIZE-1
#define NUM_DATA_SAMPLE 27
#define NUM_CLASSES 22
#define ORIG_NUM_TRAINING_SAMPLES 1670430
#define ORIG_NUM_TESTING_SAMPLES 272442

#endif

#define NUM_KNOWN_POINTS ((ORIG_NUM_TRAINING_SAMPLES-WSIZE)/(WSIZE-OVERLAP)+1)
#define NUM_NEW_POINTS ((ORIG_NUM_TESTING_SAMPLES-WSIZE)/(WSIZE-OVERLAP)+1)

#define NUM_TRAINING_SAMPLES (WSIZE +(NUM_KNOWN_POINTS-1)*(WSIZE-OVERLAP))
#define NUM_TESTING_SAMPLES (WSIZE +(NUM_NEW_POINTS-1)*(WSIZE-OVERLAP))


#ifndef K
#define K 3 // 3, 5 or 20 (some authors consider K=sqrt(NUM_TRAINING_INSTANCES)
#endif

#ifndef DT
#define DT 1 // 1: double; 2: float; 3: not used for now
#endif

#if DT == 1	//double
	#define DATA_TYPE double
	#define MAX_FP_VAL DBL_MAX
	#define MIN_FP_VAL -DBL_MAX
#elif DT == 2 //float
	#define DATA_TYPE float
	#define MAX_FP_VAL FLT_MAX
	#define MIN_FP_VAL -FLT_MAX
#endif

#if NUM_CLASSES > 128
	#define CLASS_ID_TYPE int16_t  // consider 0..32767 classes and -1 for unknown
#else
	#define CLASS_ID_TYPE int8_t  // consider 0..127 classes and -1 for unknown
#endif

typedef uint8_t K_TYPE;

#endif //KNN_PARAMS_H