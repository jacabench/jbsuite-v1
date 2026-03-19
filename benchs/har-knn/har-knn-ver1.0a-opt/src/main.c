/**
*	
*	JACABench - JACA Benchmark Suite v1.0	
*
*	Human Activity Recognition (HAR) using a kNN for inference.
*
*	Version 1.0
*	September 2025
*	Copyright University of Porto, Faculty of Engineering (FEUP), Porto, Portugal
*
*	Author: João MP Cardoso 
*	Email = jmpc@fe.up.pt
*/

/*	Scenario A, WISDM dataset
*	Files in scenario-wisdm/
*	READ 4 // default 4
*	NUM_TRAINING_SAMPLES 738400
*	NUM_TESTING_SAMPLES 316400
*	NUM_FEATURES 43 
*	NUM_CLASSES 6
*
*	Scenario B, PAMAP2 dataset
*	Files in scenario-pamap2/
*	READ 4 // default 4
*	NUM_TRAINING_SAMPLES 1670250
*	NUM_TESTING_SAMPLES 272190
*	NUM_FEATURES 90 
*	NUM_CLASSES 22 // 21 activities + 1 activity representing transient activities
*
*	TIMING 0 or *1* // default 1
*	ACCURACY 0 or *1* // default 1
*	K = *3*, 5 or 20 // default 3
* 	DATA_TYPE = float or double // default double
*	DIMEM 0 or *1* // default 0, i.e., no dynamic memory allocation
*	STREAMING 0 or *1* // default 1, i.e., streaming
*
*	Both use minmax normalization of features
*/

/*
* Inputs
*	READ = 1	READ = 2	READ = 3		READ = 4
*	Static		From file	Static			From file
*	--------------------------------------------------
* 	Raw			Raw			Vectors			Vectors
*	train1.dat 	train2.dat 	train1_norm.dat train2_norm.dat 
*	--------------------------------------------------
* 	Raw			Raw			Raw				Raw
*	test1.dat	test2.dat	test1.dat		test2.dat
*/

#ifndef READ
	#define READ 4 // 1 and 3: static initalization; 2 and 4: read from .dat
#endif

#if READ == 2
#ifndef TRAIN_PATH
#define TRAIN_PATH "train2.dat"
#endif
#ifndef TEST_PATH
#define TEST_PATH "test2.dat"
#endif

#elif READ == 4
#ifndef TRAIN_PATH
#define TRAIN_PATH "train2_norm.dat" 
#endif
#ifndef TEST_PATH
#define TEST_PATH "test2.dat"
#endif
#endif

#ifndef TIMING
	#define TIMING 1 // 0: w/o timing; 1: w/ timing
#endif

#ifndef VERIFY
	#define VERIFY 0 	// 0: none verification;
						// 1: to verify if the results are according to the ones expected
#endif

#ifndef ACCURACY
	#define ACCURACY 1 	// 0: w/o reporting; 1: reporting the accuracy of the classification
#endif

#ifndef STREAMING
	#define STREAMING 1 
	// 0: all the points are first normalized and only then the classification occurs; 
	// 1: each point is normalized in the loop of the classification
#endif

#ifndef SPECIALIZED
#define SPECIALIZED 1 // use functions according to the value of K
#endif

#ifndef TOP_K
#define TOP_K 1 // the option to determine the k nearest
#endif

#ifndef USE_SQRT
#define USE_SQRT 1 // use sqrt in Euclidean distance
#endif

#ifndef MATH_TYPE
#define MATH_TYPE 1 // use the math.h functions according to the type float or double, e.g., sqrt or sqrtf 
#endif

#ifndef DIST_METHOD
#define DIST_METHOD 1 // calculate, 1: Euclidean distance; 2: Manhattan distance
#endif 

#include <stdio.h>
#include "timing.h"
#include "params.h"
#include "types.h"
#include "utils.h"
#include "io.h"
#include "knn.h"
#include "features.h"

#if READ == 1 // data embedded in program, arrays statically filled
	
	Instance known_instances1[ORIG_NUM_TRAINING_SAMPLES] = {
					#include "train1.dat"
				}; //{{{1,2,3,4,5},1}};
	Instance *known_instances = known_instances1;

	Instance new_instances1[ORIG_NUM_TESTING_SAMPLES] = {
					#include "test1.dat"
				}; //{{{2.0,1.0, 1.0},0},{{2.0,3.0,2.0},0},...};
	Instance *new_instances = new_instances1;
	
	Point known_points1[NUM_KNOWN_POINTS];
	Point *known_points = known_points1;

#elif READ == 2 // data read from files 

	#if DIMEM == 0
	Instance known_instances1[ORIG_NUM_TRAINING_SAMPLES];
	Instance *known_instances = known_instances1;

	Instance new_instances1[ORIG_NUM_TESTING_SAMPLES];
	Instance *new_instances = new_instances1;
	
	Point known_points1[NUM_KNOWN_POINTS];
	Point *known_points = known_points1;
    #else
	Point *known_points;
	Instance *known_instances;
	Instance *new_instances;
	#endif

#elif READ == 3 // data embedded in program, arrays statically filled
	
	Point known_points1[NUM_KNOWN_POINTS] = {
					#include "train1_norm.dat"
				}; //{{{1,2,3,4,5},1}};
	Point *known_points = known_points1;

	Instance new_instances1[ORIG_NUM_TESTING_SAMPLES] = {
					#include "test1.dat"
				}; //{{{2.0,1.0, 1.0},0},{{2.0,3.0,2.0},0},...};
	Instance *new_instances = new_instances1;

#elif READ == 4 // data read from files 

	#if DIMEM == 0	
	Point known_points1[NUM_KNOWN_POINTS];
	Point *known_points = known_points1;
	Instance new_instances1[ORIG_NUM_TESTING_SAMPLES];
	Instance *new_instances = new_instances1;
	#else
	Point *known_points;
	Instance *new_instances;
	#endif
#endif

#if STREAMING == 0
#if DIMEM == 0
Point new_points1[NUM_NEW_POINTS]; //{{{2.0,1.0,1.0},0},{{2.0,3.0,2.0},0},...};
Point *new_points = new_points1;
#else
Point *new_points;
#endif
#endif

#if VERIFY == 1  // verify if classifications are still as the golden ones
	CLASS_ID_TYPE key[NUM_NEW_POINTS] = {
		#if K == 20
			#include "key-READ4-k20.dat"
		#elif K == 3
			#include "key-READ4-k3.dat"
		#elif K == 5
			#include "key-READ4-k5.dat"
		#elif K == 7
			#include "key-READ4-k7.dat"
		#endif
	};
#endif

// for buffering the input data in windows of samples input to feature extration 
// other versions may consider the use of one arrays or FIFO foreach sensor source 
DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE];

int main() {

	const K_TYPE k = K;
	const int num_features = NUM_FEATURES;
	const int num_sensor_data = NUM_DATA_SAMPLE;
	const int window_size = WSIZE;
	const int overlap = OVERLAP;
	const int num_known_samples = NUM_TRAINING_SAMPLES;
	const int num_new_samples = NUM_TESTING_SAMPLES;
    const int num_classes = NUM_CLASSES;
    const int num_known_points = NUM_KNOWN_POINTS;
	const int num_new_points = NUM_NEW_POINTS;
	
	#if DIMEM == 1 && READ == 2
		known_points = (Point *) malloc(NUM_KNOWN_POINTS*sizeof(Point));
		known_instances = (Instance *) malloc(ORIG_NUM_TRAINING_SAMPLES*sizeof(Instance));
		new_instances = (Instance *) malloc(ORIG_NUM_TESTING_SAMPLES*sizeof(Instance)) ;
	#elif DIMEM == 1 && READ == 4
		known_points = (Point *) malloc(NUM_KNOWN_POINTS*sizeof(Point));
		new_instances = (Instance *) malloc(ORIG_NUM_TESTING_SAMPLES*sizeof(Instance));
	#endif
	
	#if STREAMING == 0 && DIMEM == 1
		new_points = (Point *) malloc(NUM_NEW_POINTS*sizeof(Point));
	#endif

	#if READ == 1 || READ == 3 // data embedded in program
		printf("All statically initialized from data in files\n");
	#elif READ == 2  // data read from .dat files
		printf("Initializing raw training data from .dat ...\n");
		read_data_instances(TRAIN_PATH, num_sensor_data, num_known_samples, known_instances); 
		printf("Initializing raw testing data from .dat ...\n");
		read_data_instances(TEST_PATH, num_sensor_data, num_new_samples, new_instances);
		printf("Initialization done.\n\n");
	#elif READ == 4  // data read from .dat files
		printf("Initializing normalized training vectors of features from .dat ...\n");
		read_data_points(TRAIN_PATH, num_features, num_known_points, known_points); 
		printf("Initializing raw testing data from .dat ...\n");
		read_data_instances(TEST_PATH, num_sensor_data, num_new_samples, new_instances);
		printf("Initialization done.\n\n");
	#endif
	
	// for the knowledge base
	#if READ == 1 || READ == 2 // consider raw data and calculates vector of features
		do_features_forall(known_instances, num_sensor_data, num_known_samples, known_points, num_classes, buffers, window_size, overlap);
	#endif
	
	#if STREAMING == 0 // no streaming: calculates vector of features
		do_features_forall(new_instances, num_sensor_data, num_new_samples, new_points, num_classes, buffers, window_size, overlap);
	#endif
	
	#if READ == 3 || READ == 4 // load min and max feature values for normalization
		DATA_TYPE min[NUM_FEATURES] = {
				#include "min.dat"
			};
		DATA_TYPE max[NUM_FEATURES] = {
				#include "max.dat"
			};
	#elif READ == 1 || READ == 2 // calculate min and max feature values for normalization
		DATA_TYPE min[NUM_FEATURES];
		DATA_TYPE max[NUM_FEATURES];
		// determine min and max from known points
		minmax(min, max, num_known_points, known_points, num_features);
		
		// normalize known points
		minmax_normalize(min, max, num_known_points, known_points, num_features);
	#endif
		
	#if STREAMING == 0 // no streaming: normalize all features of test vectors before 
		minmax_normalize(min, max, num_new_points, new_points, num_features);
	#endif
	
	#if ACCURACY == 1
	int fail = 0; // count the number of test instances incorrectly classified
	#endif
	
	#if STREAMING == 1
	int index = 0;
	#endif
	
    printf("Executing kNN...\n");

	__INIT_TIMING();
	__START_TIMING();
	
	// loop over the input instances to classify.
	// Note that depending on the application this can be
	// instances arriving as streaming data.
	// Here assume that the loop below needs to run in serial mode and the
	// value of num_new_points is just to test
    for (int i = 0; i < num_new_points; i++) {
		// get instance to classify
		// in a streaming implementation this might involve something like: get(new_point);
		#if STREAMING == 0
		Point *new_point = &new_points[i];	

		#elif STREAMING == 1 
		Point new_point1;
		Point *new_point = &new_point1;
		CLASS_ID_TYPE classid;
		
		do_class(&classid, &new_instances[index], window_size, num_classes);
		fill_buffers(&new_instances[index], num_sensor_data, buffers, window_size);	
		new_point->classification_id = classid;
		
		do_features(new_point, buffers, window_size);	
		
		index += window_size-overlap;
		
		// normalize the point to classify
		minmax_normalize_point(min, max, new_point, num_features);
		#endif

		// classify
		#if SPECIALIZED == 1 && K == 3
        CLASS_ID_TYPE instance_class = knn_classifyinstance_3(new_point,
                                       known_points, num_known_points, num_features);
		#else
        CLASS_ID_TYPE instance_class = knn_classifyinstance(new_point, k, num_classes,
                                       known_points, num_known_points, num_features);
		#endif

		#if ACCURACY
			if(new_point->classification_id != instance_class) fail++;
		#endif
		
        // The following store the inferred class in the point structure
        // In practice and especially in streaming operation, this
        // may not be done and the class is output to the subsequent
        // stages of the application
        // For now this is used to verify the results by comparing
        // the class obtained for each point to a golden class
		// for now: output the inferred class of the instance
		put(&instance_class, 0, "class id:"); // 0 is the id used for char
		
		#if STREAMING == 0 // store the inferred class: we don't need the label anymore
		new_point->classification_id = instance_class;
		#endif
    }

	__END_TIMING();
	
    printf("kNN done.\n\n");

    printf("HAR hyperparameters: number of samples per sensor reading = %d\n", num_sensor_data);
    printf("HAR: number of raw data training instances = %d\n", num_known_samples);
    printf("HAR: number of raw data testing instances = %d\n", num_new_samples);
    printf("HAR: number of features = %d\n", num_features);
    printf("HAR: number of classes = %d\n", num_classes);
	
    printf("HAR hyperparameters: window size = %d samples\n", window_size);
    printf("HAR hyperparameters: overlap = %.2f %s, %d instance(s)\n", ((float) overlap/window_size*100), "%", overlap);
    
	printf("kNN: k = %d\n", k);
    printf("kNN: number of training instances = %d\n", num_known_points);
    printf("kNN: number of testing instances = %d\n", num_new_points);

    #if DT == 2
      printf("kNN: data type used = float\n");
    #elif DT == 1
      printf("kNN: data type used = double\n");
	#endif

    printf("kNN: number of classified instances = %d\n", num_new_points);

	#if ACCURACY == 1
		printf("kNN: number of classifications wrong = %d\n", fail);
		printf("kNN: number of classifications right = %d\n", num_new_points-fail);
		printf("kNN: accuracy = %.2f %s\n\n", 100*((float)(num_new_points-fail)/(float) num_new_points),"%");
	#endif

	#if VERIFY == 1
		verify_results(num_new_points, new_known_points, key);
	#endif

	__REPORT_TIMING_S();
	
	#if TIMING == 1	
		printf("Inference time: %.4f ms\n", (time/num_new_points*1000));
		printf("Throughput: %.2f inferences/s\n", num_new_points/time);
	#endif

	#if DIMEM != 0
		#if STREAMING == 0
		free(new_points);
		#endif
		free(known_points);
		#if READ == 1 || READ == 2
		free(known_instances);
		#endif
		free(new_instances);
	#endif

    return 0;
}