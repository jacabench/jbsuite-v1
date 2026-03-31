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
*	Both use minmax normalization of features
*/

/*
* Inputs
*	READ = 3			READ = 4
*	--------------------------------------------------
* 	Vectors	static		Vectors static
*	train1_norm.dat 	train1_norm.dat
*	--------------------------------------------------
* 	Raw	from file		Raw static
*	test1.dat			test2.dat
*/

#include <stdio.h>
#include "timing.h"
#include "params.h"
#include "types.h"
#include "utils.h"
#include "io.h"
#include "knn.h"
#include "features.h"

#if READ == 3 // data embedded in program, arrays statically filled
	Instance new_instances1[ORIG_NUM_TESTING_SAMPLES] = {
					#include "test1.dat"
				}; //{{{2.0,1.0, 1.0},0},{{2.0,3.0,2.0},0},...};
	Instance *new_instances = new_instances1;

#elif READ == 4 // data read from files 
	#if DIMEM == 0	
	Instance new_instances1[ORIG_NUM_TESTING_SAMPLES];
	Instance *new_instances = new_instances1;
	#else
	Instance *new_instances;
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

int main() {

	const int num_features = NUM_FEATURES;
	const int num_sensor_data = NUM_DATA_SAMPLE;
	const int window_size = WSIZE;
	const int overlap = OVERLAP;
	const int num_known_samples = NUM_TRAINING_SAMPLES;
	const int num_new_samples = NUM_TESTING_SAMPLES;
    const int num_classes = NUM_CLASSES;
    const int num_known_points = NUM_KNOWN_POINTS;
	const int num_new_points = NUM_NEW_POINTS;
	
	#if DIMEM == 1 && READ == 4
		new_instances = (Instance *) malloc(ORIG_NUM_TESTING_SAMPLES*sizeof(Instance));
	#endif

	#if READ == 3 // data embedded in program
		printf("All statically initialized from data in files\n");
	#elif READ == 4  // data read from .dat files
		printf("Initializing raw testing data from .dat ...\n");
		read_data_instances(TEST_PATH, num_sensor_data, num_new_samples, new_instances);
		printf("Initialization done.\n\n");
	#endif
	
	#if ACCURACY == 1
	int fail = 0; // count the number of test instances incorrectly classified
	#endif
	
	int index = 0;
	
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
		
		Point new_point;
		CLASS_ID_TYPE classid;

		do_class(&classid, &new_instances[index], window_size, num_classes);
		new_point.classification_id = classid;

		index += window_size-overlap;

		CLASS_ID_TYPE instance_class = features_normalize_classify(&new_instances[index], &new_point);

		#if ACCURACY
			if(new_point.classification_id != instance_class) fail++;
		#endif
		
        // The following store the inferred class in the point structure
        // In practice and especially in streaming operation, this
        // may not be done and the class is output to the subsequent
        // stages of the application
        // For now this is used to verify the results by comparing
        // the class obtained for each point to a golden class
		// for now: output the inferred class of the instance
		put(&instance_class, 0, "class id:"); // 0 is the id used for char
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
    
	printf("kNN: k = %d\n", K);
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
		free(new_instances);
	#endif

    return 0;
}