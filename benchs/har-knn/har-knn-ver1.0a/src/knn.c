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

#include <math.h>

#include "knn.h"
#include "params.h"

/**
*  Copy the top k nearest points (first k elements of dist_points)
*  to a data structure (best_points) with k points
*/
void copy_k_nearest(BestPoint *dist_points, BestPoint *best_points, K_TYPE k) {

    for(int i = 0; i < k; i++) {   // we only need the top k minimum distances
       best_points[i].classification_id = dist_points[i].classification_id;
       best_points[i].distance = dist_points[i].distance;
    }

}

/**
*  Get the k nearest points.
*  This version stores the k nearest points in the first k positions of dist_point
*/
void select_k_nearest(BestPoint *dist_points, int num_points, K_TYPE k) {

    DATA_TYPE min_distance, distance_i;
    CLASS_ID_TYPE class_id_1;
    int index;

    for(int i = 0; i < k; i++) {  // we only need the top k minimum distances
        
        min_distance = dist_points[i].distance;
		index = i;
		for(int j = i+1; j < num_points; j++) {
            if(dist_points[j].distance < min_distance) {
                min_distance = dist_points[j].distance;
                index = j;
            }
		}
		if(index != i) { //swap
			distance_i = dist_points[index].distance;
			class_id_1 = dist_points[index].classification_id;

			dist_points[index].distance = dist_points[i].distance;
			dist_points[index].classification_id = dist_points[i].classification_id;

			dist_points[i].distance = distance_i;
			dist_points[i].classification_id = class_id_1;
		}
    }
}

/*
* Main kNN function.
* It calculates the distances and calculates the nearest k points.
*/
void get_k_NN(Point *new_point, Point *known_points, int num_points,
	BestPoint *best_points, K_TYPE k, int num_features) {
      
	//printf("num points %d, num featurs %d\n", num_points, num_features);
	#if DIMEM == 0
	BestPoint dist_points[num_points];
	//BestPoint dist_points[NUM_KNOWN_POINTS];
    #else
	BestPoint *dist_points = (BestPoint *) malloc(num_points*sizeof(BestPoint));
	#endif 
	 
    // calculate the Euclidean distance between the Point to classify and each Point in the
    // training dataset (knowledge base)
    for (int i = 0; i < num_points; i++) {
        DATA_TYPE distance = (DATA_TYPE) 0.0;

        // calculate the Euclidean distance
        for (int j = 0; j < num_features; j++) {
          DATA_TYPE diff = (DATA_TYPE)new_point->features[j] -
                           (DATA_TYPE)known_points[i].features[j];
          distance += diff * diff;
        }		
		distance = sqrt(distance);
		
        dist_points[i].classification_id = known_points[i].classification_id;
        dist_points[i].distance = distance;
    }

	// select the k nearest Points: k first elements of dist_points
    select_k_nearest(dist_points, num_points, k);

	// copy the k first elements of dist_points to the best_points
	// data structure
    copy_k_nearest(dist_points, best_points, k);
	
	#if DIMEM != 0
	free(dist_points);
	#endif
}

/*
*	Classify using the k nearest neighbors identified by the get_k_NN
*	function. The classification uses plurality voting.
*
*	Note: it assumes that classes are identified from 0 to
*	num_classes - 1.
*/
CLASS_ID_TYPE plurality_voting(K_TYPE k, BestPoint *best_points, int num_classes) {

	#if DIMEM == 0
	K_TYPE histogram[num_classes];  // maximum equals the value of k;

    //initialize the histogram
    for (int i = 0; i < num_classes; i++) {
        histogram[i] = 0;
    }
    #else
	K_TYPE *histogram = (K_TYPE *) calloc(NUM_CLASSES, sizeof(CLASS_ID_TYPE)) ;
	#endif

    // build the histogram
    for (int i = 0; i < k; i++) {
        BestPoint p = best_points[i];
        histogram[(int) p.classification_id] += 1;
    }
	
	CLASS_ID_TYPE classification_id = best_points[0].classification_id;
    K_TYPE max = 1; // maximum is k
    for (int i = 0; i < num_classes; i++) {
        if (histogram[i] > max) {
            max = histogram[i];
            classification_id = (CLASS_ID_TYPE) i;
        }
    }

	#if DIMEM != 0
	free(histogram);
	#endif
	
    return classification_id;
}

/*
* Classify a given Point (instance).
* It returns the classified class ID.
*/
CLASS_ID_TYPE knn_classifyinstance(Point *new_point, K_TYPE k, int num_classes, Point *known_points, int num_points, int num_features) {

	//BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
    //BestPoint best_points[k]; // Array with the k nearest points to the Point to classify
    BestPoint best_points[k]; // Array with the k nearest points to the Point to classify

    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_k_NN(new_point, known_points, num_points, best_points, k, num_features);

	// use plurality voting to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting(k, best_points, num_classes);

	return classID;
}