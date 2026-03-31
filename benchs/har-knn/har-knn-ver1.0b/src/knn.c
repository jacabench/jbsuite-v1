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

/*
* Initialize the data structure to store the k best (nearest) points.
*/
void initialize_best(BestPoint *best_points, K_TYPE  k) {

    for (int i = 0; i < k; i++) {
        BestPoint *bp = &(best_points[i]);
        bp->distance = MAX_FP_VAL;
        //printf("initialize distance %e\n", bp->distance);
        bp->classification_id = (CLASS_ID_TYPE) -1; // unknown
    }
}

/*
* Keep the data structure with the k nearest points updated.
* It receives a new Point and updates the k nearest accordingly.
*/
void update_best(DATA_TYPE distance, CLASS_ID_TYPE classID, BestPoint *best_points, K_TYPE  k) {

    DATA_TYPE max_distance = (DATA_TYPE) best_points[0].distance;
    int index = 0;

    for (int i = 1; i < k; i++) {
        if (best_points[i].distance > max_distance) {
            max_distance = best_points[i].distance;
            index = i;
        }
    }
    // if the Point is near (shorter distance) than the farmost one (longest distance)
    // in the best_points update best_points substituting the farmost one
    if (distance < max_distance) {
		best_points[index].classification_id = classID;
		best_points[index].distance = distance;
    }
}

/*
* Main kNN function.
* It calculates the distances and calculates the nearest k points.
*/
void get_k_NN(Point *new_point, Point *known_points, int num_points,
	BestPoint *best_points, K_TYPE k,  int num_features) {
	 
    // calculate the Euclidean distance between the Point to classify and each Point in the
    // training dataset (knowledge base)
    for (int i = 0; i < num_points; i++) {
        DATA_TYPE distance = (DATA_TYPE) 0.0;

        for (int j = 0; j < num_features; j++) {
            DATA_TYPE diff = (DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j];
            distance += diff * diff;
        }
		distance = sqrt(distance);
		
		// update the k nearest Points
        update_best(distance, known_points[i].classification_id, best_points, k);
    }
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
	K_TYPE *histogram = (K_TYPE *) calloc(num_classes, sizeof(CLASS_ID_TYPE)) ;
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

    if(max == 1) {
        DATA_TYPE min_dist = best_points[0].distance;
        int index = 0;
        for (int i = 1; i < k; i++) {
            if(min_dist > best_points[i].distance) { 
                min_dist = best_points[i].distance;
                index = i;
            }
        }
        classification_id = best_points[index].classification_id;
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

	#if DIMEM != 0
	BestPoint *best_points = (BestPoint *) malloc(k, sizeof(BestPoint)) ;
    #else
	BestPoint best_points[k]; // Array with the k nearest points to the Point to classify
	#endif

    initialize_best(best_points, k);

    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_k_NN(new_point, known_points, num_points, best_points, k, num_features);

	// use plurality voting to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting(k, best_points, num_classes);

	#if DIMEM != 0
	free(best_points);
	#endif

	return classID;
}