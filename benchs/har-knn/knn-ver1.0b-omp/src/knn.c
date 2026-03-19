/**
*	
*	JACABench - JACA Benchmark Suite v1.0	
*
*	a kNN implementation using a train/test dataset from Human Activity Recognition (HAR)
*
*	Version 1.0
*	September 2025
*	Copyright University of Porto, Faculty of Engineering (FEUP), Porto, Portugal
*
*	Author: João MP Cardoso 
*	Email = jmpc@fe.up.pt
*/

#include "knn.h"

#if SPECIALIZED == 1 && K == 3
#define SWAP(a,b,c) c = a; a = b; b = c;  

/*
* Initialize the data structure to store the k best (nearest) points.
*/
void initialize_3_best(BestPoint *best_3x_points) {

    for (int i = 0; i < 3*NUM_THREADS; i++) {
        BestPoint *bp = &(best_3x_points[i]);
        bp->distance = MAX_FP_VAL;
        bp->classification_id = (CLASS_ID_TYPE) -1; // unknown
    }
}

/*
* Keep the data structure with the k nearest points updated.
* It receives a new Point and updates the k nearest accordingly.
*/
void update_3_best(DATA_TYPE distance, CLASS_ID_TYPE classID, BestPoint *best_points) {
	
	#if UPDATE == 2 // insertion sort
	for (int i = 0; i < 3; i++) {
		if (distance <= best_points[i].distance) {
			for (int j = 2; j > i; j--) {
				best_points[j].distance = best_points[j-1].distance;
				best_points[j].classification_id = best_points[j-1].classification_id;
			}
			best_points[i].classification_id = classID;
			best_points[i].distance = distance;
			break;
		}
	}
	#elif UPDATE == 1 //find the farmost Point in the best_points, i.e., the point with the longest distance
    DATA_TYPE max_distance = (DATA_TYPE) best_points[0].distance;
    int index = 0;

    for (int i = 1; i < 3; i++) {
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
	#endif
}

/*
* Merge data from different data structures with the k nearest points updated.
*/
void merge_in_3_best(BestPoint *best_3x_points, BestPoint *best_points) {

    DATA_TYPE min;
    int index;
	
    //find the farmost Point in the best_points, i.e., the point with the longest distance
    for(int j=0; j < 3; j++) {
		min = best_3x_points[0].distance;
		index = 0;
		for (int i = 1; i < 3*NUM_THREADS; i++) {
			if (best_3x_points[i].distance < min) {
				min = best_3x_points[i].distance;
				index = i;
			}
		}
		
		best_3x_points[index].distance = MAX_FP_VAL;
		best_points[j].classification_id = best_3x_points[index].classification_id;
		best_points[j].distance = min;
	}
}

/*
*	Classify using the 3 nearest neighbors identified by the get_3_NN
*	function. The classification uses plurality voting.
*
*	Note: it assumes that classes are identified from 0 to num_classes - 1.
*/
/*
*	Classify using the 3 nearest neighbors identified by the get_3_NN function. 
*   This function is a specialized version of plurality voting.
*
*	Note: it assumes that classes are identified from 0 to num_classes - 1.
*/
CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points) {

	CLASS_ID_TYPE ids0 = best_points[0].classification_id;
    CLASS_ID_TYPE ids1 = best_points[1].classification_id;
    CLASS_ID_TYPE ids2 = best_points[2].classification_id;
	
	CLASS_ID_TYPE classification_id = ids0; // the first one is the class when the 3 are different and are sorted
		
   if(ids0 == ids2)
        classification_id = ids0;
    else if(ids1 == ids2)
        classification_id = ids1;

	#if UPDATE == 1 // when the best points are not sorted
	if((ids0 != ids2) && (ids0 != ids1) && (ids1 != ids2)) { // all different
		DATA_TYPE min_dist = best_points[0].distance;
		int index = 0;
		if(best_points[1].distance < min_dist) { min_dist = best_points[1].distance; index = 1;}
		if(best_points[2].distance < min_dist) { index = 2;}
		classification_id = best_points[index].classification_id;
	}
	#endif
	
    return classification_id;
}

void get_3_NN(Point *new_point, Point *known_points, int num_points,
	BestPoint *best_points, BestPoint *best_3x_points, int num_features) {

    int i,j;
    DATA_TYPE distance;
    int thread_id;
		
    // calculate the Euclidean distance between the Point to classify and each Point in the
    // training dataset (knowledge base)
	#pragma omp parallel for private(i, j, distance) num_threads(NUM_THREADS)
    for (i = 0; i < num_points; i++) {
		thread_id = omp_get_thread_num();
        distance = (DATA_TYPE) 0.0;

		#if DIST_METHOD == 1 // calculate the Euclidean distance
        for (j = 0; j < num_features; j++) {
            DATA_TYPE diff = (DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j];
            distance += diff * diff;
        }
		#if USE_SQRT == 1
			#if MATH_TYPE == 1 && DT == 2 // float
				distance = sqrtf(distance);
			#else // double
				distance = sqrt(distance);
			#endif
		#endif
	
		#elif DIST_METHOD == 2 // calculate the Manhattan distance
		for (j = 0; j < num_features; j++) {
			#if MATH_TYPE == 1 && DT == 2 // float
            DATA_TYPE absdiff = fabsf((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
            #else // double
            DATA_TYPE absdiff = fabs((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
			#endif
			distance += absdiff;
        }
		#endif

	    // update the 3 nearest Points
        update_3_best(distance, known_points[i].classification_id, &best_3x_points[thread_id*3]);
    }
	merge_in_3_best(best_3x_points, best_points);
}

/*
* Classify a given Point (instance) with a code version specialized for k=3.
* It returns the classified class ID.
*/
CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features) {

	//BestPoint *best_points = (BestPoint *) calloc(k, sizeof(BestPoint)) ;
    BestPoint best_points[3]; // Array with the k nearest points to the Point to classify
	BestPoint best_3x_points[3*NUM_THREADS]; // Array with the k nearest points to the Point to classify per each thread
	
    initialize_3_best(best_3x_points);

    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_3_NN(new_point, known_points, num_points, best_points, best_3x_points, num_features);

	// use plurality voting of 3 classes to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting_3(best_points);
	
	return classID;
}

#else

/*
* Initialize the data structure to store the k best (nearest) points.
*/
void initialize_best(BestPoint *best_3x_points, K_TYPE  k) {

    for (int i = 0; i < k*NUM_THREADS; i++) {
        BestPoint *bp = &(best_3x_points[i]);
        bp->distance = MAX_FP_VAL;
        bp->classification_id = (CLASS_ID_TYPE) -1; // unknown
    }
}

/*
* Keep the data structure with the k nearest points updated.
* It receives a new Point and updates the k nearest accordingly.
*/
void update_best(DATA_TYPE distance, CLASS_ID_TYPE classID, BestPoint *best_points, K_TYPE  k) {

    #if UPDATE == 2 // insertion sort
	for (int i = 0; i < k; i++) {
		if (distance < best_points[i].distance) {
			for (int j = k-1; j > i; j--) {
				best_points[j].distance = best_points[j-1].distance;
				best_points[j].classification_id = best_points[j-1].classification_id;
			}
			best_points[i].classification_id = classID;
			best_points[i].distance = distance;
			break;
		}
	}
	#elif UPDATE == 1 //find the farmost Point in the best_points, i.e., the point with the longest distance
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
	#endif
}

/*
* Merge data from different data structures with the k nearest points updated.
*/
void merge_in_best(BestPoint *best_kx_points, BestPoint *best_points, K_TYPE  k) {

    DATA_TYPE min;
    int index;
	
    //find the farmost Point in the best_points, i.e., the point with the longest distance
    for(int j=0; j < k; j++) {
		min = best_kx_points[0].distance;
		index = 0;
		for (int i = 1; i < k*NUM_THREADS; i++) {
			if (best_kx_points[i].distance < min) {
				min = best_kx_points[i].distance;
				index = i;
			}
		}
		
		best_kx_points[index].distance = MAX_FP_VAL;
		best_points[j].classification_id = best_kx_points[index].classification_id;
		best_points[j].distance = min;
	}
}

/*
* Main kNN function.
* It calculates the distances and calculates the nearest k points.
*/
void get_k_NN(Point *new_point, Point *known_points, int num_points,
	BestPoint *best_points, BestPoint *best_kx_points, K_TYPE k, int num_features) {
	
    int i,j;
    DATA_TYPE distance;
    int thread_id;

    // calculate the Euclidean distance between the Point to classify and each Point in the
    // training dataset (knowledge base)
	#pragma omp parallel for private(i, j, distance) num_threads(NUM_THREADS)
    for (i = 0; i < num_points; i++) {
		thread_id = omp_get_thread_num();
        distance = (DATA_TYPE) 0.0;

		#if DIST_METHOD == 1 // calculate the Euclidean distance
        for (j = 0; j < num_features; j++) {
            DATA_TYPE diff = (DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j];
            distance += diff * diff;
        }
		#if USE_SQRT == 1
			#if MATH_TYPE == 1 && DT == 2 // float
				distance = sqrtf(distance);
			#else // double
				distance = sqrt(distance);
			#endif
		#endif
	
		#elif DIST_METHOD == 2 // calculate the Manhattan distance
		for (j = 0; j < num_features; j++) {
			#if MATH_TYPE == 1 && DT == 2 // float
            DATA_TYPE absdiff = fabsf((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
            #else // double
            DATA_TYPE absdiff = fabs((DATA_TYPE) new_point->features[j] - (DATA_TYPE) known_points[i].features[j]);
			#endif
			distance += absdiff;
        }
		#endif
		
		// update the k nearest Points
        update_best(distance, known_points[i].classification_id, &best_kx_points[thread_id*k], k);
    }
    merge_in_best(best_kx_points, best_points, k);
}

/*
*	Classify using the k nearest neighbors identified by the get_k_NN
*	function. The classification uses plurality voting.
*
*	Note: it assumes that classes are identified from 0 to num_classes - 1.
*/
CLASS_ID_TYPE plurality_voting(K_TYPE k, BestPoint *best_points, int num_classes) {
  CLASS_ID_TYPE classification_id;

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
	
    K_TYPE max = 1; // maximum is k
    for (int i = 0; i < num_classes; i++) {
        if (histogram[i] > max) {
            max = histogram[i];
            classification_id = (CLASS_ID_TYPE) i;
        }
    }

    #if UPDATE == 2 // sorted
	if(max == 1) classification_id = best_points[0].classification_id;
    #elif UPDATE == 1 // not sorted
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
    #endif

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
	BestPoint *best_kx_points = (BestPoint *) malloc(k*NUM_THREADS, sizeof(BestPoint)) ;
    #else
	BestPoint best_points[k]; // Array with the k nearest points to the Point to classify
	BestPoint best_kx_points[k*NUM_THREADS]; // Array with the k nearest points to the Point to classify per each thread
	#endif

    initialize_best(best_kx_points, k);

    // calculate the distances of the new point to each of the known points and get
    // the k nearest points
    get_k_NN(new_point, known_points, num_points, best_points, best_kx_points, k, num_features);

	// use plurality voting to return the class inferred for the new point
	CLASS_ID_TYPE classID = plurality_voting(k, best_points, num_classes);

	#if DIMEM != 0
	free(best_points);
	free(best_kx_points);
	#endif
	
	return classID;
}

#endif