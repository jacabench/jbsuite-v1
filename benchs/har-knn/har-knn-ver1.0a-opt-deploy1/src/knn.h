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

#ifndef KNN_H
#define KNN_H

#include <math.h>
#include "params.h"
#include "types.h"

#ifndef DIMEM
#define DIMEM 0 // 0: not using dynamic memory allocation
#endif

#if DIMEM != 0
#include <stdlib.h>
#endif

CLASS_ID_TYPE features_normalize_classify(Instance *new_instances, 
						int window_size, int num_classes, int num_sensor_data, DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE],
						DATA_TYPE min[NUM_FEATURES], DATA_TYPE max[NUM_FEATURES], Point *new_point, int num_features,
						Point *known_points, int num_known_points, K_TYPE k);

#if SPECIALIZED == 1 && K == 3
void copy_3_nearest(BestPoint *dist_points, BestPoint *best_points);

void select_3_nearest(BestPoint *dist_points, int num_points);

void get_3_NN(Point *new_point, Point *known_points, int num_points, BestPoint *best_points, int num_features);

CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points);

CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features);

#else
#if TOP_K != 3 
void copy_k_nearest(BestPoint *dist_points, BestPoint *best_points, K_TYPE k);
#endif

#if TOP_K == 1 // select k elements
void select_k_nearest1(BestPoint *dist_points, int num_points, K_TYPE k);
#elif TOP_K == 2 // bubble sort for k elements
void select_k_nearest2(BestPoint *dist_points, int num_points, K_TYPE k);
#elif TOP_K == 3 // insertion sort for k elements
void select_k_nearest3(BestPoint *dist_points, BestPoint *best_points, int num_points, K_TYPE k);
#endif

void get_k_NN(Point *new_point, Point *known_points, int num_points, BestPoint *best_points, K_TYPE k,  int num_features);

CLASS_ID_TYPE plurality_voting(K_TYPE k, BestPoint *best_points, int num_classes);

CLASS_ID_TYPE knn_classifyinstance(Point *new_point, K_TYPE k, int num_classes, Point *known_points, int num_points, int num_features);
#endif	

#endif
