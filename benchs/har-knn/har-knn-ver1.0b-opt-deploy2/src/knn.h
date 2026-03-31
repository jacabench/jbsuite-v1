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
#include "features.h"

#ifndef DIMEM
#define DIMEM 0 // 0: not using dynamic memory allocation
#endif

#if DIMEM != 0
#include <stdlib.h>
#endif

CLASS_ID_TYPE features_normalize_classify(Instance *new_instances, Point *new_point, Point *known_points);

#if SPECIALIZED == 1 && K == 3
void initialize_3_best(BestPoint *best_points);

void update_3_best(DATA_TYPE distance, CLASS_ID_TYPE classID, BestPoint *best_points);

void get_3_NN(Point *new_point, Point *known_points, int num_points, BestPoint *best_points, int num_features);

CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points);

CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points, int num_points, int num_features);

#else
void initialize_best(BestPoint *best_points, K_TYPE k);

void update_best(DATA_TYPE distance, CLASS_ID_TYPE classID, BestPoint *best_points, K_TYPE k);

void get_k_NN(Point *new_point, Point *known_points, int num_points, BestPoint *best_points, K_TYPE k,  int num_features);

CLASS_ID_TYPE plurality_voting(K_TYPE k, BestPoint *best_points, int num_classes);

CLASS_ID_TYPE knn_classifyinstance(Point *new_point, K_TYPE k, int num_classes, Point *known_points, int num_points, int num_features);
#endif	

#endif
