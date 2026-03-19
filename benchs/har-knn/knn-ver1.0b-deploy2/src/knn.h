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

#ifndef DIMEM
#define DIMEM 0 // 0: not using dynamic memory allocation
#endif

#if DIMEM != 0
#include <stdlib.h>
#endif

#include <math.h>
#include "params.h"
#include "types.h"
#include "features.h"


CLASS_ID_TYPE normalize_classify(Point *new_point, Point *known_points);

#if SPECIALIZED == 1 && K == 3
void initialize_3_best(BestPoint *best_points);

void update_3_best(DATA_TYPE distance, CLASS_ID_TYPE classID, BestPoint *best_points);

void get_3_NN(Point *new_point, Point *known_points, BestPoint *best_points);

CLASS_ID_TYPE plurality_voting_3(BestPoint *best_points);

CLASS_ID_TYPE knn_classifyinstance_3(Point *new_point, Point *known_points);

#else

void initialize_best(BestPoint *best_points);

void update_best(DATA_TYPE distance, CLASS_ID_TYPE classID, BestPoint *best_points);

void get_k_NN(Point *new_point, Point *known_points, BestPoint *best_points);

CLASS_ID_TYPE plurality_voting(BestPoint *best_points);

CLASS_ID_TYPE knn_classifyinstance(Point *new_point, Point *known_points);
#endif	

#endif // KNN_H
