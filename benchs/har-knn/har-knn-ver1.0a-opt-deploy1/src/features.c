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

/**
* Functions for extracting the features from sensing data. 
*
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "params.h"
#include "utils.h"
#include "features.h"

// for WISDM and PAMAP2

void do_mean(DATA_TYPE *mean, DATA_TYPE *window, int wsize) {
	DATA_TYPE sum = (DATA_TYPE) 0.0;
	for(int i=0; i < wsize; i++) {
		sum += window[i*NUM_DATA_SAMPLE];
	}
	*mean = sum / wsize;
}

void do_variance(DATA_TYPE *variance, DATA_TYPE *window, int wsize) {
	DATA_TYPE mean;
	
	do_mean(&mean, window, wsize);
	
	*variance = (DATA_TYPE) 0.0;
	for(int i=0; i < wsize; i++) {
		*variance += (window[i*NUM_DATA_SAMPLE]-mean)*(window[i*NUM_DATA_SAMPLE]-mean);
	}
	*variance = *variance / (wsize-1);
}

void do_std(DATA_TYPE *std, DATA_TYPE *window, int wsize) {
	DATA_TYPE variance;
	do_variance(&variance, window, wsize);
	
	*std = sqrt(variance);
}

// WISDM
#if SCENARIO_FEATURES == 1

void do_absoldev(DATA_TYPE *absoldev, DATA_TYPE *window, int wsize) {
	DATA_TYPE mean, sum;
	
	do_mean(&mean, window, wsize);
	
	sum = (DATA_TYPE) 0.0;
	for(int i=0; i < wsize; i++) {
		sum += fabs(window[i*NUM_DATA_SAMPLE]-mean);
	}
	
	*absoldev = sum / wsize;
}

void do_max(DATA_TYPE *max, DATA_TYPE *window, int wsize){
	*max = MIN_FP_VAL;
	for(int i=0; i < wsize; i++) {
		if (*max < window[i*NUM_DATA_SAMPLE]) 
			*max = window[i*NUM_DATA_SAMPLE];
	}
}

void do_min(DATA_TYPE *min, DATA_TYPE *window, int wsize){
	*min = MAX_FP_VAL;
	for(int i=0; i < wsize; i++) {
		if (*min > window[i*NUM_DATA_SAMPLE]) 
			*min = window[i*NUM_DATA_SAMPLE];
	}
}

//X0..x9, Y0..Y9, Z0..Z9 are bins, their values are the fraction
//	of accelerometer samples that fell within that bin
void do_bins10(int *bins, DATA_TYPE *window, int wsize) {
	const int num_bins = 10;
	DATA_TYPE bin_range;
	DATA_TYPE max;
	DATA_TYPE min;
	DATA_TYPE amplitude;
	
	do_max(&max, window, wsize);
	do_min(&min, window, wsize);
	
	amplitude = max-min;
	bin_range = amplitude/num_bins;
	
	for(int i=0; i < wsize; i++) {
		DATA_TYPE value = window[i*NUM_DATA_SAMPLE];
		if(value > max-bin_range)
			bins[9]++;
		else if(value > max-2*bin_range)
			bins[8]++;
		else if(value > max-3*bin_range)
			bins[7]++;
		else if(value > max-4*bin_range)
			bins[6]++;
		else if(value > max-5*bin_range)
			bins[5]++;
		else if(value > max-6*bin_range)
			bins[4]++;
		else if(value > max-7*bin_range)
			bins[3]++;
		else if(value > max-8*bin_range)
			bins[2]++;
		else if(value > max-9*bin_range)
			bins[1]++;
		else 
			bins[0]++;
	}
}

void do_peaks(DATA_TYPE *distpeaks, DATA_TYPE *window, int wsize) {
	
	DATA_TYPE max, cutoff;
	int distance;
	int num_peaks;
	int index1, index2;
	
	#if DIMEM == 0
	DATA_TYPE peaks[wsize];
	for(int i=1; i < wsize-1; i++) {
		peaks[i] = (DATA_TYPE) 0.0;
	}
	#else
	DATA_TYPE *peaks = (DATA_TYPE *) calloc(wsize, sizeof(DATA_TYPE)) ;
	#endif
	
	do_max(&max, window, wsize);
	
	// find all peaks
	for(int i=1; i < wsize-1; i++) {
		if((window[i*NUM_DATA_SAMPLE] > window[(i+1)*NUM_DATA_SAMPLE]) && (window[i*NUM_DATA_SAMPLE] > window[(i-1)*NUM_DATA_SAMPLE])) {
			peaks[i] = window[i*NUM_DATA_SAMPLE];
		}
	}
	
	index1 = 0;
	index2 = 0;
	num_peaks = 0;
	DATA_TYPE factor = 0.9;

	// JMPC: to remove
	int iter=0;
	do { iter++; // JMPC: to remove
		
		cutoff = max * factor;
		num_peaks = 0;
		index1 = 0;
		for(int i=1; i<wsize-1; i++) {//foreach(p in peaks)
			if((peaks[i] != (DATA_TYPE) 0.0) && (peaks[i] >= cutoff)) {	
				num_peaks++;
				if(index1 == 0) index1 = i;
				index2 = i;
			}
		}
		factor -= (DATA_TYPE) 0.1;
	} while(factor > 0.0 && num_peaks < 3);
		
	distance = index2 - index1;
	if(distance == 0)
		*distpeaks = 0;
	else
		*distpeaks = ((DATA_TYPE) distance)/num_peaks;
	
	#if DIMEM != 0
	free(peaks);
	#endif
}

void do_resultant(DATA_TYPE *resultant, DATA_TYPE *windowX, DATA_TYPE *windowY, DATA_TYPE *windowZ, int wsize) {
	DATA_TYPE sum = (DATA_TYPE) 0.0;
	for(int i=0; i < wsize; i++) {
		sum += sqrt(windowX[i*NUM_DATA_SAMPLE]*windowX[i*NUM_DATA_SAMPLE]+windowY[i*NUM_DATA_SAMPLE]*windowY[i*NUM_DATA_SAMPLE]+windowZ[i*NUM_DATA_SAMPLE]*windowZ[i*NUM_DATA_SAMPLE]);
	}
	*resultant = sum / wsize;
}

// PAMAP2
#elif SCENARIO_FEATURES == 2


void do_meanXYZ(DATA_TYPE *mean, DATA_TYPE *windowX, DATA_TYPE *windowY, DATA_TYPE *windowZ, int size) {
	DATA_TYPE sum = (DATA_TYPE) 0.0;
	for(int i=0; i < size; i++) {
		sum += windowX[i*NUM_DATA_SAMPLE];
	}
	for(int i=0; i < size; i++) {
		sum += windowY[i*NUM_DATA_SAMPLE];
	}
	for(int i=0; i < size; i++) {
		sum += windowZ[i*NUM_DATA_SAMPLE];
	}
	*mean = sum / size;
}

void do_covariance(DATA_TYPE *covarianceXY, DATA_TYPE *windowX, DATA_TYPE *windowY, int wsize) {
	DATA_TYPE meanX;
	DATA_TYPE meanY;
	
	do_mean(&meanX, windowX, wsize);
	do_mean(&meanY, windowY, wsize);
	
	*covarianceXY = (DATA_TYPE) 0.0;
	for(int i=0; i < wsize; i++) {
		*covarianceXY += (windowX[i*NUM_DATA_SAMPLE]-meanX)*(windowY[i*NUM_DATA_SAMPLE]-meanY);
	}
	*covarianceXY = *covarianceXY / (wsize-1);
}

void do_corrxy(DATA_TYPE *corrxy, DATA_TYPE *windowX, DATA_TYPE *windowY, int wsize)  {
	DATA_TYPE covarianceXY;
	DATA_TYPE stdX;
	DATA_TYPE stdY;
	
	do_covariance(&covarianceXY, windowX, windowY, wsize);
	
	do_std(&stdX, windowX, wsize);
	do_std(&stdY, windowY, wsize);
	
	*corrxy =  covarianceXY / (stdX * stdY);
}

#endif


void do_features(Point *point, DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE], int wsize) {
	// Features
	DATA_TYPE meanX, meanY, meanZ;
	DATA_TYPE stddevX, stddevY, stddevZ;
	
	int id=0;
	
	// WISDM_Act_v1.1 dataset
	// the following 43 features
	#if SCENARIO_FEATURES == 1
	
		DATA_TYPE absoldevX, absoldevY, absoldevZ;
		DATA_TYPE peakX, peakY, peakZ;
		
		int bins10X[10];
		for(int i=0; i < 10; i++) {	
			bins10X[i] = 0;
		}
		int bins10Y[10];
		for(int i=0; i < 10; i++) {	
			bins10Y[i] = 0;
		}
		int bins10Z[10];
		for(int i=0; i < 10; i++) {	
			bins10Z[i] = 0;
		}
					
		do_mean(&meanX, &buffers[0][0], wsize);
		do_mean(&meanY, &buffers[0][1], wsize);
		do_mean(&meanZ, &buffers[0][2], wsize);
	
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
	
		do_bins10(bins10X, &buffers[0][0], wsize);
		do_bins10(bins10Y, &buffers[0][1], wsize);
		do_bins10(bins10Z, &buffers[0][2], wsize);
	
	 	point->features[id++] = bins10X[0];
	 	point->features[id++] = bins10X[1];
	 	point->features[id++] = bins10X[2];
	 	point->features[id++] = bins10X[3];
	 	point->features[id++] = bins10X[4];
	 	point->features[id++] = bins10X[5];
	 	point->features[id++] = bins10X[6];
	 	point->features[id++] = bins10X[7];
	 	point->features[id++] = bins10X[8];
	 	point->features[id++] = bins10X[9];
	
	 	point->features[id++] = bins10Y[0];
	 	point->features[id++] = bins10Y[1];
	 	point->features[id++] = bins10Y[2];
	 	point->features[id++] = bins10Y[3];
	 	point->features[id++] = bins10Y[4];
	 	point->features[id++] = bins10Y[5];
	 	point->features[id++] = bins10Y[6];
	 	point->features[id++] = bins10Y[7];
	 	point->features[id++] = bins10Y[8];
	 	point->features[id++] = bins10Y[9];
	
	 	point->features[id++] = bins10Z[0];
	 	point->features[id++] = bins10Z[1];
	 	point->features[id++] = bins10Z[2];
	 	point->features[id++] = bins10Z[3];
	 	point->features[id++] = bins10Z[4];
	 	point->features[id++] = bins10Z[5];
	 	point->features[id++] = bins10Z[6];
	 	point->features[id++] = bins10Z[7];
	 	point->features[id++] = bins10Z[8];
	 	point->features[id++] = bins10Z[9];
	
		do_peaks(&peakX, &buffers[0][0], wsize);
		do_peaks(&peakY, &buffers[0][1], wsize);
		do_peaks(&peakZ, &buffers[0][2], wsize);
		
	 	point->features[id++] = peakX;
	 	point->features[id++] = peakY;
	 	point->features[id++] = peakZ;
					
		do_absoldev(&absoldevX, &buffers[0][0], wsize);
		do_absoldev(&absoldevY, &buffers[0][1], wsize);
		do_absoldev(&absoldevZ, &buffers[0][2], wsize);
	
	 	point->features[id++] = absoldevX;
	 	point->features[id++] = absoldevY;
	 	point->features[id++] = absoldevZ;
	
		do_std(&stddevX, &buffers[0][0], wsize);
		do_std(&stddevY, &buffers[0][1], wsize);
		do_std(&stddevZ, &buffers[0][2], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
	
		DATA_TYPE resultant;
		do_resultant(&resultant, &buffers[0][0], &buffers[0][1], &buffers[0][2], wsize);
	 	point->features[id++] = resultant;
	
		// End features used by WISDM_Act_v1.1 dataset
	// PAMAP2
	#elif SCENARIO_FEATURES == 2
		DATA_TYPE corrxy, corryz, corrxz;
		
		// IMU: 3 x,y,z
		do_mean(&meanX, &buffers[0][0], wsize);
		do_mean(&meanY, &buffers[0][1], wsize);
		do_mean(&meanZ, &buffers[0][2], wsize);
	
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
			
		do_mean(&meanX, &buffers[0][3], wsize);
		do_mean(&meanY, &buffers[0][4], wsize);
		do_mean(&meanZ, &buffers[0][5], wsize);
		
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
		
		do_mean(&meanX, &buffers[0][6], wsize);
		do_mean(&meanY, &buffers[0][7], wsize);
		do_mean(&meanZ, &buffers[0][8], wsize);
		
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
		
		// IMU: 3 x,y,z
		do_mean(&meanX, &buffers[0][9], wsize);
		do_mean(&meanY, &buffers[0][10], wsize);
		do_mean(&meanZ, &buffers[0][11], wsize);
	
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
		
		do_mean(&meanX, &buffers[0][12], wsize);
		do_mean(&meanY, &buffers[0][13], wsize);
		do_mean(&meanZ, &buffers[0][14], wsize);
		
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
		
		do_mean(&meanX, &buffers[0][15], wsize);
		do_mean(&meanY, &buffers[0][16], wsize);
		do_mean(&meanZ, &buffers[0][17], wsize);
		
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ;
		
		// IMU: 3 x,y,z
		do_mean(&meanX, &buffers[0][18], wsize);
		do_mean(&meanY, &buffers[0][19], wsize);
		do_mean(&meanZ, &buffers[0][20], wsize);
	
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
		
		do_mean(&meanX, &buffers[0][21], wsize);
		do_mean(&meanY, &buffers[0][22], wsize);
		do_mean(&meanZ, &buffers[0][23], wsize);
		
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ; 
		
		do_mean(&meanX, &buffers[0][24], wsize);
		do_mean(&meanY, &buffers[0][25], wsize);
		do_mean(&meanZ, &buffers[0][26], wsize);
		
	 	point->features[id++] = meanX; 
	 	point->features[id++] = meanY; 
	 	point->features[id++] = meanZ;
		
		// IMU 
		DATA_TYPE meanXYZ_1, meanXYZ_2, meanXYZ_3, meanXYZ_4, meanXYZ_5, meanXYZ_6, meanXYZ_7, meanXYZ_8, meanXYZ_9;
		do_meanXYZ(&meanXYZ_1, &buffers[0][0], &buffers[0][1], &buffers[0][2], wsize);
		do_meanXYZ(&meanXYZ_2, &buffers[0][3], &buffers[0][4], &buffers[0][5], wsize);
		do_meanXYZ(&meanXYZ_3, &buffers[0][6], &buffers[0][7], &buffers[0][8], wsize);
		do_meanXYZ(&meanXYZ_4, &buffers[0][9], &buffers[0][10], &buffers[0][11], wsize);
		do_meanXYZ(&meanXYZ_5, &buffers[0][12], &buffers[0][13], &buffers[0][14], wsize);
		do_meanXYZ(&meanXYZ_6, &buffers[0][15], &buffers[0][16], &buffers[0][17], wsize);
		do_meanXYZ(&meanXYZ_7, &buffers[0][18], &buffers[0][19], &buffers[0][20], wsize);
		do_meanXYZ(&meanXYZ_8, &buffers[0][21], &buffers[0][22], &buffers[0][23], wsize);
		do_meanXYZ(&meanXYZ_9, &buffers[0][24], &buffers[0][25], &buffers[0][26], wsize);
		
	 	point->features[id++] = meanXYZ_1; 
	 	point->features[id++] = meanXYZ_2; 
	 	point->features[id++] = meanXYZ_3;
	 	point->features[id++] = meanXYZ_4; 
	 	point->features[id++] = meanXYZ_5; 
	 	point->features[id++] = meanXYZ_6;
	 	point->features[id++] = meanXYZ_7; 
	 	point->features[id++] = meanXYZ_8; 
	 	point->features[id++] = meanXYZ_9;
		
		// IMU, x,y,z
		do_std(&stddevX, &buffers[0][0], wsize);
		do_std(&stddevY, &buffers[0][1], wsize);
		do_std(&stddevZ, &buffers[0][2], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		do_std(&stddevX, &buffers[0][3], wsize);
		do_std(&stddevY, &buffers[0][4], wsize);
		do_std(&stddevZ, &buffers[0][5], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		do_std(&stddevX, &buffers[0][6], wsize);
		do_std(&stddevY, &buffers[0][7], wsize);
		do_std(&stddevZ, &buffers[0][8], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		// IMU, x,y,z
		do_std(&stddevX, &buffers[0][9], wsize);
		do_std(&stddevY, &buffers[0][10], wsize);
		do_std(&stddevZ, &buffers[0][11], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		do_std(&stddevX, &buffers[0][12], wsize);
		do_std(&stddevY, &buffers[0][13], wsize);
		do_std(&stddevZ, &buffers[0][14], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		
		do_std(&stddevX, &buffers[0][15], wsize);
		do_std(&stddevY, &buffers[0][16], wsize);
		do_std(&stddevZ, &buffers[0][17], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		// IMU, x,y,z
		do_std(&stddevX, &buffers[0][18], wsize);
		do_std(&stddevY, &buffers[0][19], wsize);
		do_std(&stddevZ, &buffers[0][20], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		do_std(&stddevX, &buffers[0][21], wsize);
		do_std(&stddevY, &buffers[0][22], wsize);
		do_std(&stddevZ, &buffers[0][23], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		do_std(&stddevX, &buffers[0][24], wsize);
		do_std(&stddevY, &buffers[0][25], wsize);
		do_std(&stddevZ, &buffers[0][26], wsize);
	
	 	point->features[id++] = stddevX;
	 	point->features[id++] = stddevY;
	 	point->features[id++] = stddevZ;
		
		// IMU, x,y,z
		do_corrxy(&corrxy, &buffers[0][0], &buffers[0][1], wsize);
		do_corrxy(&corrxz, &buffers[0][0], &buffers[0][2], wsize);
		do_corrxy(&corryz, &buffers[0][1], &buffers[0][2], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz; 
		
		do_corrxy(&corrxy, &buffers[0][3], &buffers[0][4], wsize);
		do_corrxy(&corrxz, &buffers[0][3], &buffers[0][5], wsize);
		do_corrxy(&corryz, &buffers[0][4], &buffers[0][5], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz;
		
		do_corrxy(&corrxy, &buffers[0][6], &buffers[0][7], wsize);
		do_corrxy(&corrxz, &buffers[0][6], &buffers[0][8], wsize);
		do_corrxy(&corryz, &buffers[0][7], &buffers[0][8], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz;
		
		// IMU, x,y,z
		do_corrxy(&corrxy, &buffers[0][9], &buffers[0][10], wsize);
		do_corrxy(&corrxz, &buffers[0][9], &buffers[0][11], wsize);
		do_corrxy(&corryz, &buffers[0][10], &buffers[0][11], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz; 
		
		do_corrxy(&corrxy, &buffers[0][12], &buffers[0][13], wsize);
		do_corrxy(&corrxz, &buffers[0][12], &buffers[0][14], wsize);
		do_corrxy(&corryz, &buffers[0][13], &buffers[0][14], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz;
		
		do_corrxy(&corrxy, &buffers[0][15], &buffers[0][16], wsize);
		do_corrxy(&corrxz, &buffers[0][15], &buffers[0][17], wsize);
		do_corrxy(&corryz, &buffers[0][16], &buffers[0][17], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz;
		
		// IMU, x,y,z
		do_corrxy(&corrxy, &buffers[0][18], &buffers[0][19], wsize);
		do_corrxy(&corrxz, &buffers[0][18], &buffers[0][20], wsize);
		do_corrxy(&corryz, &buffers[0][19], &buffers[0][20], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz; 
		
		do_corrxy(&corrxy, &buffers[0][21], &buffers[0][22], wsize);
		do_corrxy(&corrxz, &buffers[0][21], &buffers[0][23], wsize);
		do_corrxy(&corryz, &buffers[0][22], &buffers[0][23], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz;
		
		do_corrxy(&corrxy, &buffers[0][24], &buffers[0][25], wsize);
		do_corrxy(&corrxz, &buffers[0][24], &buffers[0][26], wsize);
		do_corrxy(&corryz, &buffers[0][25], &buffers[0][26], wsize);
		point->features[id++] = corrxy; 
	 	point->features[id++] = corrxz; 
	 	point->features[id++] = corryz;
		
	#endif
	//printf("***** number of features = %d\n", id);
}

void do_features_forall(Instance *instances, int num_sensor_data, int num_samples, Point *points,
			int num_classes, DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE], int wsize, int overlap) {
	
	int num_point = 0;
	CLASS_ID_TYPE classid;
	
	for (int i = 0; i < num_samples-wsize+1; i += wsize-overlap) {
		
		do_class(&classid, &instances[i], wsize, num_classes);
		
		// fill the buffers
		for(int j=0; j< wsize; j++) { // fill the buffers	
			for(int l=0; l < num_sensor_data; l++) { 
				buffers[j][l]=instances[i+j].data[l]; 
			}
		}	
		
		points[num_point].classification_id = classid;
		
		do_features(&points[num_point], buffers, wsize);
		
		num_point++;
	}
}

/*
* Fill the buffers
*/
void fill_buffers(Instance *instances, int num_sensor_data, DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE], int wsize) {
	for(int j=0; j< wsize; j++) { // fill the buffers	
		for(int l=0; l < num_sensor_data; l++) { 
			buffers[j][l]=instances[j].data[l]; 
		}
	}
}	


/*
* Determine the min and max values for each feature for a set of 
* points.
*/ 
void minmax(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *known_points, int num_features) {

	for (int j = 0; j < num_features; j++) {
		min[j] = MAX_FP_VAL;
		max[j] = MIN_FP_VAL;
		//printf("%e, %e\n", MIN_FP_VAL, MAX_FP_VAL);
	}
	
	for (int i = 0; i < num_points; i++) {
		for (int j = 0; j < num_features; j++) {
            if(known_points[i].features[j] < min[j]) 
				min[j] = known_points[i].features[j];
            if(known_points[i].features[j] > max[j]) 
				max[j] = known_points[i].features[j];
        }
    }
	
}

/*
* Normalize the features of each point using minmax normalization.
*/
void minmax_normalize(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *points, int num_features) {

    for (int i = 0; i < num_points; i++) {
        for (int j = 0; j < num_features; j++) {
			
			DATA_TYPE nfeature = (DATA_TYPE) ((points[i].features[j] - min[j])/(max[j] - min[j]));
			
			// in case the normalization returns a NaN or INF
			if(isnan(nfeature)) nfeature = (DATA_TYPE) 0.0;
			else if(isinf(nfeature)) nfeature = (DATA_TYPE) 1.0;
			
			points[i].features[j] = nfeature;
		}
    }
}

/*
* Normalize the features of a single point using minmax normalization.
*/
void minmax_normalize_point(DATA_TYPE *min, DATA_TYPE *max, Point *point, int num_features) {

    for (int j = 0; j < num_features; j++) {
			
		DATA_TYPE nfeature = (DATA_TYPE) ((point->features[j] - min[j])/(max[j] - min[j]));
			
		// in case the normalization returns a NaN or INF
		if(isnan(nfeature)) nfeature = (DATA_TYPE) 0.0;
		else if(isinf(nfeature)) nfeature = (DATA_TYPE) 1.0;
			
		point->features[j] = nfeature;
    }
}


void do_class(CLASS_ID_TYPE *classid, Instance *instances, int wsize, int num_classes) {

	#if DIMEM == 0
	K_TYPE histogram[num_classes];  // maximum equals the value of k;
	//initialize the histogram
    for (int i = 0; i < num_classes; i++) {
        histogram[i] = 0;
    }
    #else
	K_TYPE *histogram = (K_TYPE *) calloc(NUM_CLASSES, sizeof(CLASS_ID_TYPE)) ;
	#endif
	
	for(int j=0; j< wsize; j++) { // fill the buffers
		CLASS_ID_TYPE classid = instances[j].classification_id;
		histogram[(int) classid]++;
	}
	
	int max = 0;
	int index = 0;
	for(int j=0; j < num_classes; j++) {
		if(max < histogram[j]) {
			max = histogram[j];
			index = j;
		}
	}
	
	#if DIMEM != 0
	free(histogram);
	#endif
	
	*classid = index;
}	
