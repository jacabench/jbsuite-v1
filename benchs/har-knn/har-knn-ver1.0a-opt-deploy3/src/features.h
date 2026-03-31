/**
* Functions for extracting the features from sensing data.
* SPeCS Group 
* Faculty of Engineering of the University of Porto (FEUP)
* Porto, Portugal
* December 2024
*/

#ifndef HAR_FEATURES_H 
#define HAR_FEATURES_H

#define MAX_ACCEL_VAL 20
#define MIN_ACCEL_VAL (-20)

#ifndef INT_MIN
#define INT_MIN -2147483648
#endif

#ifndef INT_MAX
#define INT_MAX 2147483647
#endif

#define sqr(x) (x) * (x)

// for WISDM and PAMAP2
void do_mean(DATA_TYPE *mean, DATA_TYPE *window, int wsize);

void do_variance(DATA_TYPE *variance, DATA_TYPE *window, int wsize);

void do_std(DATA_TYPE *std, DATA_TYPE *window, int wsize);

// WISDM
#if SCENARIO_FEATURES == 1

//XABSOLDEV, YABSOLDEV, ZABSOLDEV are the average absolute 
//	deviations from the mean value for each axis.
void do_absoldev(DATA_TYPE *absoldev, DATA_TYPE *window, int wsize);

void do_max(DATA_TYPE *max, DATA_TYPE *window, int wsize);

void do_min(DATA_TYPE *min, DATA_TYPE *window, int wsize);

//X0..X9, Y0..Y9, Z0..Z9 are bins, their values are the fraction
//	of accelerometer samples that fell within that bin
void do_bins10(int *bins, DATA_TYPE *window, int wsize);

/*XPEAK, YPEAK, ZPEAK are approximations of the dominant 
	frequency. First, the greatest value in the series is 
	identified, then all local peak values within 10% of
	its amplitude are identified. If the number of peaks 
	is less than 3, then the threshhold is lowered until 
	at least 3 peaks can be found. The times between 
	consecutive peaks are summed and divided by the number
	of peaks.*/
void do_peaks(DATA_TYPE *distpeaks, DATA_TYPE *window, int wsize);

void do_resultant(DATA_TYPE *resultant, DATA_TYPE *windowX, DATA_TYPE *windowY, DATA_TYPE *windowZ, int wsize);

// PAMAP2
#elif SCENARIO_FEATURES == 2
void do_meanXYZ(DATA_TYPE *mean, DATA_TYPE *windowX, DATA_TYPE *windowY, DATA_TYPE *windowZ, int size);

void do_covariance(DATA_TYPE *covarianceXY, DATA_TYPE *windowX, DATA_TYPE *windowY, int wsize);

void do_corrxy(DATA_TYPE *corrxy, DATA_TYPE *windowX, DATA_TYPE *windowY, int wsize);

#endif

void do_features(Point *point, DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE], int wsize);

void do_features_forall(Instance *instances, int num_sensor_data, int num_samples, Point *points, 
				int num_classes, DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE], int wsize, int overlap);

void fill_buffers(Instance *instances, int num_sensor_data, DATA_TYPE buffers[WSIZE][NUM_DATA_SAMPLE], int wsize);

void minmax(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *known_points, 
			int num_features);
								
void minmax_normalize_point(DATA_TYPE *min, DATA_TYPE *max, Point *point, int num_features);

void minmax_normalize(DATA_TYPE *min, DATA_TYPE *max, int num_points, Point *points, 
					int num_features);

void do_class(CLASS_ID_TYPE *classid, Instance *instances, int wsize, int num_classes);

#endif //HAR_FEATURES_H