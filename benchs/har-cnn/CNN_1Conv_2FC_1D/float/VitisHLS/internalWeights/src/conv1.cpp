
#include <math.h>
#include <stdio.h>

#include "weights.h"


// --- ACTIVATION & OUTPUT FUNCTIONS ---
type_t relu(type_t x) {
    #pragma HLS INLINE
 	return (x > 0) ? x : (type_t)0;
}

void softmax(type_t input[N_CLASSES], float output[N_CLASSES]) {
 	float sum = 0.0f;
 	float max_val = (float)input[0];
 	for (int k = 1; k < N_CLASSES; k++) {
 		if ((float)input[k] > max_val) max_val = (float)input[k];
 	}
 	for (int k = 0; k < N_CLASSES; k++) {
 		output[k] = expf((float)input[k] - max_val);
 		sum += output[k];
 	}
 	for (int k = 0; k < N_CLASSES; k++) {
 		output[k] /= sum;
 	}
}


#if GLOBAL_DIM == 1
// ** Specialized 1D Convolution Layer 1 **
void conv_1d_1(type_t I[C1 * H1], type_t const W[M1 * C1 * R1], const type_t B[M1], type_t O[M1 * E1]) {
 	for(int m = 0; m < M1; m++) {
 		for(int x = 0; x < E1; x++) {
 			O[x + m * E1] = B[m];
 			#pragma HLS PIPELINE II=1
 			for(int c = 0; c < C1; c++) {
 				for(int l = 0; l < R1; l++) {
 					int h2 = x * S1 - PAD1 + l;
 					type_t val = (h2 < 0 || h2 >= H1) ? (type_t)0 : I[h2 + c * H1];
 					O[x + m * E1] += val * W[l + c * R1 + m * C1 * R1];
 				}
 			}
 		}
 	}
}
#endif // GLOBAL_DIM == 1


#if GLOBAL_DIM == 2
// ** Specialized 2D Convolution Layer 1 **
void conv_2d_1(type_t I[C1 * H1 * H1], type_t W[M1 * C1 * R1 * R1], type_t B[M1], type_t O[M1 * E1 * E1]) {
 	for(int m = 0; m < M1; m++) {
 		for(int y = 0; y < E1; y++) {
 			for(int x = 0; x < E1; x++) {
 				O[x + (y + (m * E1)) * E1] = B[m];
 				#pragma HLS PIPELINE II=1
 				for(int c = 0; c < C1; c++) {
 					for(int k = 0; k < R1; k++) {
 						for(int l = 0; l < R1; l++) {
 							int h1 = y * S1 - PAD1 + k;
 							int h2 = x * S1 - PAD1 + l;
 							type_t val = (h1 < 0 || h1 >= H1 || h2 < 0 || h2 >= H1) ? (type_t)0 : I[h2 + (h1 + (c * H1)) * H1];
 							O[x + (y + (m * E1)) * E1] += val * W[l + (k + (c + (m * C1)) * R1) * R1];
 						}
 					}
 				}
 			}
 		}
 	}
}
#endif // GLOBAL_DIM == 2

// ** Specialized 1D MaxPool for Layer 1 **
void maxpool_1d_1(type_t I[M1*E1], type_t O[M1*L1_POOL_E]) {
 	for (int m = 0; m < M1; ++m) {
 		for (int x_out = 0; x_out < L1_POOL_E; ++x_out) {
 			type_t max_val = -32768;
 			#pragma HLS PIPELINE II=1
 			for (int k = 0; k < L1_POOL_K; ++k) {
 				int h_in = x_out * L1_POOL_S + k;
 				if (I[h_in + m * E1] > max_val) {
 					max_val = I[h_in + m * E1];
 				}
 			}
 			O[x_out + m * L1_POOL_E] = max_val;
 		}
 	}
}

// ** Specialized Fully Connected Layer 1 **
void fc_layer_1(type_t input[FC1_IN_SIZE], const type_t W[FC1_IN_SIZE * FC1_OUT_SIZE], const type_t B[FC1_OUT_SIZE], type_t output[FC1_OUT_SIZE]) {
 	for (int k = 0; k < FC1_OUT_SIZE; k++) {
 		output[k] = B[k];
 		for (int j = 0; j < FC1_IN_SIZE; j++) {
 			#pragma HLS PIPELINE II=1
 			output[k] += input[j] * W[j + k * FC1_IN_SIZE];
 		}
 	}
}

// ** Specialized Fully Connected Layer 2 **
void fc_layer_2(type_t input[FC2_IN_SIZE], const type_t W[FC2_IN_SIZE * FC2_OUT_SIZE], const type_t B[FC2_OUT_SIZE], type_t output[FC2_OUT_SIZE]) {
 	for (int k = 0; k < FC2_OUT_SIZE; k++) {
 		output[k] = B[k];
 		for (int j = 0; j < FC2_IN_SIZE; j++) {
 			#pragma HLS PIPELINE II=1
 			output[k] += input[j] * W[j + k * FC2_IN_SIZE];
 		}
 	}
}

// ** Wrapper CNN Function (HLS Inference) **
void cnn(type_t *input, float *output) {
//#pragma HLS DATAFLOW

 	static type_t O1_conv_relu[M1 * E1];
 	static type_t O1_pool[M1 * L1_POOL_E];
 	static type_t O_conv_flat[CONV_FLAT_SIZE];
 	static type_t O_fc1_raw[FC1_OUT_SIZE];
 	static type_t O_fc1_relu[FC1_OUT_SIZE];
 	static type_t O_fc2_raw[FC2_OUT_SIZE];


 	// --- Stage 1: CONV -> ReLU -> Optional-MaxPool ---
 #if GLOBAL_DIM == 1
 	conv_1d_1((type_t*)input, W1, B1, O1_conv_relu);
 #elif GLOBAL_DIM == 2
 	conv_2d_1((type_t*)input, W1, B1, O1_conv_relu);
 #endif
	for (int j = 0; j < 160; j++) O1_conv_relu[j] = relu(O1_conv_relu[j]);
 #if GLOBAL_DIM == 1
 	maxpool_1d_1(O1_conv_relu, O1_pool);
 #elif GLOBAL_DIM == 2
 	maxpool_2d_1(O1_conv_relu, O1_pool);
 #endif

 	// --- Flatten Stage ---
	for (int j = 0; j < CONV_FLAT_SIZE; j++) O_conv_flat[j] = ((type_t*)O1_pool)[j];

 	// --- FC Layer 1 (Activation: ReLU) ---
 	fc_layer_1(O_conv_flat, W_fc1, B_fc1, O_fc1_raw);
	for (int j = 0; j < FC1_OUT_SIZE; j++) O_fc1_relu[j] = relu(O_fc1_raw[j]);

 	// --- FC Layer 2 (Activation: None) ---
 	fc_layer_2(O_fc1_relu, W_fc2, B_fc2, O_fc2_raw);

 	// --- Final Layer: Softmax ---
 	softmax(O_fc2_raw, output);
}