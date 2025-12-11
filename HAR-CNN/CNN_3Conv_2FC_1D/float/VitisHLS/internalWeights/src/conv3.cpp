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
void conv_1d_1(type_t I[C1 * H1], type_t const W[M1 * C1 * R1], type_t const B[M1], type_t O[M1 * E1]) {
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


#if GLOBAL_DIM == 1
// ** Specialized 1D Convolution Layer 2 **
void conv_1d_2(type_t I[C2 * H2], type_t const W[M2 * C2 * R2], type_t const B[M2], type_t O[M2 * E2]) {
 	for(int m = 0; m < M2; m++) {
 		for(int x = 0; x < E2; x++) {
 			O[x + m * E2] = B[m];
 			#pragma HLS PIPELINE II=1
 			for(int c = 0; c < C2; c++) {
 				for(int l = 0; l < R2; l++) {
 					int h2 = x * S2 - PAD2 + l;
 					type_t val = (h2 < 0 || h2 >= H2) ? (type_t)0 : I[h2 + c * H2];
 					O[x + m * E2] += val * W[l + c * R2 + m * C2 * R2];
 				}
 			}
 		}
 	}
}
#endif // GLOBAL_DIM == 1


#if GLOBAL_DIM == 2
// ** Specialized 2D Convolution Layer 2 **
void conv_2d_2(type_t I[C2 * H2 * H2], type_t W[M2 * C2 * R2 * R2], type_t B[M2], type_t O[M2 * E2 * E2]) {
 	for(int m = 0; m < M2; m++) {
 		for(int y = 0; y < E2; y++) {
 			for(int x = 0; x < E2; x++) {
 				O[x + (y + (m * E2)) * E2] = B[m];
 				#pragma HLS PIPELINE II=1
 				for(int c = 0; c < C2; c++) {
 					for(int k = 0; k < R2; k++) {
 						for(int l = 0; l < R2; l++) {
 							int h1 = y * S2 - PAD2 + k;
 							int h2 = x * S2 - PAD2 + l;
 							type_t val = (h1 < 0 || h1 >= H2 || h2 < 0 || h2 >= H2) ? (type_t)0 : I[h2 + (h1 + (c * H2)) * H2];
 							O[x + (y + (m * E2)) * E2] += val * W[l + (k + (c + (m * C2)) * R2) * R2];
 						}
 					}
 				}
 			}
 		}
 	}
}
#endif // GLOBAL_DIM == 2


#if GLOBAL_DIM == 1
// ** Specialized 1D Convolution Layer 3 **
void conv_1d_3(type_t I[C3 * H3], type_t const W[M3 * C3 * R3], type_t const B[M3], type_t O[M3 * E3]) {
 	for(int m = 0; m < M3; m++) {
 		for(int x = 0; x < E3; x++) {
 			O[x + m * E3] = B[m];
 			#pragma HLS PIPELINE II=1
 			for(int c = 0; c < C3; c++) {
 				for(int l = 0; l < R3; l++) {
 					int h2 = x * S3 - PAD3 + l;
 					type_t val = (h2 < 0 || h2 >= H3) ? (type_t)0 : I[h2 + c * H3];
 					O[x + m * E3] += val * W[l + c * R3 + m * C3 * R3];
 				}
 			}
 		}
 	}
}
#endif // GLOBAL_DIM == 1


#if GLOBAL_DIM == 2
// ** Specialized 2D Convolution Layer 3 **
void conv_2d_3(type_t I[C3 * H3 * H3], type_t W[M3 * C3 * R3 * R3], type_t B[M3], type_t O[M3 * E3 * E3]) {
 	for(int m = 0; m < M3; m++) {
 		for(int y = 0; y < E3; y++) {
 			for(int x = 0; x < E3; x++) {
 				O[x + (y + (m * E3)) * E3] = B[m];
 				#pragma HLS PIPELINE II=1
 				for(int c = 0; c < C3; c++) {
 					for(int k = 0; k < R3; k++) {
 						for(int l = 0; l < R3; l++) {
 							int h1 = y * S3 - PAD3 + k;
 							int h2 = x * S3 - PAD3 + l;
 							type_t val = (h1 < 0 || h1 >= H3 || h2 < 0 || h2 >= H3) ? (type_t)0 : I[h2 + (h1 + (c * H3)) * H3];
 							O[x + (y + (m * E3)) * E3] += val * W[l + (k + (c + (m * C3)) * R3) * R3];
 						}
 					}
 				}
 			}
 		}
 	}
}
#endif // GLOBAL_DIM == 2

// ** Specialized 1D MaxPool for Layer 3 **
void maxpool_1d_3(type_t I[M3*E3], type_t O[M3*L3_POOL_E]) {
 	for (int m = 0; m < M3; ++m) {
 		for (int x_out = 0; x_out < L3_POOL_E; ++x_out) {
 			type_t max_val = -32768;
 			#pragma HLS PIPELINE II=1
 			for (int k = 0; k < L3_POOL_K; ++k) {
 				int h_in = x_out * L3_POOL_S + k;
 				if (I[h_in + m * E3] > max_val) {
 					max_val = I[h_in + m * E3];
 				}
 			}
 			O[x_out + m * L3_POOL_E] = max_val;
 		}
 	}
}

// ** Specialized Fully Connected Layer 1 **
void fc_layer_1(type_t input[FC1_IN_SIZE], type_t const W[FC1_IN_SIZE * FC1_OUT_SIZE], type_t const B[FC1_OUT_SIZE], type_t output[FC1_OUT_SIZE]) {
 	for (int k = 0; k < FC1_OUT_SIZE; k++) {
 		output[k] = B[k];
 		for (int j = 0; j < FC1_IN_SIZE; j++) {
 			#pragma HLS PIPELINE II=1
 			output[k] += input[j] * W[j + k * FC1_IN_SIZE];
 		}
 	}
}

// ** Specialized Fully Connected Layer 2 **
void fc_layer_2(type_t input[FC2_IN_SIZE], type_t const W[FC2_IN_SIZE * FC2_OUT_SIZE], type_t const B[FC2_OUT_SIZE], type_t output[FC2_OUT_SIZE]) {
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
 	static type_t O2_conv_relu[M2 * E2];
 	static type_t O3_conv_relu[M3 * E3];
 	static type_t O3_pool[M3 * L3_POOL_E];
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
	for (int j = 0; j < 60; j++) O1_conv_relu[j] = relu(O1_conv_relu[j]);

 	// --- Stage 2: CONV -> ReLU -> Optional-MaxPool ---
 #if GLOBAL_DIM == 1
 	conv_1d_2((type_t*)O1_conv_relu, W2, B2, O2_conv_relu);
 #elif GLOBAL_DIM == 2
 	conv_2d_2((type_t*)O1_conv_relu, W2, B2, O2_conv_relu);
 #endif
	for (int j = 0; j < 120; j++) O2_conv_relu[j] = relu(O2_conv_relu[j]);

 	// --- Stage 3: CONV -> ReLU -> Optional-MaxPool ---
 #if GLOBAL_DIM == 1
 	conv_1d_3((type_t*)O2_conv_relu, W3, B3, O3_conv_relu);
 #elif GLOBAL_DIM == 2
 	conv_2d_3((type_t*)O2_conv_relu, W3, B3, O3_conv_relu);
 #endif
	for (int j = 0; j < 160; j++) O3_conv_relu[j] = relu(O3_conv_relu[j]);
 #if GLOBAL_DIM == 1
 	maxpool_1d_3(O3_conv_relu, O3_pool);
 #elif GLOBAL_DIM == 2
 	maxpool_2d_3(O3_conv_relu, O3_pool);
 #endif

 	// --- Flatten Stage ---
	for (int j = 0; j < CONV_FLAT_SIZE; j++) O_conv_flat[j] = ((type_t*)O3_pool)[j];

 	// --- FC Layer 1 (Activation: ReLU) ---
 	fc_layer_1(O_conv_flat, W_fc1, B_fc1, O_fc1_raw);
	for (int j = 0; j < FC1_OUT_SIZE; j++) O_fc1_relu[j] = relu(O_fc1_raw[j]);

 	// --- FC Layer 2 (Activation: None) ---
 	fc_layer_2(O_fc1_relu, W_fc2, B_fc2, O_fc2_raw);

 	// --- Final Layer: Softmax ---
 	softmax(O_fc2_raw, output);
}