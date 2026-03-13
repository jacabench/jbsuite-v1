#include "backprop.h"

// --- UTILITY & ACTIVATION FUNCTIONS ---
void cross_entropy_softmax_bwd(size_t num_classes, const dtype_t* scores, const size_t* labels, dtype_t* dO_raw) {
 	dtype_t max_score = scores[0];
 	for (size_t k = 1; k < num_classes; ++k) { if (scores[k] > max_score) max_score = scores[k]; }
 	dtype_t exp_sum = 0.0f;
 	for (size_t k = 0; k < num_classes; ++k) { exp_sum += expf(scores[k] - max_score); }
 	for (size_t k = 0; k < num_classes; ++k) {
 	 	dO_raw[k] = expf(scores[k] - max_score) / exp_sum;
 	 	if (k == labels[0]) { dO_raw[k] -= 1.0f; }
 	}
}
void relu_fwd(size_t size, dtype_t* x) { for (size_t i = 0; i < size; ++i) { x[i] = (x[i] > 0.0f) ? x[i] : 0.0f; }}
void relu_bwd(size_t size, const dtype_t* O_store, const dtype_t* dO, dtype_t* dI) { for (size_t i = 0; i < size; ++i) { dI[i] = (O_store[i] > 0.0f) ? dO[i] : 0.0f; }}

// --- FULLY CONNECTED (FC) LAYER ---
void fc_layer_fwd(const dtype_t* input, const dtype_t* W, const dtype_t* B, dtype_t* output, size_t in_size, size_t out_size) {
 	for (size_t k = 0; k < out_size; k++) {
 	 	output[k] = B[k];
 	 	for (size_t i = 0; i < in_size; i++) { output[k] += input[i] * W[i + k * in_size]; }
 	}
}
void fc_layer_bwd(const dtype_t* input, const dtype_t* W, const dtype_t* dO, dtype_t* dI, dtype_t* dW, dtype_t* dB, size_t in_size, size_t out_size) {
 	for (size_t k = 0; k < out_size; ++k) { dB[k] += dO[k]; }
 	for (size_t k = 0; k < out_size; ++k) { for (size_t i = 0; i < in_size; ++i) { dW[i + k * in_size] += input[i] * dO[k]; }}
 	for (size_t i = 0; i < in_size; ++i) { dI[i] = 0.0f; }
 	for (size_t k = 0; k < out_size; ++k) { for (size_t i = 0; i < in_size; ++i) { dI[i] += W[i + k * in_size] * dO[k]; }}
}

// --- CONVOLUTIONAL (CONV) LAYER ---
void conv_1d_fwd(size_t H, size_t C, size_t R, size_t M, size_t S, size_t PAD, const dtype_t* I, const dtype_t* W, const dtype_t* B, dtype_t* O) {
 	size_t E = (H - R + 2 * PAD) / S + 1;
 	for(size_t m = 0; m < M; m++) { for(size_t x = 0; x < E; x++) {
 	 	O[x + m * E] = B[m];
 	 	for(size_t c = 0; c < C; c++) { for(size_t l = 0; l < R; l++) {
 	 	 	long h2 = (long)x * S - PAD + l;
 	 	 	if (h2 >= 0 && h2 < H) { O[x + m * E] += I[h2 + c * H] * W[l + c * R + m * C * R]; }
 	 	}}
 	}}
}
void conv_1d_bwd(size_t H, size_t C, size_t R, size_t M, size_t S, size_t PAD, const dtype_t* I, const dtype_t* dO, const dtype_t* W, dtype_t* dI, dtype_t* dW, dtype_t* dB) {
 	size_t E = (H - R + 2 * PAD) / S + 1;
 	for (size_t m = 0; m < M; ++m) { for (size_t c = 0; c < C; ++c) { for (size_t l = 0; l < R; ++l) {
 	 	dtype_t grad_w = 0.0f;
 	 	for (size_t x = 0; x < E; ++x) { long h2 = (long)x * S - PAD + l; if (h2 >= 0 && h2 < H) { grad_w += I[h2 + c * H] * dO[x + m * E]; } }
 	 	dW[l + c * R + m * C * R] += grad_w;
 	}}}
 	for (size_t m = 0; m < M; ++m) { dtype_t grad_b = 0.0f; for (size_t x = 0; x < E; ++x) { grad_b += dO[x + m * E]; } dB[m] += grad_b; }
 	for (size_t i = 0; i < C * H; ++i) { dI[i] = 0.0f; }
 	for (size_t m = 0; m < M; ++m) { for (size_t c = 0; c < C; ++c) { for (size_t l = 0; l < R; ++l) {
 	 	for (size_t x = 0; x < E; ++x) { long h2 = (long)x * S - PAD + l; if (h2 >= 0 && h2 < H) { dI[h2 + c * H] += W[l + c * R + m * C * R] * dO[x + m * E]; } }
 	}}}
}

// --- MAXPOOL LAYER ---
void maxpool_1d_fwd_with_indices(const dtype_t* I, dtype_t* O, int* max_indices, size_t M, size_t E_in, size_t E_out, size_t K, size_t S) {
 	for (size_t m = 0; m < M; ++m) {
 	 	for (size_t x_out = 0; x_out < E_out; ++x_out) {
 	 	 	dtype_t max_val = -3.4028235E+38f; // FLT_MIN
 	 	 	int max_idx = -1;
 	 	 	for (size_t k = 0; k < K; ++k) {
 	 	 	 	size_t h_in = x_out * S + k;
 	 	 	 	if (I[h_in + m * E_in] > max_val) {
 	 	 	 	 	max_val = I[h_in + m * E_in];
 	 	 	 	 	max_idx = h_in + m * E_in;
 	 	 	 	}
 	 	 	}
 	 	 	O[x_out + m * E_out] = max_val;
 	 	 	max_indices[x_out + m * E_out] = max_idx;
 	 	}
 	}
}
void maxpool_1d_bwd(const dtype_t* dO, const int* max_indices, dtype_t* dI, size_t in_fmap_size, size_t out_fmap_size) {
 	for(size_t i = 0; i < in_fmap_size; ++i) { dI[i] = 0.0f; }
 	for(size_t i = 0; i < out_fmap_size; ++i) {
 	 	int index = max_indices[i];
 	 	if(index != -1) { dI[index] += dO[i]; }
 	}
}

void cnn_fwd(const dtype_t I[CONV_FLAT_SIZE],
                const dtype_t W1[M1*C1*R1], const dtype_t B1[M1], dtype_t O1[L1_FMAP_SIZE],
                const dtype_t W2[M2*C2*R2], const dtype_t B2[M2], dtype_t O2[L2_FMAP_SIZE],
                const dtype_t W3[M3*C3*R3], const dtype_t B3[M3], dtype_t O3[L3_FMAP_SIZE],
                const dtype_t W4[M4*C4*R4], const dtype_t B4[M4], dtype_t O4[L4_FMAP_SIZE],
                const dtype_t W5[M5*C5*R5], const dtype_t B5[M5], dtype_t O5[L5_FMAP_SIZE], dtype_t O5_pool[L5_POOL_FMAP_SIZE],
                const dtype_t W_fc1[FC1_IN_SIZE*FC1_OUT_SIZE], const dtype_t B_fc1[FC1_OUT_SIZE], dtype_t O_fc1[FC1_OUT_SIZE],
                int max_indices5[L5_POOL_FMAP_SIZE]) {
	conv_1d_fwd(H1, C1, R1, M1, S1, PAD1, (const dtype_t*)I, W1, B1, O1);
	relu_fwd(L1_FMAP_SIZE, O1);
	conv_1d_fwd(H2, C2, R2, M2, S2, PAD2, (const dtype_t*)O1, W2, B2, O2);
	relu_fwd(L2_FMAP_SIZE, O2);
	conv_1d_fwd(H3, C3, R3, M3, S3, PAD3, (const dtype_t*)O2, W3, B3, O3);
	relu_fwd(L3_FMAP_SIZE, O3);
	conv_1d_fwd(H4, C4, R4, M4, S4, PAD4, (const dtype_t*)O3, W4, B4, O4);
	relu_fwd(L4_FMAP_SIZE, O4);
	conv_1d_fwd(H5, C5, R5, M5, S5, PAD5, (const dtype_t*)O4, W5, B5, O5);
	relu_fwd(L5_FMAP_SIZE, O5);
	maxpool_1d_fwd_with_indices(O5, O5_pool, max_indices5, M5, E5, L5_POOL_E, L5_POOL_K, L5_POOL_S);
	fc_layer_fwd(O5_pool, W_fc1, B_fc1, O_fc1, FC1_IN_SIZE, FC1_OUT_SIZE);
}

void cnn_bwd(const dtype_t I[CONV_FLAT_SIZE], const size_t LABEL[1],
             const dtype_t O1[L1_FMAP_SIZE],
             const dtype_t O2[L2_FMAP_SIZE],
             const dtype_t O3[L3_FMAP_SIZE],
             const dtype_t O4[L4_FMAP_SIZE],
             const dtype_t O5[L5_FMAP_SIZE], const dtype_t O5_pool[L5_POOL_FMAP_SIZE],
             const dtype_t O_fc1[FC1_OUT_SIZE],
             const dtype_t W1[M1*C1*R1], dtype_t dW1[M1*C1*R1], dtype_t dB1[M1],
             const dtype_t W2[M2*C2*R2], dtype_t dW2[M2*C2*R2], dtype_t dB2[M2],
             const dtype_t W3[M3*C3*R3], dtype_t dW3[M3*C3*R3], dtype_t dB3[M3],
             const dtype_t W4[M4*C4*R4], dtype_t dW4[M4*C4*R4], dtype_t dB4[M4],
             const dtype_t W5[M5*C5*R5], dtype_t dW5[M5*C5*R5], dtype_t dB5[M5],
             const dtype_t W_fc1[FC1_IN_SIZE*FC1_OUT_SIZE], dtype_t dW_fc1[FC1_IN_SIZE*FC1_OUT_SIZE], dtype_t dB_fc1[FC1_OUT_SIZE],
             dtype_t dI_fc1[CONV_FLAT_SIZE], dtype_t dI1[C1*H1*(GLOBAL_DIM==2?H1:1)], dtype_t dI5[L4_FMAP_SIZE], dtype_t dI4[L3_FMAP_SIZE], dtype_t dI3[L2_FMAP_SIZE], dtype_t dI2[L1_FMAP_SIZE],
             const int max_indices5[L5_POOL_FMAP_SIZE]) {
	dtype_t dO_raw[N_CLASSES];
	cross_entropy_softmax_bwd(N_CLASSES, O_fc1, LABEL, dO_raw);
	fc_layer_bwd(O5_pool, W_fc1, dO_raw, dI_fc1, dW_fc1, dB_fc1, FC1_IN_SIZE, FC1_OUT_SIZE);
	dtype_t dI5_pool[L5_FMAP_SIZE];
	maxpool_1d_bwd(dI_fc1, max_indices5, dI5_pool, L5_FMAP_SIZE, L5_POOL_FMAP_SIZE);
	dtype_t dI5_relu[L5_FMAP_SIZE];
	relu_bwd(L5_FMAP_SIZE, O5, dI5_pool, dI5_relu);
	conv_1d_bwd(H5, C5, R5, M5, S5, PAD5, O4, dI5_relu, W5, dI5, dW5, dB5);
	dtype_t dI4_relu[L4_FMAP_SIZE];
	relu_bwd(L4_FMAP_SIZE, O4, dI5, dI4_relu);
	conv_1d_bwd(H4, C4, R4, M4, S4, PAD4, O3, dI4_relu, W4, dI4, dW4, dB4);
	dtype_t dI3_relu[L3_FMAP_SIZE];
	relu_bwd(L3_FMAP_SIZE, O3, dI4, dI3_relu);
	conv_1d_bwd(H3, C3, R3, M3, S3, PAD3, O2, dI3_relu, W3, dI3, dW3, dB3);
	dtype_t dI2_relu[L2_FMAP_SIZE];
	relu_bwd(L2_FMAP_SIZE, O2, dI3, dI2_relu);
	conv_1d_bwd(H2, C2, R2, M2, S2, PAD2, O1, dI2_relu, W2, dI2, dW2, dB2);
	dtype_t dI1_relu[L1_FMAP_SIZE];
	relu_bwd(L1_FMAP_SIZE, O1, dI2, dI1_relu);
	conv_1d_bwd(H1, C1, R1, M1, S1, PAD1, (const dtype_t*)I, dI1_relu, W1, dI1, dW1, dB1);
}