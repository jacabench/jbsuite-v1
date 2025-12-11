#ifndef BACKPROP_H
#define BACKPROP_H

#include <cstddef>
#include <cmath>
#include <iostream>

typedef float dtype_t;

// --- GLOBAL & CONV CONSTANTS ---
const size_t N_CLASSES = 6;
const size_t GLOBAL_DIM = 1;

// Layer 1
const size_t C1 = 3; const size_t H1 = 10;
const size_t M1 = 6; const size_t E1 = 10;
const size_t R1 = 3; const size_t S1 = 1; const size_t PAD1 = 1;
const size_t L1_FMAP_SIZE = M1 * E1;

// Layer 2
const size_t C2 = 6; const size_t H2 = 10;
const size_t M2 = 12; const size_t E2 = 10;
const size_t R2 = 3; const size_t S2 = 1; const size_t PAD2 = 1;
const size_t L2_FMAP_SIZE = M2 * E2;

// Layer 3
const size_t C3 = 12; const size_t H3 = 10;
const size_t M3 = 16; const size_t E3 = 10;
const size_t R3 = 3; const size_t S3 = 1; const size_t PAD3 = 1;
const size_t L3_FMAP_SIZE = M3 * E3;

// Layer 4
const size_t C4 = 16; const size_t H4 = 10;
const size_t M4 = 24; const size_t E4 = 10;
const size_t R4 = 3; const size_t S4 = 1; const size_t PAD4 = 1;
const size_t L4_FMAP_SIZE = M4 * E4;

// Layer 5
const size_t C5 = 24; const size_t H5 = 10;
const size_t M5 = 32; const size_t E5 = 10;
const size_t R5 = 3; const size_t S5 = 1; const size_t PAD5 = 1;
const size_t L5_FMAP_SIZE = M5 * E5;

// --- MAXPOOL CONSTANTS ---
const size_t L5_POOL_K = 2;
const size_t L5_POOL_S = 2;
const size_t L5_POOL_E = 5;
const size_t L5_POOL_FMAP_SIZE = M5 * 5;

// --- FLATTEN & FC CONSTANTS ---
const size_t CONV_FLAT_SIZE = 160;
const size_t FC1_IN_SIZE = 160;
const size_t FC1_OUT_SIZE = 6;

// --- MAIN FUNCTION PROTOTYPES ---
void cnn_fwd(const dtype_t I[CONV_FLAT_SIZE],
                const dtype_t W1[M1*C1*R1], const dtype_t B1[M1], dtype_t O1[L1_FMAP_SIZE],
                const dtype_t W2[M2*C2*R2], const dtype_t B2[M2], dtype_t O2[L2_FMAP_SIZE],
                const dtype_t W3[M3*C3*R3], const dtype_t B3[M3], dtype_t O3[L3_FMAP_SIZE],
                const dtype_t W4[M4*C4*R4], const dtype_t B4[M4], dtype_t O4[L4_FMAP_SIZE],
                const dtype_t W5[M5*C5*R5], const dtype_t B5[M5], dtype_t O5[L5_FMAP_SIZE], dtype_t O5_pool[L5_POOL_FMAP_SIZE],
                const dtype_t W_fc1[FC1_IN_SIZE*FC1_OUT_SIZE], const dtype_t B_fc1[FC1_OUT_SIZE], dtype_t O_fc1[FC1_OUT_SIZE],
                int max_indices5[L5_POOL_FMAP_SIZE]);

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
                const int max_indices5[L5_POOL_FMAP_SIZE]);


// --- HELPER FUNCTION PROTOTYPES ---
void adam_update(dtype_t* W, const dtype_t* dW, dtype_t* m, dtype_t* v, size_t size, dtype_t lr, dtype_t beta1, dtype_t beta2, dtype_t epsilon, size_t t);
void cross_entropy_softmax_bwd(size_t num_classes, const dtype_t* scores, const size_t* labels, dtype_t* dO_raw);
void relu_bwd(size_t size, const dtype_t* O_store, const dtype_t* dO, dtype_t* dI);
void maxpool_1d_bwd(const dtype_t* dO, const int* max_indices, dtype_t* dI, size_t in_fmap_size, size_t out_fmap_size);

#endif // BACKPROP_H