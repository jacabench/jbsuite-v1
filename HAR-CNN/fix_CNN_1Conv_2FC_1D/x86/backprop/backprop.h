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
const size_t M1 = 16; const size_t E1 = 10;
const size_t R1 = 3; const size_t S1 = 1; const size_t PAD1 = 1;
const size_t L1_FMAP_SIZE = M1 * E1;

// --- MAXPOOL CONSTANTS ---
const size_t L1_POOL_K = 2;
const size_t L1_POOL_S = 2;
const size_t L1_POOL_E = 5;
const size_t L1_POOL_FMAP_SIZE = M1 * 5;

// --- FLATTEN & FC CONSTANTS ---
const size_t CONV_FLAT_SIZE = 80;
const size_t FC1_IN_SIZE = 80;
const size_t FC1_OUT_SIZE = 64;

const size_t FC2_IN_SIZE = 64;
const size_t FC2_OUT_SIZE = 6;

// --- MAIN FUNCTION PROTOTYPES ---
void cnn_fwd(const dtype_t I[CONV_FLAT_SIZE],
                const dtype_t W1[M1*C1*R1], const dtype_t B1[M1], dtype_t O1[L1_FMAP_SIZE], dtype_t O1_pool[L1_POOL_FMAP_SIZE],
                const dtype_t W_fc1[FC1_IN_SIZE*FC1_OUT_SIZE], const dtype_t B_fc1[FC1_OUT_SIZE], dtype_t O_fc1[FC1_OUT_SIZE],
                const dtype_t W_fc2[FC2_IN_SIZE*FC2_OUT_SIZE], const dtype_t B_fc2[FC2_OUT_SIZE], dtype_t O_fc2[FC2_OUT_SIZE],
                int max_indices1[L1_POOL_FMAP_SIZE]);

void cnn_bwd(const dtype_t I[CONV_FLAT_SIZE], const size_t LABEL[1],
                const dtype_t O1[L1_FMAP_SIZE], const dtype_t O1_pool[L1_POOL_FMAP_SIZE],
                const dtype_t O_fc1[FC1_OUT_SIZE],
                const dtype_t O_fc2[FC2_OUT_SIZE],
                const dtype_t W1[M1*C1*R1], dtype_t dW1[M1*C1*R1], dtype_t dB1[M1],
                const dtype_t W_fc1[FC1_IN_SIZE*FC1_OUT_SIZE], dtype_t dW_fc1[FC1_IN_SIZE*FC1_OUT_SIZE], dtype_t dB_fc1[FC1_OUT_SIZE],
                const dtype_t W_fc2[FC2_IN_SIZE*FC2_OUT_SIZE], dtype_t dW_fc2[FC2_IN_SIZE*FC2_OUT_SIZE], dtype_t dB_fc2[FC2_OUT_SIZE],
                dtype_t dI_fc1[CONV_FLAT_SIZE], dtype_t dI1[C1*H1*(GLOBAL_DIM==2?H1:1)], dtype_t dI_fc2[FC1_OUT_SIZE],
                const int max_indices1[L1_POOL_FMAP_SIZE]);


// --- HELPER FUNCTION PROTOTYPES ---
void adam_update(dtype_t* W, const dtype_t* dW, dtype_t* m, dtype_t* v, size_t size, dtype_t lr, dtype_t beta1, dtype_t beta2, dtype_t epsilon, size_t t);
void cross_entropy_softmax_bwd(size_t num_classes, const dtype_t* scores, const size_t* labels, dtype_t* dO_raw);
void relu_bwd(size_t size, const dtype_t* O_store, const dtype_t* dO, dtype_t* dI);
void maxpool_1d_bwd(const dtype_t* dO, const int* max_indices, dtype_t* dI, size_t in_fmap_size, size_t out_fmap_size);

#endif // BACKPROP_H