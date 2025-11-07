#ifndef CONV_H
#define CONV_H

#include <cstddef>
#include <stdio.h>
#include "ap_fixed.h"
// Using fixed-point for HLS synthesis
typedef ap_fixed<32, 16> type_t;
// ** GLOBAL PARAMETERS **
const size_t N_CLASSES = 6;
#define GLOBAL_DIM 1

// ** CONV LAYER 1 **
const size_t C1 = 3;
const size_t H1 = 10;
const size_t M1 = 16;
const size_t R1 = 3;
const size_t S1 = 1;
const size_t E1 = 10;
const size_t PAD1 = 1;

// ** MAXPOOL for LAYER 1 **
const size_t L1_POOL_K = 2;
const size_t L1_POOL_S = 2;
const size_t L1_POOL_E = 5;

// ** FLATTEN & FC SIZES **
const size_t CONV_FLAT_SIZE = 80;

// ** FC LAYER 1 **
const size_t FC1_IN_SIZE = 80;
const size_t FC1_OUT_SIZE = 64;

// ** FC LAYER 2 **
const size_t FC2_IN_SIZE = 64;
const size_t FC2_OUT_SIZE = 6;

// ** FUNCTION PROTOTYPES **
void conv_1d_1(type_t I[C1*H1], type_t W[M1*C1*R1], type_t B[M1], type_t O[M1*E1]);
void conv_2d_1(type_t I[C1*H1*H1], type_t W[M1*C1*R1*R1], type_t B[M1], type_t O[M1*E1*E1]);
void maxpool_1d_1(type_t I[M1*E1], type_t O[M1*L1_POOL_E]);
void maxpool_2d_1(type_t I[M1*E1*E1], type_t O[M1*L1_POOL_E*L1_POOL_E]);
void fc_layer_1(type_t input[FC1_IN_SIZE], type_t W[FC1_IN_SIZE * FC1_OUT_SIZE], type_t B[FC1_OUT_SIZE], type_t output[FC1_OUT_SIZE]);
void fc_layer_2(type_t input[FC2_IN_SIZE], type_t W[FC2_IN_SIZE * FC2_OUT_SIZE], type_t B[FC2_OUT_SIZE], type_t output[FC2_OUT_SIZE]);
void softmax(type_t input[N_CLASSES], float output[N_CLASSES]);

void cnn(type_t *input, type_t *W1, type_t *B1, type_t *W_fc1, type_t *B_fc1, type_t *W_fc2, type_t *B_fc2, float *output);

#endif // CONV_H