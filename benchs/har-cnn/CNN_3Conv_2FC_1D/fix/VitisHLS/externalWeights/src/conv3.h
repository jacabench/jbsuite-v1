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
const size_t M1 = 6;
const size_t R1 = 3;
const size_t S1 = 1;
const size_t E1 = 10;
const size_t PAD1 = 1;

// ** CONV LAYER 2 **
const size_t C2 = 6;
const size_t H2 = 10;
const size_t M2 = 12;
const size_t R2 = 3;
const size_t S2 = 1;
const size_t E2 = 10;
const size_t PAD2 = 1;

// ** CONV LAYER 3 **
const size_t C3 = 12;
const size_t H3 = 10;
const size_t M3 = 16;
const size_t R3 = 3;
const size_t S3 = 1;
const size_t E3 = 10;
const size_t PAD3 = 1;

// ** MAXPOOL for LAYER 3 **
const size_t L3_POOL_K = 2;
const size_t L3_POOL_S = 2;
const size_t L3_POOL_E = 5;

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
void conv_1d_2(type_t I[C2*H2], type_t W[M2*C2*R2], type_t B[M2], type_t O[M2*E2]);
void conv_2d_2(type_t I[C2*H2*H2], type_t W[M2*C2*R2*R2], type_t B[M2], type_t O[M2*E2*E2]);
void conv_1d_3(type_t I[C3*H3], type_t W[M3*C3*R3], type_t B[M3], type_t O[M3*E3]);
void conv_2d_3(type_t I[C3*H3*H3], type_t W[M3*C3*R3*R3], type_t B[M3], type_t O[M3*E3*E3]);
void maxpool_1d_3(type_t I[M3*E3], type_t O[M3*L3_POOL_E]);
void maxpool_2d_3(type_t I[M3*E3*E3], type_t O[M3*L3_POOL_E*L3_POOL_E]);
void fc_layer_1(type_t input[FC1_IN_SIZE], type_t W[FC1_IN_SIZE * FC1_OUT_SIZE], type_t B[FC1_OUT_SIZE], type_t output[FC1_OUT_SIZE]);
void fc_layer_2(type_t input[FC2_IN_SIZE], type_t W[FC2_IN_SIZE * FC2_OUT_SIZE], type_t B[FC2_OUT_SIZE], type_t output[FC2_OUT_SIZE]);
void softmax(type_t input[N_CLASSES], float output[N_CLASSES]);

void cnn(type_t *input, type_t *W1, type_t *B1, type_t *W2, type_t *B2, type_t *W3, type_t *B3, type_t *W_fc1, type_t *B_fc1, type_t *W_fc2, type_t *B_fc2, float *output);

#endif // CONV_H