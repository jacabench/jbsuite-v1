/**
*	JACABench - JACA Benchmark Suite
*	Some useful functions.
*
*/
#ifndef UTILS_H
#define UTILS_H

#include "config.h"

void init_via_loop(uint8 in[][sizeX], int sX, int sY);

void show(uint8 out[][sizeX], int sX, int  sY);

void clear(uint8 out[][sizeX]);

long calc_checksum(uint8 *input, int num_elements);

void check_validity_checksum(uint8 *input, int num_pixels, long right_checksum);

void check_validity(uint8 *in1, uint8 *in2, int num_pixels);

#endif

