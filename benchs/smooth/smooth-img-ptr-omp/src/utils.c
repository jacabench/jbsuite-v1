
#include <stdio.h>

#include "config.h"
#include "utils.h"
#include "utilsjacabenchs.h"


/* for now a simple initialization */
void init_via_loop(uint8 *in, int sX, int sY) {
	int j, i;
	for (j=0; j < sY; j++) {
		for (i= 0; i < sX; i++) {
			in[j*sX+i] = i;
		}
	}
}
	
void show(uint8 *out, int sX, int  sY) {
	int i,j;
	printf("\n-> showing %d x %d elements:\n", sY, sX);
	for (j=0; j < sY; j++) {
		for (i= 0; i < sX; i++) {
			printf("%d,%d=%d ",j, i, out[j*sX+i]);
		}
	}
}


void clear(uint8 *out, int sX) {
	int i,j;
	for (j=0; j < sizeY; j++) {
		for (i= 0; i < sizeX; i++) {
			out[j*sX+i]=0;
		}
	}
}

long calc_checksum(uint8 *input, int num_elements) {
	int i;
	long checksum = 0;
	for (i=0; i < num_elements; i++) {
			checksum += input[i];
	}
	
	return checksum;
}

void check_validity_checksum(uint8 *input, int num_pixels, long right_checksum) {
	long checksum = calc_checksum(input, num_pixels);
	
	printf("checksum = %ld\n",checksum);
	
  	if(checksum == right_checksum)
		printf("Output is RIGHT!\n");
	else
		printf("Output is WRONG!\n");
}


void check_validity(uint8 *in1, uint8 *in2, int num_pixels) {
	int success;
	//printf("address: %d\n", &in2[0]);
	success = compare_unsigned_char(in1, in2, num_pixels);
	
  	if(success)
		printf("Output is RIGHT!\n");
	else
		printf("Output is WRONG!\n");
}
