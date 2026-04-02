/**
*
*	The code versions of the smooth.
*
*/
#ifndef SMOOTH_H
#define SMOOTH_H

#include <omp.h>

void smooth(uint8 in[][sizeX], uint8 out[][sizeX]);

void smooth_reuse1(uint8 in[][sizeX], uint8 out[][sizeX]);

void smooth_reuse2(uint8 in[][sizeX], uint8 out[][sizeX]);

void smooth_reuse3(uint8 in[][sizeX], uint8 out[][sizeX]);
		
void smooth_reuse4(uint8 in[][sizeX], uint8 out[][sizeX]);

void rgb2gray(uint8 *in, uint8 *out);

void convert(uint8 *in, uint8 out[][sizeX]);

#endif 
