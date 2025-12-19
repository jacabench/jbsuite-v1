/**
*
*	The code versions of the smooth.
*
*/
#ifndef SMOOTH_H
#define SMOOTH_H

void smooth(uint8 *in, uint8 *out, int sX, int  sY);

void smooth_reuse1(uint8 *in, uint8 *out, int sX, int  sY);

void smooth_reuse2(uint8 *in, uint8 *out, int sX, int  sY);

void smooth_reuse3(uint8 *in, uint8 *out, int sX, int  sY);
		
void smooth_reuse4(uint8 *in, uint8 *out, int sX, int  sY);

void rgb2gray(uint8 *in, uint8 *out, int sX, int  sY);

void convert(uint8 *in, uint8 *out, int sX, int  sY);

#endif 
