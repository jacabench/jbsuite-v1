/**
 *	JACABench - JACA Benchmark Suite
 *
 *  A smooth image kernel (also commonly known as 2D FIR) using a 3x3 coefficients window.
 * 
 *	The benchmark consists of the original version and four versions using data reuse:
 * 		smooth: the original version
 *		smooth_reuse1: 	data reuse of 6 pixels for each sliding window step
 *		smooth_reuse2: 	the use of 3 buffers (one per line)  and reuse of data in the buffers
 *		smooth_reuse3: 	the use of 3 buffers (one per line), reuse of data in the buffers, 
 * 						and reuse of 6 pixels for each sliding window step
 *		smooth_reuse4: 	the use of 3 buffers (1 per line), reuse of data in the buffers, 
 * 						reuse of 6 pixels for each sliding window step,
 *						and resuse the 6 pixels when starting a new row of pixels	
 *
 *	Comments:
 *		The line buffers versions maintain in 3 buffers 3 consecutive image lines, 
 * 		update with the subsequent line in a "rotating registers" based scheme.
 *		A possible optimization in the versions with line buffers is to use 4 line buffers
 *		and to overlap the loading of the 4th line with the computations regarding the
 *		pixels in the other 3 line buffers.
 *
 *	Version 1.0
 *	December 2025
 *	Copyright University of Porto, Faculty of Engineering (FEUP) Porto, Portugal
 *
 *	Author: João MP Cardoso 
 *	Email = jmpc@fe.up.pt*
 */

/* Timing options
	TIMING:
	0: none timing measurements
	1: for considering timing measurements in Windows
	2: for considering timing measurements in Linux
*/

#include "config.h"
#include "smooth.h"

//#pragma distribute_point
//#pragma omp simd
void smooth(uint8 *in, uint8 *out, int sX, int sY) {
    int j, i, r, c, sum;
	uint8 k[W*W] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	for (j=0; j < sY-2; j++) {
		for (i=0; i < sX-2; i++) {
			sum = 0;
			for (r=0; r < W; r++) {
				for (c = 0; c < W; c++) {
					sum += in[(j+r)*sX+i+c]*k[r*W+c];
				}
			}
			sum = sum / 16;
			out[(j+1)*sX+i+1] = (uint8) sum;
		}
	}
}


void smooth_reuse1(uint8 *in, uint8 *out, int sX, int sY) {
    int j, i, sum;
	uint8 k[W*W] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	
	// for data reuse
	uint8 p11, p12, p13;
	uint8 p21, p22, p23;
	uint8 p31, p32, p33;
	
	for (j=0; j < sY-2; j++) {
		p11 = in[j*sX+0];
		p12 = in[j*sX+1];
		p13 = in[j*sX+2];
					
		p21 = in[(j+1)*sX+0];
		p22 = in[(j+1)*sX+1];
		p23 = in[(j+1)*sX+2];
					
		p31 = in[(j+2)*sX+0];
		p32 = in[(j+2)*sX+1];
		p33 = in[(j+2)*sX+2];
		
		sum  = p11*k[0*W+0];
		sum += p12*k[0*W+1];
		sum += p13*k[0*W+2];
					
		sum += p21*k[1*W+0];
		sum += p22*k[1*W+1];
		sum += p23*k[1*W+2];
					
		sum += p31*k[2*W+0];
		sum += p32*k[2*W+1];
		sum += p33*k[2*W+2];
			
		sum = sum / 16;
		out[(j+1)*sX+1] = (uint8) sum;

		for (i = 3; i < sX; i++) {
			p11 = p12;
			p21 = p22;
			p31 = p32;
			
			p12 = p13;
			p22 = p23;
			p32 = p33;
			
			p13 = in[j*sX+i];
			p23 = in[(j+1)*sX+i];
			p33 = in[(j+2)*sX+i];
			
			sum  = p11*k[0*W+0];
			sum += p12*k[0*W+1];
			sum += p13*k[0*W+2];
					
			sum += p21*k[1*W+0];
			sum += p22*k[1*W+1];
			sum += p23*k[1*W+2];
					
			sum += p31*k[2*W+0];
			sum += p32*k[2*W+1];
			sum += p33*k[2*W+2];
			
			sum = sum / 16;
			out[(j+1)*sX+i-1] = (uint8) sum;
		}
	}
}

void smooth_reuse2(uint8 *in, uint8 *out, int sX, int sY) {
    int j, i, sum;
	uint8 k[W*W] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	
	// the three buffers
	uint8 buf1[sX];
	uint8 buf2[sX];
	uint8 buf3[sX];
	uint8 *b1 = buf1;
	uint8 *b2 = buf2;
	uint8 *b3 = buf3;
	uint8 *t;
	
	for (i = 0; i < sX; i++) { // load the first 3 rows to the 3 buffers
		b1[i] = in[0*sX+i];
		b2[i] = in[1*sX+i];
		b3[i] = in[2*sX+i];
	}
	for (j=0; j < sY-2; j++) {
		if(j != 0) { // after the first row, do rotating buffers and load next row
			t = b1;
			b1 = b2;
			b2 = b3;
			b3 = t;	
			for (i = 0; i < sX; i++) { // load the new row to the third buffer
				b3[i] = in[(j+2)*sX+i];
			}
		}
		for (i = 0; i < sX-2; i++) { // do smoothing 
			sum  = b1[i]*k[0*W+0];
			sum += b2[i]*k[1*W+0];
			sum += b3[i]*k[2*W+0];
			
			sum += b1[i+1]*k[0*W+1];
			sum += b2[i+1]*k[1*W+1];
			sum += b3[i+1]*k[2*W+1];
			
			sum += b1[i+2]*k[0*W+2];
			sum += b2[i+2]*k[1*W+2];
			sum += b3[i+2]*k[2*W+2];
			
			sum = sum / 16;
			out[(j+1)*sX+i+1] = (uint8) sum;
		}
	}
}


void smooth_reuse3(uint8 *in, uint8 *out, int sX, int sY) {
    int j, i, sum;
	uint8 k[W*W] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	
	// the three buffers
	uint8 buf1[sX];
	uint8 buf2[sX];
	uint8 buf3[sX];
	uint8 *b1 = buf1;
	uint8 *b2 = buf2;
	uint8 *b3 = buf3;
	uint8 *t;
	
	// for data reuse
	uint8 p11, p12, p13;
	uint8 p21, p22, p23;
	uint8 p31, p32, p33;

	for (i = 0; i < sX; i++) { // load the first 3 rows to the 3 buffers
		b1[i] = in[0*sX+i];
		b2[i] = in[1*sX+i];
		b3[i] = in[2*sX+i];
	}
	for (j=0; j < sY-2; j++) {
		if(j != 0) { // after the first row, do rotating buffers and load next row
			t = b1;
			b1 = b2;
			b2 = b3;
			b3 = t;	
			for (i = 0; i < sX; i++) { // load the new row to the third buffer
				b3[i] = in[(j+2)*sX+i];
			}
		}
		
		// load from buffers to scalar variables
		p11 = b1[0];
		p12 = b1[1];
		p13 = b1[2];
						
		p21 = b2[0];
		p22 = b2[1];
		p23 = b2[2];
					
		p31 = b3[0];
		p32 = b3[1];
		p33 = b3[2];
		
		for (i = 0; i < sX-2; i++) { // do smoothing 
			if(i != 0) { // rotate buffers and load the W pixels of next column
				p11 = p12;
				p21 = p22;
				p31 = p32;
			
				p12 = p13;
				p22 = p23;
				p32 = p33;
			
				p13 = b1[i+2];
				p23 = b2[i+2];
				p33 = b3[i+2];
			}
			sum  = p11*k[0*W+0];
			sum += p21*k[1*W+0];
			sum += p31*k[2*W+0];
			
			sum += p12*k[0*W+1];
			sum += p22*k[1*W+1];
			sum += p32*k[2*W+1];
			
			sum += p13*k[0*W+2];
			sum += p23*k[1*W+2];
			sum += p33*k[2*W+2];
			
			sum = sum / 16;
			out[(j+1)*sX+i+1] = (uint8) sum;
		}
	}
}


void smooth_reuse4(uint8 *in, uint8 *out, int sX, int sY) {
    int j, i, sum;
	uint8 k[W*W] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
	
	// the three buffers
	uint8 buf1[sX];
	uint8 buf2[sX];
	uint8 buf3[sX];
	uint8 *b1 = buf1;
	uint8 *b2 = buf2;
	uint8 *b3 = buf3;
	uint8 *t;
	
	// for data reuse
	uint8 p11, p12, p13;
	uint8 p21, p22, p23;
	uint8 p31, p32, p33;
	uint8 p21b, p22b, p23b; // vertically
	uint8 p31b, p32b, p33b; // vertically

	for (i = 0; i < sX; i++) { // load the first 3 rows to the 3 buffers
		b1[i] = in[0*sX+i];
		b2[i] = in[1*sX+i];
		b3[i] = in[2*sX+i];
	}
	for (j=0; j < sY-2; j++) {
		if(j != 0) { // after the first row, do rotating buffers and load next row
			t = b1;
			b1 = b2;
			b2 = b3;
			b3 = t;	
			for (i = 0; i < sX; i++) { // load the new row to the third buffer
				b3[i] = in[(j+2)*sX+i];
			}
		}

		if(j == 0) {
			p11 = b1[0];
			p12 = b1[1];
			p13 = b1[2];
					
			p21 = b2[0];
			p22 = b2[1];
			p23 = b2[2];
					
			p31 = b3[0];
			p32 = b3[1];
			p33 = b3[2];
		
			p21b = p21;
			p22b = p22;
			p23b = p23;
			
			p31b = p31;
			p32b = p32;
			p33b = p33;
		} else {
			p11 = p21b;
			p12 = p22b;
			p13 = p23b;
			
			p21 = p31b;
			p22 = p32b;
			p23 = p33b;
			
			p31 = b3[0];
			p32 = b3[1];
			p33 = b3[2];
			
			p21b = p21;
			p22b = p22;
			p23b = p23;
			
			p31b = p31;
			p32b = p32;
			p33b = p33;
		}
		
		for (i = 0; i < sX-2; i++) { // do smoothing 
			if(i != 0) {
				p11 = p12;
				p21 = p22;
				p31 = p32;
			
				p12 = p13;
				p22 = p23;
				p32 = p33;
			
				p13 = b1[i+2];
				p23 = b2[i+2];
				p33 = b3[i+2];
			}
			
			sum  = p11*k[0*W+0];
			sum += p21*k[1*W+0];
			sum += p31*k[2*W+0];
			
			sum += p12*k[0*W+1];
			sum += p22*k[1*W+1];
			sum += p32*k[2*W+1];
			
			sum += p13*k[0*W+2];
			sum += p23*k[1*W+2];
			sum += p33*k[2*W+2];
			
			sum = sum / 16;
			out[(j+1)*sX+i+1] = (uint8) sum;
		}
	}
}

/**
* Convert RGB to gray image stored as a list of bytes to the gray image as a 
* 2D array sY*sX used in the smooth benchmark.
*/
void rgb2gray(uint8 *in, uint8 *out, int sX, int sY) {
	int i, j=0;
	for(i=0; i<sX*sY;i++) {
		out[i]=(in[j]+in[j+1]+in[j+3])/3;
		j+=3;
		//printf("%d, ", out[i]);
	}
}

/**
* Convert gray image stored as a list of bytes to the 2D array sY*sX used 
* in the smooth benchmark.
*/
void convert(uint8 *in, uint8 *out, int sX, int sY) {
	int i, j, k=0;
	for(i=0; i<sY;i++) {
		for(j=0; j<sX;j++) {
			out[i*sX+j]=in[k++];
		}
	}
}
