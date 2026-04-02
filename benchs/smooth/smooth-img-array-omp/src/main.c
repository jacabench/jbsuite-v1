/**
 *	JACABench - JACA Benchmark Suite
 *
 *  A smooth image kernel (also commonly known as 2D FIR) using a 3x3 coefficients window.
 *          
 *  Version using arrays and openmp for smooth and smooth_reuse1 versions
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
 *	Email = jmpc@fe.up.pt
 *
 */

#include <stdio.h>

#include "config.h"
#include "timing.h"
#include "utils.h"
#include "smooth.h"

/* input and output images */
/* each pixel of the input image is 8-bit gray and is represented by an unsigned char */
/* gold outputs are used for validity checking for now in the case of two IO options */
#if READ == VIA_INIT_ARRAYS
uint8 in[sizeY][sizeX] = 	{
							#include "init-qvga-gray.dat" 
							}; // data-text
							
	#if CHECK_VALIDITY == 2
	uint8 out_gold[sizeY][sizeX] = 	{
									#include "out-qvga-gray.dat"
									};  // gold
	#endif
#elif READ == VIA_ASCII_FILES
uint8 in[sizeY][sizeX];	
	#if CHECK_VALIDITY == 2
	uint8 out_gold[sizeY][sizeX];
	#endif
#else
uint8 in[sizeY][sizeX];
#endif

uint8 out[sizeY][sizeX];

/* for validity checking using checksum and for CONFIG=1*/
/* this is for now not used for input bmp files */
#if READ == VIA_LOOP && CHECK_VALIDITY == 1
#define RIGHT_CHECKSUM  8233134
#elif READ == VIA_INIT_ARRAYS && CHECK_VALIDITY == 1
#define RIGHT_CHECKSUM 11539953
#elif READ == VIA_ASCII_FILES && CHECK_VALIDITY == 1
#define RIGHT_CHECKSUM 11539953
#endif

/*
* main to execute the smooth versions
*/
int main() {
	
	printf("-------------- JACABench - smooth image filter -------------\n");
	printf("-------------- * array-based version -------------\n");

	/* Input image */
	#if READ == VIA_LOOP
    printf("Image initialized with a loop: width=%d, height=%d\n", sizeX, sizeY);
	/* for now a simple initialization */
	init_via_loop(in, sizeX, sizeY);

	#elif READ == VIA_ASCII_FILES
    printf("Image loaded from input text file: width=%d, height=%d\n", sizeX, sizeY);
	printf("Input gray image from text file: %s\n", INPUT_RESOURCE);
	
    /* Read input image from ascii file. */
	if(input_dsp(in, sizeX*sizeY, 4, INPUT_RESOURCE) == 0) {
		printf("Error reading input image!\n");
		return 1;
	}
	#elif READ == VIA_BMP_FILES
	BMPHeader bmpHeader;
    /* Read input bmp image. */
    uint8* img = getBMPImage(INPUT_RESOURCE, &bmpHeader);
    //uint8* img = getBMPImage("in.bmp", &bmpHeader);
    int width = bmpHeader.width;
    int height = bmpHeader.height;
    int bitsPerPixel = bmpHeader.bitsPerPixel;
	//setBMPImage("ex.bmp", ((unsigned char *) img), &bmpHeader, RGB);
	
    printf("Image loaded: %s, width=%d, height=%d\n", INPUT_RESOURCE, width, height);
    
    if(sizeX != width || sizeY != height) {
    	printf("Exiting: image size defined in CONFIG != size of image in file!\n");
    	return 1;	
	}
	
    if (bitsPerPixel == 8){
		convert(img, in);
        printf("Input gray image\n");
    } else if (bitsPerPixel == 24){
        printf("Input RGB image -> translated to gray image\n\n");
        rgb2gray(img, &in[0][0]); 
    }
    //setBMPImage("ex.bmp", &in[0][0], &bmpHeader, GRAY);
	
	#elif READ == VIA_INIT_ARRAYS
    printf("Image loaded via static initialization of input array: width=%d, height=%d\n", sizeX, sizeY);
	#endif
	
	__INIT_TIMING();

#if RUN_OPTION == 1
	printf("** Executing smooth...\n");

	__START_TIMING();

	smooth(in, out);
	
	__END_TIMING();
	__REPORT_TIMING_MS();
	
#elif RUN_OPTION == 2
	printf("** Executing smooth_reuse1...\n");
	
	__START_TIMING();

	smooth_reuse1(in, out);

	__END_TIMING();
	__REPORT_TIMING_MS();

#elif RUN_OPTION == 3

	printf("** Executing smooth_reuse2...\n");

	__START_TIMING();

	smooth_reuse2(in, out);

	__END_TIMING();
	__REPORT_TIMING_MS();

#elif RUN_OPTION == 4
	printf("** Executing smooth_reuse3...\n");
	
	__START_TIMING();

	smooth_reuse3(in, out);

	__END_TIMING();
	__REPORT_TIMING_MS();
	
#elif RUN_OPTION == 5 
	printf("** Executing smooth_reuse4...\n");		

	__START_TIMING();

	smooth_reuse4(in, out);

	__END_TIMING();
	__REPORT_TIMING_MS();
	
#endif

// Validation for CONFIG=1
#if CONFIG == 1 && CHECK_VALIDITY == 1 && (READ == VIA_LOOP || READ == VIA_INIT_ARRAYS || READ == VIA_ASCII_FILES)
	printf("** Validating results with gold...\n");
	/* some validity checking */
	check_validity_checksum(&out[0][0], sizeX*sizeY, RIGHT_CHECKSUM);
	printf("right checksum = %ld\n", (long int) RIGHT_CHECKSUM);
#elif CONFIG == 1 && CHECK_VALIDITY == 2 && (READ == VIA_INIT_ARRAYS || READ == VIA_ASCII_FILES)
	printf("** Validating results with gold...\n");
	#if READ == VIA_ASCII_FILES
	/* Read input image from ascii file. */
	printf("Gold data from: %s\n", OUTPUT_GOLD_RESOURCE);
		
	if(input_dsp(out_gold, sizeX*sizeY, 4, OUTPUT_GOLD_RESOURCE) == 0) {
		printf("Error reading input image!\n");
		return 1;
	}
	#endif
	check_validity(&out[0][0], &out_gold[0][0], sizeX*sizeY); 
#endif

#if READ == VIA_ASCII_FILES
    /* Write ouput image. */
    printf("Saving gray image to: %s\n\n", OUTPUT_RESOURCE);
    output_dsp(out, sizeX*sizeY, 4, OUTPUT_RESOURCE);
	
#elif READ == VIA_BMP_FILES
	/* Write ouput bmp image. */
    printf("Saving bmp gray image to: %s\n\n", OUTPUT_RESOURCE);
	setBMPImage(OUTPUT_RESOURCE, &out[0][0], &bmpHeader, GRAY);
	
#elif READ == VIA_INIT_ARRAYS
    printf("Image stored in out array!\n\n");
#endif

	return 0;
}
