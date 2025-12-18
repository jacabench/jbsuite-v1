/**
 *	JACABench - JACA Benchmark Suite
 *
 *  Library for reading bmp files and output bmp or text data files.
 *          
 *	Comments:
 *		...
 *
 *	Version 1.0
 *	XXX, July 2024
 *
 *	Author: Arnaldo .... 
 *	Email = ...
 *
 */
 
 // TODO: options to select the output: bmp file or text data file
 
#include <stdio.h>
#include <stdlib.h>
#include "bmplib.h"

int main(int argc, char *argv[]){

    unsigned char* img;
    int width;
    int height;
    int bitsPerPixel;
	int colortype;

    BMPHeader bmpHeader;

    if (argc != 2){
        printf("main: Invalid argument");
        return 1;
    }

    img = getBMPImage(argv[1], &bmpHeader);

    if (bmpHeader.bitsPerPixel == 8){
		colortype = GRAY;
		printf("Color type: gray\n");
        // gray scale function
    }
    else if (bmpHeader.bitsPerPixel == 24){
		colortype = RGB;
		printf("Color type: RGB\n");
        // RGB function
    }

	
    setBMPImage("out.bmp", img, &bmpHeader, colortype);

	// save the image as a list of values seprated by ','
	outImageAsData("out.dat", img, &bmpHeader, colortype);
	
    return 0;
}