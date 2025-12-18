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

#include <stdio.h>
#include <stdlib.h>
#include "bmplib.h"

unsigned char* getBMPImage(const char* path, BMPHeader* bmpHeader){

    FILE *bmpFile;
    unsigned char* bmpImage;
    unsigned char gray;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    //JMPC: unsigned int nullWord;

    bmpFile = fopen(path, "rb");

    if (bmpFile == NULL){
        printf("getBMPImagem: File pointer error");
        return NULL;
    }

    // // read BMP header from file
    // // -------------------------------
    fread(&(bmpHeader->signature), sizeof(bmpHeader->signature), 1, bmpFile);
    fread(&(bmpHeader->fileSize), sizeof(bmpHeader->fileSize), 1, bmpFile);
    fread(&(bmpHeader->reserved), sizeof(bmpHeader->reserved), 1, bmpFile);
    fread(&(bmpHeader->dataOffset), sizeof(bmpHeader->dataOffset), 1, bmpFile);

    // // read BMP info header from file
    // // -------------------------------
    fread(&(bmpHeader->size), sizeof(bmpHeader->size), 1, bmpFile);
    fread(&(bmpHeader->width), sizeof(bmpHeader->width), 1, bmpFile);
    fread(&(bmpHeader->height), sizeof(bmpHeader->height), 1, bmpFile);
    fread(&(bmpHeader->planes), sizeof(bmpHeader->planes), 1, bmpFile);
    fread(&(bmpHeader->bitsPerPixel), sizeof(bmpHeader->bitsPerPixel), 1, bmpFile);
    fread(&(bmpHeader->compression), sizeof(bmpHeader->compression), 1, bmpFile);
    fread(&(bmpHeader->imageSize), sizeof(bmpHeader->imageSize), 1, bmpFile);
    fread(&(bmpHeader->horizontalResolution), sizeof(bmpHeader->horizontalResolution), 1, bmpFile);
    fread(&(bmpHeader->verticalResolution), sizeof(bmpHeader->verticalResolution), 1, bmpFile);
    fread(&(bmpHeader->colorsUsed), sizeof(bmpHeader->colorsUsed), 1, bmpFile);
    fread(&(bmpHeader->importantColors), sizeof(bmpHeader->importantColors), 1, bmpFile);

    if (bmpHeader->height <= 0 || bmpHeader->width <= 0 || bmpHeader->signature[0] != 'B' || bmpHeader->signature[1] != 'M'){
        printf("getBMPImagem: The input file is not in standard BMP format");
        return NULL;
    }

    // read 8 bits per pixel array
    if (bmpHeader->bitsPerPixel == 8){

        //JMPC: char table[bmpHeader->importantColors][4];

        for (int i = 0; i < 256; i++){
            int red, green,blue, unused;
            fread(&red, sizeof(unsigned char), 1, bmpFile);
            fread(&green, sizeof(unsigned char), 1, bmpFile);
            fread(&blue, sizeof(unsigned char), 1, bmpFile);
            fread(&unused, sizeof(unsigned char), 1, bmpFile);
        }


        bmpImage = (unsigned char*)malloc(sizeof(unsigned char) * bmpHeader->height * bmpHeader->width);

        for(int i = bmpHeader->height - 1; i >= 0; i--){
            for(int j = 0; j < bmpHeader->width; j++){
                fread(&gray, sizeof(unsigned char), 1, bmpFile);
                bmpImage[i * bmpHeader->width + j] = gray;
            }
        }
    }

    // read 24 bits per pixel array
    else if(bmpHeader->bitsPerPixel == 24){

        // Beginning of the bitmap data
        fseek(bmpFile, bmpHeader->dataOffset, SEEK_SET);


        bmpImage = (unsigned char*)malloc(sizeof(unsigned char) * bmpHeader->height * bmpHeader->width * 3);

        for(int i = bmpHeader->height - 1; i >= 0; i--){
            for(int j = 0; j < bmpHeader->width; j++){
                fread(&blue, sizeof(unsigned char), 1, bmpFile);
                fread(&green, sizeof(unsigned char), 1, bmpFile);
                fread(&red, sizeof(unsigned char), 1, bmpFile);

                bmpImage[IDXRGB(i,j,0,bmpHeader->width)] = red;
                bmpImage[IDXRGB(i,j,1,bmpHeader->width)] = green;
                bmpImage[IDXRGB(i,j,2,bmpHeader->width)] = blue;
            }

            // remove null bytes
            int amountNullBytes = (bmpHeader->imageSize / bmpHeader->height) - (bmpHeader->width*3);
            unsigned char nullbyte;
            for(int i = 0; i < amountNullBytes; i++){
                fread(&nullbyte, sizeof(unsigned char), 1, bmpFile);
            }
        }       
    }

   else {
        printf("getBMPImagem: Only 8 or 24 bits per pixel allowed");
        return NULL;
    }


    fclose(bmpFile);
    return bmpImage;
}



int setBMPImage(const char* path, unsigned char* img, BMPHeader* bmpHeader, int colortype){

    FILE* bmpFile;

    bmpFile = fopen(path, "wb");

    if (bmpFile == NULL){
        printf("setBMPImage: file pointer error");
        return 0;
    }
	
	if(colortype == GRAY && bmpHeader->bitsPerPixel == 24) {
		bmpHeader->bitsPerPixel = 8;
        printf("setBMPImage: writing a gray bmp from an rgb bmp header");
	}
	
	if(colortype == RGB && bmpHeader->bitsPerPixel == 8) {
        printf("setBMPImage: writing an rgb bmp from a gray bmp header not supported");
        return 0;
    }

    fwrite(&(bmpHeader->signature), sizeof(bmpHeader->signature), 1, bmpFile);
    fwrite(&(bmpHeader->fileSize), sizeof(bmpHeader->fileSize), 1, bmpFile);
    fwrite(&(bmpHeader->reserved), sizeof(bmpHeader->reserved), 1, bmpFile);
    fwrite(&(bmpHeader->dataOffset), sizeof(bmpHeader->dataOffset), 1, bmpFile);
    fwrite(&(bmpHeader->size), sizeof(bmpHeader->size), 1, bmpFile);
    fwrite(&(bmpHeader->width), sizeof(bmpHeader->width), 1, bmpFile);
    fwrite(&(bmpHeader->height), sizeof(bmpHeader->height), 1, bmpFile);
    fwrite(&(bmpHeader->planes), sizeof(bmpHeader->planes), 1, bmpFile);
    fwrite(&(bmpHeader->bitsPerPixel), sizeof(bmpHeader->bitsPerPixel), 1, bmpFile);
    fwrite(&(bmpHeader->compression), sizeof(bmpHeader->compression), 1, bmpFile);
    fwrite(&(bmpHeader->imageSize), sizeof(bmpHeader->imageSize), 1, bmpFile);
    fwrite(&(bmpHeader->horizontalResolution), sizeof(bmpHeader->horizontalResolution), 1, bmpFile);
    fwrite(&(bmpHeader->verticalResolution), sizeof(bmpHeader->verticalResolution), 1, bmpFile);
    fwrite(&(bmpHeader->colorsUsed), sizeof(bmpHeader->colorsUsed), 1, bmpFile);
    fwrite(&(bmpHeader->importantColors), sizeof(bmpHeader->importantColors), 1, bmpFile);

    // write 8 bits per pixel array
    if (bmpHeader->bitsPerPixel == 8){


        for (int i = 0; i < 256; i++){
            //JMPC: int red, green,blue, unused = 0;
            int unused = 0;
            fwrite(&i, sizeof(unsigned char), 1, bmpFile);
            fwrite(&i, sizeof(unsigned char), 1, bmpFile);
            fwrite(&i, sizeof(unsigned char), 1, bmpFile);
            fwrite(&unused, sizeof(unsigned char), 1, bmpFile);
        }

        for(int i = bmpHeader->height - 1; i >= 0; i--){
            for(int j = 0; j < bmpHeader->width; j++){
                fwrite(&(img[IDXGRAY(i,j,bmpHeader->width)]), sizeof(unsigned char), 1, bmpFile);
            }
        }
    }

    // write 24 bits per pixel array
    else if(bmpHeader->bitsPerPixel == 24){

        int amountNullBytes = (bmpHeader->imageSize / bmpHeader->height) - (bmpHeader->width*3);

        for(int i = bmpHeader->height - 1; i >= 0; i--){
            for(int j = 0; j < bmpHeader->width; j++){
                fwrite(&img[IDXRGB(i,j,2,bmpHeader->width)], sizeof(unsigned char), 1, bmpFile);
                fwrite(&img[IDXRGB(i,j,1,bmpHeader->width)], sizeof(unsigned char), 1, bmpFile);
                fwrite(&img[IDXRGB(i,j,0,bmpHeader->width)], sizeof(unsigned char), 1, bmpFile);
            }

            // add null bytes
            unsigned char nullbyte = 0;
            for(int i = 0; i < amountNullBytes; i++){
                fwrite(&nullbyte, sizeof(unsigned char), 1, bmpFile);
            }
        }       
    }    

    fclose(bmpFile);
    return 1;
}


int outImageAsData(const char* path, unsigned char* img, BMPHeader* bmpHeader, int colortype) {
	
    FILE* textFile;

    textFile = fopen(path, "w");

    if (textFile == NULL){
        printf("outImageAsData: file pointer error");
        return 0;
    }
	
	if(colortype == GRAY && bmpHeader->bitsPerPixel == 24) {
		bmpHeader->bitsPerPixel = 8;
        printf("setBMPImage: writing a gray bmp from an rgb bmp header");
	}
	
	if(colortype == RGB && bmpHeader->bitsPerPixel == 8) {
        printf("setBMPImage: writing an rgb bmp from a gray bmp header not supported");
        return 0;
    }


	printf("width = %d, height = %d\n", bmpHeader->width, bmpHeader->height);
		
    // write 8 bits per pixel array
    if (bmpHeader->bitsPerPixel == 8){

        for(int i = 0; i<bmpHeader->height; i++){	
            for(int j = 0; j < bmpHeader->width; j++){
				if(j == bmpHeader->width-1 && i == bmpHeader->height-1)
					fprintf(textFile, "%d", img[IDXGRAY(i,j,bmpHeader->width)]);
				else 
					fprintf(textFile, "%d,", img[IDXGRAY(i,j,bmpHeader->width)]);
            }
        }
    }
    // write 24 bits per pixel array
    else if(bmpHeader->bitsPerPixel == 24){

        for(int i = 0; i < bmpHeader->height; i++){
            for(int j = 0; j < bmpHeader->width; j++){
				if(j == bmpHeader->width-1 && i == bmpHeader->height-1) {
					fprintf(textFile, "%d", img[IDXRGB(i,j,2,bmpHeader->width)]);
					fprintf(textFile, "%d", img[IDXRGB(i,j,1,bmpHeader->width)]);
					fprintf(textFile, "%d", img[IDXRGB(i,j,0,bmpHeader->width)]);
				} else {
					fprintf(textFile, "%d,", img[IDXRGB(i,j,2,bmpHeader->width)]);
					fprintf(textFile, "%d,", img[IDXRGB(i,j,1,bmpHeader->width)]);
					fprintf(textFile, "%d,", img[IDXRGB(i,j,0,bmpHeader->width)]);
				}
			}       
		}
	}		

    fclose(textFile);
    return 1;
}
