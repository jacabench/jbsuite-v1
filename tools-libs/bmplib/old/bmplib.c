#include <stdio.h>
#include <stdlib.h>
#include "bmplib.h"

BMPHeader* getBMPHeader(FILE* bmpFile){

    if (bmpFile == NULL){
        perror("getBMPHeader: File pointer error");
        return 0;
    }

    BMPHeader* bmpHeader;

    bmpHeader = (BMPHeader*)malloc(sizeof(BMPHeader));

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

    // printf("File size: %d\n",bmpHeader->fileSize);
    // printf("Reserved: %d\n",bmpHeader->reserved);
    // printf("Data Offset: %d\n",bmpHeader->dataOffset);

    // printf("Size: %d\n",bmpHeader->size);
    // printf("Width: %d\n",bmpHeader->width);
    // printf("Height: %d\n",bmpHeader->height);
    // printf("Planes: %d\n",bmpHeader->planes);
    // printf("Bits per pixel: %d\n",bmpHeader->bitsPerPixel);
    // printf("Compression: %d\n",bmpHeader->compression);
    // printf("Image size: %d\n",bmpHeader->imageSize);
    // printf("Horizontal Resolution: %d\n",bmpHeader->horizontalResolution);
    // printf("Vertical Resolution: %d\n",bmpHeader->verticalResolution);
    // printf("Color used: %d\n",bmpHeader->colorsUsed);
    // printf("Important colors: %d\n",bmpHeader->importantColors);

    return bmpHeader;
}

unsigned char* getBMPImage(const char* path, int* width, int* height, int* bitsperPixel){

    FILE *bmpFile;
    BMPHeader* bmpHeader;
    unsigned char* bmpImage;
    unsigned char gray;
    unsigned char red;
    unsigned char green;
    unsigned char blue;
    unsigned int nullWord;

    bmpFile = fopen(path, "rb");

    if (bmpFile == NULL){
        printf("getBMPImagem: File pointer error");
        return NULL;
    }

    bmpHeader = getBMPHeader(bmpFile);

    if (bmpHeader->height <= 0 || bmpHeader->width <= 0 || bmpHeader->signature[0] != 'B' || bmpHeader->signature[1] != 'M'){
        printf("getBMPImagem: The input file is not in standard BMP format");
        return NULL;
    }

    *width = bmpHeader->width;
    *height = bmpHeader->height;
    *bitsperPixel = bmpHeader->bitsPerPixel;

    // Beginning of the bitmap data
    fseek(bmpFile, bmpHeader->dataOffset, SEEK_SET);

    // read 8 bits per pixel array
    if (bmpHeader->bitsPerPixel == 8){

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

int setBMPImage(const char* path, unsigned char* img, int width, int height, int bitsPerPixel){

    BMPHeader bmpHeader;

    FILE* bmpFile;

    bmpFile = fopen(path, "wb");

    if (bmpFile == NULL){
        printf("setBMPImage: file pointer error");
        return 0;
    }

    int nchannel = (bitsPerPixel == 24) ? 3 : 1;

    // BMP header
    bmpHeader.signature[0] = (unsigned char) 'B';
    bmpHeader.signature[1] = (unsigned char) 'M';
    fwrite(&(bmpHeader.signature), sizeof(bmpHeader.signature), 1, bmpFile);
    
    bmpHeader.fileSize = (int)(width * height * nchannel+ 54);
    fwrite(&(bmpHeader.fileSize), sizeof(bmpHeader.fileSize), 1, bmpFile);

    bmpHeader.reserved = 0;
    fwrite(&(bmpHeader.reserved), sizeof(bmpHeader.reserved), 1, bmpFile);

    bmpHeader.dataOffset = 54; // ??
    fwrite(&(bmpHeader.dataOffset), sizeof(bmpHeader.dataOffset), 1, bmpFile);

    bmpHeader.size = 40; // size of info header
    fwrite(&(bmpHeader.size), sizeof(bmpHeader.size), 1, bmpFile);

    bmpHeader.width = (int) width;
    fwrite(&( bmpHeader.width), sizeof( bmpHeader.width), 1, bmpFile);

    bmpHeader.height = (int) height;
    fwrite(&(bmpHeader.height), sizeof(bmpHeader.height), 1, bmpFile);

    bmpHeader.planes = (short int) 1;
    fwrite(&(bmpHeader.planes), sizeof(bmpHeader.planes), 1, bmpFile);

    bmpHeader.bitsPerPixel = (short int) bitsPerPixel;
    fwrite(&(bmpHeader.bitsPerPixel), sizeof(bmpHeader.bitsPerPixel), 1, bmpFile);
 
    bmpHeader.compression = (int)0;
    fwrite(&(bmpHeader.compression), sizeof(bmpHeader.compression), 1, bmpFile);

    bmpHeader.imageSize = (int)(bmpHeader.width * bmpHeader.height * nchannel);
    fwrite(&(bmpHeader.imageSize), sizeof(bmpHeader.imageSize), 1, bmpFile);

    bmpHeader.horizontalResolution = (int)2834;
    fwrite(&(bmpHeader.horizontalResolution), sizeof(bmpHeader.horizontalResolution), 1, bmpFile);

    bmpHeader.verticalResolution = (int)2834;
    fwrite(&(bmpHeader.verticalResolution), sizeof(bmpHeader.verticalResolution), 1, bmpFile);

    bmpHeader.colorsUsed = (int)0;
    fwrite(&(bmpHeader.colorsUsed), sizeof(bmpHeader.colorsUsed), 1, bmpFile);

    bmpHeader.importantColors = (int)0;    
    fwrite(&(bmpHeader.importantColors), sizeof(bmpHeader.importantColors), 1, bmpFile);

    //write BMP header
    //fwrite(&bmpHeader, sizeof(bmpHeader), 1, bmpFile);

    // read 8 bits per pixel array
    if (bmpHeader.bitsPerPixel == 8){

        for(int i = bmpHeader.height - 1; i >= 0; i--){
            for(int j = 0; j < bmpHeader.width; j++){
                //fread(&gray, sizeof(unsigned char), 1, bmpFile);
                fwrite(&img[IDXGRAY(i,j,width)], sizeof(unsigned char), 1, bmpFile);
            }
        }
    }

    // read 24 bits per pixel array
    else if(bmpHeader.bitsPerPixel == 24){

        int amountNullBytes = (bmpHeader.imageSize / bmpHeader.height) - (bmpHeader.width*3);

        for(int i = bmpHeader.height - 1; i >= 0; i--){
            for(int j = 0; j < bmpHeader.width; j++){
                fwrite(&img[IDXRGB(i,j,2,width)], sizeof(unsigned char), 1, bmpFile);
                fwrite(&img[IDXRGB(i,j,1,width)], sizeof(unsigned char), 1, bmpFile);
                fwrite(&img[IDXRGB(i,j,0,width)], sizeof(unsigned char), 1, bmpFile);
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