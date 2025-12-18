#ifndef SRC_BMPLIB_H_
#define SRC_BMPLIB_H_

#include<stdio.h>

#define IDXGRAY(I,J,W) (I*W+J)
#define IDXRGB(I,J,K,W) (3*I*W+J*3+K)

typedef struct
{
    char signature[2];       // 2B
    int fileSize;            // 4B
    int reserved;            // 4B
    int dataOffset;          // 4B 
    int size;                // 4B
    int width;               // 4B
    int height;              // 4B
    short int planes;        // 2B
    short int bitsPerPixel;  // 2B
    int compression;         // 4B
    int imageSize;           // 4B
    int horizontalResolution;// 4B
    int verticalResolution;  // 4B
    int colorsUsed;          // 4B
    int importantColors;     // 4B
} BMPHeader;

BMPHeader* getBMPHeader(FILE* bmpFile);

unsigned char* getBMPImage(const char* path, int* width, int* height, int* bitsperPixel);

int setBMPImage(const char* path, unsigned char* img, int width, int height, int bitsPerPixel);

#endif /* SRC_BMPLIB_H_ */