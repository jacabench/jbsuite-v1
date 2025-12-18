#include <stdio.h>
#include <stdlib.h>
#include "bmplib.h"
#include <string.h>

int main(int argc, char *argv[]){

    unsigned char* img;
    int width;
    int height;
    int bitsPerPixel;

    if (argc != 2){
        printf("main: Invalid argument");
        return 1;
    }

    img = getBMPImage(argv[1], &width, &height, &bitsPerPixel);

    if (bitsPerPixel == 8){

        // gray scale function
    }
    else if (bitsPerPixel == 24){

        // RGB function
    }

    setBMPImage("out.bmp", img, width, height, bitsPerPixel);

    return 0;
}