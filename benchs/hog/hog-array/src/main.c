#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include "bmplib.h"
#include "hog.h"

static unsigned char inputImg[IMG_WIDTH * IMG_HEIGHT * 3];
static Detection outputDets[MAX_DETECTIONS];

int main() {
    
    // Directory Iterator
    struct dirent *de; 
    DIR *dr = opendir(IMG_PATH);

    if (dr == NULL) {
        printf("Could not open directory: %s\n", IMG_PATH);
        return 1;
    }

    printf("\nStarting HOG Testbench...\n");

    // Process each file
    while ((de = readdir(dr)) != NULL) {
        
        if (strstr(de->d_name, ".bmp")) {
            
            // 1. Load Image
            char filePath[1024];
            sprintf(filePath, "%s/%s", IMG_PATH, de->d_name);
            
            BMPHeader header;
            int status = loadBMPStatic(filePath, inputImg, sizeof(inputImg), &header);
            
            if (status != 0) continue; 
            
            printf("Processing: %s (%dx%d)\n", de->d_name, header.width, header.height);

            // 2. Run HOG Kernel
            int detCount = hogDetectorStatic(inputImg, outputDets);

            printf("{");
            for (int i = 0; i < detCount; i++){
                printf("%f %f %f %f %f\n", outputDets[i].x, outputDets[i].y, outputDets[i].w, outputDets[i].h, outputDets[i].score);
            }
            printf("}");

        }
    }

    closedir(dr);
    
    return 0;
}






