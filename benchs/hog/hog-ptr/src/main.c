#include <stdio.h>
#include <stdlib.h>
#include <dirent.h> 
#include <string.h>
#include <math.h>
#include "bmplib.h"
#include "hog.h"

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
            unsigned char* inputImg = getBMPImage(filePath, &header);
            
            if (inputImg == NULL) continue; 

            printf("Processing: %s (%dx%d)\n", de->d_name, header.width, header.height);

            // 2. Run HOG Kernel
            Detection* output = (Detection*)malloc(MAX_DETECTIONS * sizeof(Detection));
            int detCount = hogDetector(inputImg, header.width, header.height, output);

            printf("\nDet count: %d\n", detCount);

            printf("{");
            for (int i = 0; i < detCount; i++){
                printf("%f %f %f %f %f\n", output[i].x, output[i].y, output[i].w, output[i].h, output[i].score);
            }
            printf("}");

            free(output);

        }
    }

    closedir(dr);
    
    return 0;
}
