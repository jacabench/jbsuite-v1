#ifndef HOG_DYNAMIC_H
#define HOG_DYNAMIC_H

#define IMG_PATH "../../images/"

// HOG (Dalal & Triggs)
#define HOG_WIN_WIDTH 64
#define HOG_WIN_HEIGHT 128
#define CELL_SIZE 8
#define BLOCK_SIZE 2 // cells per block (2x2)
#define BINS 9
#define INITIAL_SCALE 1.0
#define SCALE_FACTOR 1.05

// Dimensions
#define CELLS_X (HOG_WIN_WIDTH / CELL_SIZE)     // 8
#define CELLS_Y (HOG_WIN_HEIGHT / CELL_SIZE)    // 16
#define BLOCKS_X (CELLS_X - BLOCK_SIZE + 1)     // 7
#define BLOCKS_Y (CELLS_Y - BLOCK_SIZE + 1)     // 15

// Memory configuration
#define IMG_WIDTH  320   // VGA width
#define IMG_HEIGHT 240   // VGA height
#define NUM_PIXELS (IMG_WIDTH * IMG_HEIGHT)

#define MAX_DETECTIONS 1000   // Max detected pedestrians per frame
#define DESCRIPTOR_SIZE (BLOCKS_X * BLOCKS_Y * BLOCK_SIZE * BLOCK_SIZE * BINS) // 3780 features

typedef struct {
    float x, y, w, h;
    float score;
} Detection;

int hogDetector(unsigned char* input, int width, int height, Detection* output); 

#endif
