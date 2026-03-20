#include "hog.h"
#include "weights.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define PI 3.14159265f
#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

static float g_gaussian_lut[256] = {0.415237, 0.463230, 0.508759, 0.550099, 0.585578, 0.613680, 0.633161, 0.643131, 0.643131, 0.633161, 0.613680, 0.585578, 0.550099, 0.508759, 0.463230, 0.415237, 0.463230, 0.516771, 0.567561, 0.613680, 0.653259, 0.684610, 0.706342, 0.717465, 0.717465, 0.706342, 0.684610, 0.653259, 0.613680, 0.567561, 0.516771, 0.463230, 0.508759, 0.567561, 0.623344, 0.673996, 0.717465, 0.751897, 0.775765, 0.787981, 0.787981, 0.775765, 0.751897, 0.717465, 0.673996, 0.623344, 0.567561, 0.508759, 0.550099, 0.613680, 0.673996, 0.728763, 0.775765, 0.812994, 0.838801, 0.852011, 0.852011, 0.838801, 0.812994, 0.775765, 0.728763, 0.673996, 0.613680, 0.550099, 0.585578, 0.653259, 0.717465, 0.775765, 0.825797, 0.865428, 0.892900, 0.906961, 0.906961, 0.892900, 0.865428, 0.825797, 0.775765, 0.717465, 0.653259, 0.585578, 0.613680, 0.684610, 0.751897, 0.812994, 0.865428, 0.906961, 0.935751, 0.950487, 0.950487, 0.935751, 0.906961, 0.865428, 0.812994, 0.751897, 0.684610, 0.613680, 0.633161, 0.706342, 0.775765, 0.838801, 0.892900, 0.935751, 0.965455, 0.980658, 0.980658, 0.965455, 0.935751, 0.892900, 0.838801, 0.775765, 0.706342, 0.633161, 0.643131, 0.717465, 0.787981, 0.852011, 0.906961, 0.950487, 0.980658, 0.996101, 0.996101, 0.980658, 0.950487, 0.906961, 0.852011, 0.787981, 0.717465, 0.643131, 0.643131, 0.717465, 0.787981, 0.852011, 0.906961, 0.950487, 0.980658, 0.996101, 0.996101, 0.980658, 0.950487, 0.906961, 0.852011, 0.787981, 0.717465, 0.643131, 0.633161, 0.706342, 0.775765, 0.838801, 0.892900, 0.935751, 0.965455, 0.980658, 0.980658, 0.965455, 0.935751, 0.892900, 0.838801, 0.775765, 0.706342, 0.633161, 0.613680, 0.684610, 0.751897, 0.812994, 0.865428, 0.906961, 0.935751, 0.950487, 0.950487, 0.935751, 0.906961, 0.865428, 0.812994, 0.751897, 0.684610, 0.613680, 0.585578, 0.653259, 0.717465, 0.775765, 0.825797, 0.865428, 0.892900, 0.906961, 0.906961, 0.892900, 0.865428, 0.825797, 0.775765, 0.717465, 0.653259, 0.585578, 0.550099, 0.613680, 0.673996, 0.728763, 0.775765, 0.812994, 0.838801, 0.852011, 0.852011, 0.838801, 0.812994, 0.775765, 0.728763, 0.673996, 0.613680, 0.550099, 0.508759, 0.567561, 0.623344, 0.673996, 0.717465, 0.751897, 0.775765, 0.787981, 0.787981, 0.775765, 0.751897, 0.717465, 0.673996, 0.623344, 0.567561, 0.508759, 0.463230, 0.516771, 0.567561, 0.613680, 0.653259, 0.684610, 0.706342, 0.717465, 0.717465, 0.706342, 0.684610, 0.653259, 0.613680, 0.567561, 0.516771, 0.463230, 0.415237, 0.463230, 0.508759, 0.550099, 0.585578, 0.613680, 0.633161, 0.643131, 0.643131, 0.633161, 0.613680, 0.585578, 0.550099, 0.508759, 0.463230, 0.415237};

// RGB Pre-processing: Normalize [0,1] and apply Gamma (sqrt)
void hog_preprocess_rgb(unsigned char input[IMG_WIDTH * IMG_HEIGHT * 3], 
                        int width, 
                        int height, 
                        float output[IMG_WIDTH * IMG_HEIGHT * 3]) {
    
    int num_pixels = width * height;

    
    int total_elements = num_pixels * 3;

    static unsigned char scratch_local[IMG_WIDTH * IMG_HEIGHT * 3];

    for (int i = 0; i < total_elements; i++) {
        scratch_local[i] = input[i];
    }

    for (int i = 0; i < total_elements; i++) {
        float norm = scratch_local[i] * (1.0f / 255.0f);
        output[i] = sqrtf(norm);
    }

}

// Compute Gradients
void computeGradientsRGB(float input[IMG_WIDTH * IMG_HEIGHT * 3], 
                         int width, 
                         int height, 
                         float out_mag[IMG_WIDTH * IMG_HEIGHT], 
                         float out_angle[IMG_WIDTH * IMG_HEIGHT]) {
    
    
    int stride = width * height;

    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {

            int is_border = (x == 0 || x == width - 1 || y == 0 || y == height - 1);

            if (is_border) {
                int idx = y * width + x;
                out_mag[idx] = 0.0f;
                out_angle[idx] = 0.0f;

            } else {

                int idx = y * width + x;
                int idx_prev_x = y * width + (x - 1);
                int idx_next_x = y * width + (x + 1);
                int idx_prev_y = (y - 1) * width + x;
                int idx_next_y = (y + 1) * width + x;

                float max_mag_sq = -1.0f;
                float best_dx = 0.0f;
                float best_dy = 0.0f;

                int offsets[3];
                offsets[0] = 0;
                offsets[1] = stride;
                offsets[2] = 2 * stride;
                
                for (int c = 0; c < 3; c++) {
                    
                    int ch_offset = offsets[c];

                    float val_prev_x = input[ch_offset + idx_prev_x];
                    float val_next_x = input[ch_offset + idx_next_x];
                    float val_prev_y = input[ch_offset + idx_prev_y];
                    float val_next_y = input[ch_offset + idx_next_y];

                    float dx = val_next_x - val_prev_x;
                    float dy = val_next_y - val_prev_y;

                    float mag_sq = dx * dx + dy * dy;

                    if (mag_sq > max_mag_sq) {
                        max_mag_sq = mag_sq;
                        best_dx = dx;
                        best_dy = dy;
                    }
                }

                out_mag[idx] = sqrtf(max_mag_sq);
                
                float ang = atan2f(best_dy, best_dx) * (180.0f / 3.14159265f);
                if (ang < 0) ang += 360.0f; 
                if (ang >= 180.0f) ang -= 180.0f;

                out_angle[idx] = ang;
            }
        }
    }
}


void resizeBilinearPlanar(unsigned char src[IMG_WIDTH * IMG_HEIGHT * 3], 
                          int src_w,
                          int src_h, 
                          unsigned char dst[IMG_WIDTH * IMG_HEIGHT * 3], 
                          int dst_w, 
                          int dst_h) {
    
    float x_ratio = ((float)(src_w - 1)) / dst_w;
    float y_ratio = ((float)(src_h - 1)) / dst_h;
    
    int src_plane_size = src_w * src_h;
    int dst_plane_size = dst_w * dst_h;


    for (int c = 0; c < 3; c++) {
        const unsigned char* src_plane = src + (c * src_plane_size);
        unsigned char* dst_plane = dst + (c * dst_plane_size);

        float y = 0.0f;

        for (int i = 0; i < dst_h; i++) {

            int y_l = (int)y;
            int y_h = (y_l + 1 < src_h) ? y_l + 1 : y_l;
            float y_diff = y - y_l;

            float x = 0.0f;

            for (int j = 0; j < dst_w; j++) {
              
                int x_l = (int)x;
                int x_h = (x_l + 1 < src_w) ? x_l + 1 : x_l;
                float x_diff = x - x_l;

                int idxA = y_l * src_w + x_l;
                int idxB = y_l * src_w + x_h;
                int idxC = y_h * src_w + x_l;
                int idxD = y_h * src_w + x_h;

                float A = src_plane[idxA];
                float B = src_plane[idxB];
                float C = src_plane[idxC];
                float D = src_plane[idxD];

                float w1 = (1.0f - x_diff);
                float w2 = (1.0f - y_diff);

                float p1 = A * w1;
                float p2 = B * x_diff;
                float p3 = C * w1;
                float p4 = D * x_diff;

                float pixel =
                    p1 * w2 +
                    p2 * w2 +
                    p3 * y_diff +
                    p4 * y_diff;

                dst_plane[i * dst_w + j] = (unsigned char)pixel;

                x += x_ratio;
            }

            y += y_ratio;
        }
    }
}


void apply_l2_hys(float block_vector[36], int size) {
    
    float buf[36];
    
    float sum_sq = 0.0f;
    float eps = 0.001f; 

    // for (int i = 0; i < size; i++) {
    //     sum_sq += block_vector[i] * block_vector[i];
    // }
    for (int i = 0; i < size; i++) {
        buf[i] = block_vector[i];
        sum_sq += buf[i] * buf[i];
    }

    float scale = 1.0f / sqrtf(sum_sq + eps * eps);
    
    // for (int i = 0; i < size; i++) {
    //     block_vector[i] *= scale;
    // }
    for (int i = 0; i < size; i++) {
        buf[i] *= scale;
    }


    sum_sq = 0.0f; 
    // for (int i = 0; i < size; i++) {
    //     if (block_vector[i] > 0.2f) {
    //         block_vector[i] = 0.2f;
    //     }
    //     sum_sq += block_vector[i] * block_vector[i];
    // }
    for (int i = 0; i < size; i++) {
        if (buf[i] > 0.2f)
            buf[i] = 0.2f;
        sum_sq += buf[i] * buf[i];
    }

    scale = 1.0f / sqrtf(sum_sq + eps * eps);
    // for (int i = 0; i < size; i++) {
    //     block_vector[i] *= scale;
    // }
    for (int i = 0; i < size; i++) {
        block_vector[i] = buf[i] * scale;
    }
}


void extractHOGFeature(float mag[IMG_WIDTH * IMG_HEIGHT], 
                       float angle[IMG_WIDTH * IMG_HEIGHT],
                       int img_w, 
                       int img_h, 
                       int win_x, 
                       int win_y, 
                       float gaussian_lut[256],
                       float descriptor[DESCRIPTOR_SIZE]) {


    int desc_idx = 0;
    
    int cells_per_block = BLOCK_SIZE; // 2
    int pixels_per_cell = CELL_SIZE;  // 8
    int pixels_per_block = cells_per_block * pixels_per_cell; // 16
    int bins = BINS;                  // 9
    float angle_unit = 180.0f / bins; 


    // -----------------------------------------------------------
    
    for (int by = 0; by < BLOCKS_Y; by++) {
        for (int bx = 0; bx < BLOCKS_X; bx++) {

            float hist[BLOCK_SIZE][BLOCK_SIZE][BINS];
           
            
            //memset(hist, 0, sizeof(hist));
            for(int i=0; i<BLOCK_SIZE; i++) {
                for(int j=0; j<BLOCK_SIZE; j++) {
                    for(int k=0; k<BINS; k++) {
                         hist[i][j][k] = 0.0f;
                    }
                }
            }
            
            int block_x_img = win_x + bx * pixels_per_cell;
            int block_y_img = win_y + by * pixels_per_cell;

            for (int py = 0; py < pixels_per_block; py++) {
                for (int px = 0; px < pixels_per_block; px++) {

                    int abs_x = block_x_img + px;
                    int abs_y = block_y_img + py;
                    
                    if (abs_x >= img_w || abs_y >= img_h) continue;
                    
                    int idx = abs_y * img_w + abs_x;
                
                    float m = mag[idx];
                    float a = angle[idx];
                    
                    if (m < 0.0001f) continue;

                    m *= gaussian_lut[(py * 16) + px]; 

                    float half_cell = pixels_per_cell / 2.0f;
                    float cell_x_continuous = (px - half_cell + 0.5f) / pixels_per_cell;
                    float cell_y_continuous = (py - half_cell + 0.5f) / pixels_per_cell;
                    float bin_continuous = a / angle_unit;

                    int icell_x = (int)floorf(cell_x_continuous);
                    int icell_y = (int)floorf(cell_y_continuous);
                    int ibin    = (int)floorf(bin_continuous);

                    float fcell_x = cell_x_continuous - icell_x;
                    float fcell_y = cell_y_continuous - icell_y;
                    float fbin    = bin_continuous - ibin;

                    float w_cell_x_0 = 1.0f - fcell_x;
                    float w_cell_y_0 = 1.0f - fcell_y;
                    float w_bin_0    = 1.0f - fbin;

                    float acc[2][2][2];

                    for (int ny = 0; ny < 2; ny++)
                        for (int nx = 0; nx < 2; nx++)
                            for (int nb = 0; nb < 2; nb++)
                                acc[ny][nx][nb] = 0.0f;

                    for (int n_y = 0; n_y <= 1; n_y++) {
                        for (int n_x = 0; n_x <= 1; n_x++) {
                            for (int n_b = 0; n_b <= 1; n_b++) {

                                int target_y = icell_y + n_y;
                                int target_x = icell_x + n_x;
                                // int target_b = ibin + n_b;

                                if (target_y >= 0 && target_y < cells_per_block &&
                                    target_x >= 0 && target_x < cells_per_block) {
                                    
                                    // int final_bin = target_b;
                                    // if (final_bin >= bins) final_bin -= bins;
                                    // if (final_bin < 0) final_bin += bins;

                                    float weight_y = (n_y == 0) ? w_cell_y_0 : fcell_y;
                                    float weight_x = (n_x == 0) ? w_cell_x_0 : fcell_x;
                                    float weight_b = (n_b == 0) ? w_bin_0    : fbin;

                                    //hist[(target_y*BLOCK_SIZE*BINS)+(target_x*BINS)+final_bin] += m * weight_x * weight_y * weight_b;
                                    //hist[target_y][target_x][final_bin] += m * weight_x * weight_y * weight_b;
                                    acc[n_y][n_x][n_b] += m * weight_x * weight_y * weight_b;
                                }
                            }
                        }
                    }


                    for (int ny = 0; ny < 2; ny++) {
                        for (int nx = 0; nx < 2; nx++) {
                            for (int nb = 0; nb < 2; nb++) {

                                int ty = icell_y + ny;
                                int tx = icell_x + nx;
                                int tb = ibin + nb;

                                if (tb >= bins) tb -= bins;

                                if (ty >= 0 && ty < cells_per_block &&
                                    tx >= 0 && tx < cells_per_block) {

                                    hist[ty][tx][tb] += acc[ny][nx][nb];
                                }
                            }
                        }
                    }

                } 
            } 

            float block_vec[36];

            int ptr = 0;
            for (int cy = 0; cy < cells_per_block; cy++) {
                for (int cx = 0; cx < cells_per_block; cx++) {
                    for (int b = 0; b < bins; b++) {
                        block_vec[ptr++] = hist[cy][cx][b];
                        //block_vec[ptr++] = hist[(cy*BLOCK_SIZE*BINS)+(cx*BINS)+b];
                    }
                }
            }

            apply_l2_hys(block_vec, 36);

            //int base = desc_idx;

            for (int k = 0; k < 36; k++) {

                 descriptor[desc_idx++] = block_vec[k];
            //     //descriptor[base + k] = block_vec[k];
            }




        } 
    } 
}


// ---------------------------------------------------------
// Single Scale Detection
// ---------------------------------------------------------
int detectSingleScaleStatic(float mag[IMG_WIDTH * IMG_HEIGHT], 
                            float angle[IMG_WIDTH * IMG_HEIGHT], 
                            int width, 
                            int height, 
                            double current_scale, 
                            float gaussian_lut[256],
                            Detection output[1000], 
                            int current_count) {
    
    int count_new = 0;
    int step_size = 8;
    


    if (width > IMG_WIDTH || height > IMG_HEIGHT) return 0;

    for (int y = 0; y <= height - HOG_WIN_HEIGHT; y += step_size) {
        for (int x = 0; x <= width - HOG_WIN_WIDTH; x += step_size) {
            
            if (current_count >= MAX_DETECTIONS) {
                return count_new; 
            }

            float hog_desc_local[DESCRIPTOR_SIZE];


            extractHOGFeature(mag, angle, width, height, x, y, gaussian_lut, hog_desc_local);
            //extractHOGFeature(width, height, x, y, hog_desc_local);

            float score = SVM_BIAS;
            for (int i = 0; i < DESCRIPTOR_SIZE; i++) {
                score += hog_desc_local[i] * SVM_WEIGHTS[i];
            }

            if (score > SVM_THRESHOLD) {

                Detection d;

                d.x = (float)(x * current_scale);
                d.y = (float)(y * current_scale);
                d.w = (float)(HOG_WIN_WIDTH  * current_scale);
                d.h = (float)(HOG_WIN_HEIGHT * current_scale);
                d.score = score;

                output[current_count] = d;
                
                current_count++;
                count_new++;
            }
        }
    }
    return count_new;
}

int hogDetectorStatic(unsigned char input[IMG_WIDTH * IMG_HEIGHT * 3], Detection output[1000]) {



    static unsigned char img_buf_1[IMG_WIDTH * IMG_HEIGHT * 3];
    static unsigned char img_buf_2[IMG_WIDTH * IMG_HEIGHT * 3];
    static float rgb_buf[IMG_WIDTH * IMG_HEIGHT * 3];
    static float mag_buf[IMG_WIDTH * IMG_HEIGHT];
    static float angle_buf[IMG_WIDTH * IMG_HEIGHT];

    memcpy(img_buf_1, input, IMG_WIDTH * IMG_HEIGHT * 3);

    unsigned char* current_img = img_buf_1;
    unsigned char* next_img    = img_buf_2;

    int cur_w = IMG_WIDTH;
    int cur_h = IMG_HEIGHT;
    double scale = INITIAL_SCALE;
    double scale_factor = SCALE_FACTOR;

    int total_dets = 0;

    while (cur_w >= HOG_WIN_WIDTH && cur_h >= HOG_WIN_HEIGHT) {
        
        
        hog_preprocess_rgb(current_img, cur_w, cur_h, rgb_buf); 
        
        computeGradientsRGB(rgb_buf, cur_w, cur_h, mag_buf, angle_buf); 

        // // --- DETECTION ---
        int n_new = detectSingleScaleStatic(mag_buf, angle_buf, cur_w, cur_h, scale, g_gaussian_lut,
                                             output, total_dets);
         total_dets += n_new;

        // // --- RESIZE ---
         int next_w = (int)(cur_w / scale_factor);
         int next_h = (int)(cur_h / scale_factor);
        
         if (next_w < HOG_WIN_WIDTH || next_h < HOG_WIN_HEIGHT) break;

        resizeBilinearPlanar(current_img, cur_w, cur_h, next_img, next_w, next_h);

        // Swap Pointers
        unsigned char* tmp = current_img;
        current_img = next_img;
        next_img = tmp;

        cur_w = next_w; cur_h = next_h;
        scale *= scale_factor;
    }


     return total_dets;
}

