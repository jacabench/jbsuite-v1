#include "hog.h"
#include "weights.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define PI 3.14159265f

void resizeBilinearPlanar(const unsigned char* src, int src_w, int src_h, unsigned char* dst, int dst_w, int dst_h) {
    
    float x_ratio = ((float)(src_w - 1)) / dst_w;
    float y_ratio = ((float)(src_h - 1)) / dst_h;
    
    int src_plane_size = src_w * src_h;
    int dst_plane_size = dst_w * dst_h;

    for (int c = 0; c < 3; c++) {
        const unsigned char* src_plane = src + (c * src_plane_size);
        unsigned char* dst_plane = dst + (c * dst_plane_size);

        for (int i = 0; i < dst_h; i++) {
            for (int j = 0; j < dst_w; j++) {
                
                int x_l = (int)(x_ratio * j);
                int y_l = (int)(y_ratio * i);
                int x_h = (x_l + 1 < src_w) ? (x_l + 1) : x_l;
                int y_h = (y_l + 1 < src_h) ? (y_l + 1) : y_l;
                
                float x_diff = (x_ratio * j) - x_l;
                float y_diff = (y_ratio * i) - y_l;
                
                unsigned char A = src_plane[y_l * src_w + x_l];
                unsigned char B = src_plane[y_l * src_w + x_h];
                unsigned char C = src_plane[y_h * src_w + x_l];
                unsigned char D = src_plane[y_h * src_w + x_h];

                float pixel = A * (1 - x_diff) * (1 - y_diff) + 
                              B * (x_diff) * (1 - y_diff) +
                              C * (y_diff) * (1 - x_diff) + 
                              D * (x_diff * y_diff);

                dst_plane[i * dst_w + j] = (unsigned char)pixel;
            }
        }
    }
}


void apply_l2_hys(float* block_vector, int size) {
    float sum_sq = 0.0f;
    float eps = 0.001f; 

    for (int i = 0; i < size; i++) {
        sum_sq += block_vector[i] * block_vector[i];
    }
    float scale = 1.0f / sqrtf(sum_sq + eps * eps);
    
    for (int i = 0; i < size; i++) {
        block_vector[i] *= scale;
    }

    sum_sq = 0.0f; 
    for (int i = 0; i < size; i++) {
        if (block_vector[i] > 0.2f) {
            block_vector[i] = 0.2f;
        }
        sum_sq += block_vector[i] * block_vector[i];
    }

    scale = 1.0f / sqrtf(sum_sq + eps * eps);
    for (int i = 0; i < size; i++) {
        block_vector[i] *= scale;
    }
}


void hog_preprocess_rgb(const unsigned char* planar_input, int width, int height, float* planar_float_img) {
    int num_pixels = width * height;
    
      int total_elements = num_pixels * 3;

    for (int i = 0; i < total_elements; i++) {
        float norm = planar_input[i] / 255.0f;
        planar_float_img[i] = sqrtf(norm); 
    }
}


void computeGradientsRGB(const float* input_planar_float, int w, int h, 
                         float* mag_out, float* angle_out) {
    
    int plane_size = w * h;
    const float* planes[3];
    planes[0] = input_planar_float;                  // R
    planes[1] = input_planar_float + plane_size;     // G
    planes[2] = input_planar_float + 2 * plane_size; // B


    for (int y = 0; y < h; y++) {
        
        int y_minus = (y > 0) ? y - 1 : 0;
        int y_plus  = (y < h - 1) ? y + 1 : h - 1;

        int row_offset       = y * w;
        int row_offset_minus = y_minus * w;
        int row_offset_plus  = y_plus * w;

        for (int x = 0; x < w; x++) {
            
            int x_minus = (x > 0) ? x - 1 : 0;
            int x_plus  = (x < w - 1) ? x + 1 : w - 1;
            
            float max_mag_sq = -1.0f;
            float best_dx = 0.0f;
            float best_dy = 0.0f;

            for (int c = 0; c < 3; c++) {
                const float* ptr = planes[c];
                
                float val_x_plus  = ptr[row_offset + x_plus];
                float val_x_minus = ptr[row_offset + x_minus];
                float dx = val_x_plus - val_x_minus;

                float val_y_plus  = ptr[row_offset_plus + x];
                float val_y_minus = ptr[row_offset_minus + x];
                float dy = val_y_plus - val_y_minus;

                float mag_sq = dx * dx + dy * dy;

                if (mag_sq > max_mag_sq) {
                    max_mag_sq = mag_sq;
                    best_dx = dx;
                    best_dy = dy;
                }
            }

            int idx = row_offset + x;
            mag_out[idx] = sqrtf(max_mag_sq);

            float ang = atan2f(best_dy, best_dx) * (180.0f / PI);
            if (ang < 0) ang += 180.0f;
            if (ang >= 180.0f) ang -= 180.0f;
            angle_out[idx] = ang;
        }
    }
}


void extractHOGFeature(float* mag, float* angle, int img_w, int img_h, int win_x, int win_y, float* descriptor) {
    
    int desc_idx = 0;
    
    int cells_per_block = BLOCK_SIZE; // 2
    int pixels_per_cell = CELL_SIZE;  // 8
    int pixels_per_block = cells_per_block * pixels_per_cell; // 16
    int bins = BINS;                  // 9
    float angle_unit = 180.0f / bins; 

    static float gaussian_lut[16][16];
    static int lut_initialized = 0;

    if (!lut_initialized) {
        float sigma = 0.5f * pixels_per_block; // 8.0
        float sigma_sq_2 = 2.0f * sigma * sigma;
        float center = (pixels_per_block - 1) / 2.0f; 

        for (int y = 0; y < pixels_per_block; y++) {
            for (int x = 0; x < pixels_per_block; x++) {
                float dy = y - center;
                float dx = x - center;
                float dist_sq = dx*dx + dy*dy;
                
                gaussian_lut[y][x] = expf(-dist_sq / sigma_sq_2);
            }
        }
        lut_initialized = 1;
    }

    for (int by = 0; by < BLOCKS_Y; by++) {
        for (int bx = 0; bx < BLOCKS_X; bx++) {
            
            float hist[BLOCK_SIZE][BLOCK_SIZE][BINS];
            memset(hist, 0, sizeof(hist));
            
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

                    m *= gaussian_lut[py][px]; 

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

                    for (int n_y = 0; n_y <= 1; n_y++) {
                        for (int n_x = 0; n_x <= 1; n_x++) {
                            for (int n_b = 0; n_b <= 1; n_b++) {
                                
                                int target_y = icell_y + n_y;
                                int target_x = icell_x + n_x;
                                int target_b = ibin + n_b;

                                if (target_y >= 0 && target_y < cells_per_block &&
                                    target_x >= 0 && target_x < cells_per_block) {
                                    
                                    int final_bin = target_b;
                                    if (final_bin >= bins) final_bin -= bins;
                                    if (final_bin < 0) final_bin += bins;

                                    float weight_y = (n_y == 0) ? w_cell_y_0 : fcell_y;
                                    float weight_x = (n_x == 0) ? w_cell_x_0 : fcell_x;
                                    float weight_b = (n_b == 0) ? w_bin_0    : fbin;

                                    hist[target_y][target_x][final_bin] += m * weight_x * weight_y * weight_b;
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
                    }
                }
            }

            apply_l2_hys(block_vec, 36);

            for (int k = 0; k < 36; k++) {
                descriptor[desc_idx++] = block_vec[k];
            }
        } 
    } 
}



int detectSingleScale(float* mag, float* angle, int width, int height, double current_scale, Detection* all_detections, int* total_count) {
    
    float* descriptor = (float*)malloc(DESCRIPTOR_SIZE * sizeof(float));
    
    int count_new = 0;
    int step_size = 8;

    for (int y = 0; y <= height - HOG_WIN_HEIGHT; y += step_size) {
        for (int x = 0; x <= width - HOG_WIN_WIDTH; x += step_size) {
            
            extractHOGFeature(mag, angle, width, height, x, y, descriptor);

            float score = SVM_BIAS;
            for (int i = 0; i < DESCRIPTOR_SIZE; i++) {
                score += descriptor[i] * SVM_WEIGHTS[i];
            }

            if (score > SVM_THRESHOLD) {
                
                // (*all_detections)[*total_count].x = (float)(x * current_scale);
                // (*all_detections)[*total_count].y = (float)(y * current_scale);
                // (*all_detections)[*total_count].w = (float)(HOG_WIN_WIDTH * current_scale);
                // (*all_detections)[*total_count].h = (float)(HOG_WIN_HEIGHT * current_scale);
                // (*all_detections)[*total_count].score = score;

                all_detections[*total_count].x = (float)(x * current_scale);
                all_detections[*total_count].y = (float)(y * current_scale);
                all_detections[*total_count].w = (float)(HOG_WIN_WIDTH * current_scale);
                all_detections[*total_count].h = (float)(HOG_WIN_HEIGHT * current_scale);
                all_detections[*total_count].score = score;
                
                (*total_count)++;
                count_new++;
            }
        }
    }
    
    free(descriptor);
    return count_new;
}



// --- Main pipeline function---
int hogDetector(unsigned char* input, int width, int height, Detection* output){

    //Detection* detections = (Detection*)malloc(MAX_DETECTIONS * sizeof(Detection));

    unsigned char* current_img = (unsigned char*)malloc(width * height * 3);
    memcpy(current_img, input, width * height * 3);

    int current_w = width;
    int current_h = height;
    double scale_factor = SCALE_FACTOR;
    double current_scale = INITIAL_SCALE;
    
    int total_dets = 0;

    while (current_w >= HOG_WIN_WIDTH && current_h >= HOG_WIN_HEIGHT) {
        
        float* rgb_buf = (float*)malloc(current_w * current_h * 3 * sizeof(float));     
        float* mag_buf = (float*)malloc(current_w * current_h * sizeof(float));
        float* ang_buf = (float*)malloc(current_w * current_h * sizeof(float));
        
        hog_preprocess_rgb(current_img, current_w, current_h, rgb_buf);
        
        computeGradientsRGB(rgb_buf, current_w, current_h, mag_buf, ang_buf);
        
        // DETECTION
        detectSingleScale(mag_buf, ang_buf, current_w, current_h, current_scale, output, &total_dets);
        
        free(rgb_buf);       
        free(mag_buf);
        free(ang_buf);

        int next_w = (int)(current_w / scale_factor);
        int next_h = (int)(current_h / scale_factor);

        if (next_w < HOG_WIN_WIDTH || next_h < HOG_WIN_HEIGHT) break;

        unsigned char* next_img = (unsigned char*)malloc(next_w * next_h * 3);
        
        resizeBilinearPlanar(current_img, current_w, current_h, next_img, next_w, next_h);
        
        free(current_img);
        current_img = next_img;
        current_w = next_w;
        current_h = next_h;
        current_scale *= scale_factor;
    }

    free(current_img); // Libera a última imagem redimensionada

    //output = detections;
    
    return total_dets;
}
