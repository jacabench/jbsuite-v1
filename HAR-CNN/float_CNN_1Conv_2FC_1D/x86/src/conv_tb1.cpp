#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include "conv_tb1.h"
#include "conv1.h"

// --- WEIGHT LOADING & CONVERSION FUNCTION ---
// Loads 32-bit floats from a binary file and converts them to type_t
int load_and_convert_weights(const char* file_path, type_t* dest_buffer, size_t num_elements) {
    FILE* fp = fopen(file_path, "rb");
    if (!fp) {
        printf("ERROR: Could not open file for reading: %s\n", file_path);
        return 0; // Failure
    }

    // Create a temporary buffer to hold the float data from the file
    float* temp_buffer = (float*) malloc(num_elements * sizeof(float));
    if (!temp_buffer) {
        printf("ERROR: Could not allocate memory for temporary float buffer.\n");
        fclose(fp);
        return 0; // Failure
    }

    // Read the entire block of floats
    size_t elements_read = fread(temp_buffer, sizeof(float), num_elements, fp);
    fclose(fp);

    if (elements_read != num_elements) {
        printf("ERROR: Expected to read %zu elements from %s, but got %zu.\n", num_elements, file_path, elements_read);
        free(temp_buffer);
        return 0; // Failure
    }

    // Convert floats to type_t by direct assignment. The ap_fixed library handles this.
    for (size_t i = 0; i < num_elements; ++i) {
        dest_buffer[i] = temp_buffer[i];
    }

    free(temp_buffer);
    printf("Successfully loaded and converted %zu elements from %s\n", num_elements, file_path);
    return 1; // Success
}

// --- DATA LOADING FUNCTION for Testbench ---
#include <fstream>
#include <sstream>
#include <vector>
int load_test_data(const std::string& file_path, std::vector<float>& features, std::vector<int>& labels, int num_features_expected) {
    std::ifstream file(file_path);
    if (!file.is_open()) {{ std::cerr << "ERROR: Could not open test data file: " << file_path << std::endl; return 0; }}
    std::string line;
    long num_samples = 0;
    while (std::getline(file, line)) {
        size_t features_start = line.find("{{") + 2;
        size_t features_end = line.find("},");
        size_t label_start = features_end + 2;
        size_t label_end = line.rfind("}");
        if (features_start == std::string::npos || features_end == std::string::npos || label_start == std::string::npos || label_end == std::string::npos) continue;
        std::string features_str = line.substr(features_start, features_end - features_start);
        std::string label_str = line.substr(label_start, label_end - label_start);
        if (label_str.empty()) continue;
        std::stringstream ss(features_str);
        std::string feature_val_str;
        int features_read = 0;
        size_t initial_feature_size = features.size();
        while (std::getline(ss, feature_val_str, ',')) {
            features.push_back(std::stof(feature_val_str));
            features_read++;
        }
        if (features_read != num_features_expected) {
            features.resize(initial_feature_size); // Roll back if feature count mismatches
            continue;
        }
        labels.push_back(std::stoi(label_str));
        num_samples++;
    }
    return num_samples;
}

int main(void) {
 	const size_t INPUT_BUFFER_SIZE = C1 * H1;
 	const size_t FEATURES_PER_SAMPLE = 30;
 	type_t *I1 = (type_t *) malloc(INPUT_BUFFER_SIZE * sizeof(type_t));
 	type_t *W1 = (type_t *) malloc(M1 * C1 * R1 * sizeof(type_t));
 	type_t *B1 = (type_t *) malloc(M1 * sizeof(type_t));
 	type_t *W_fc1 = (type_t *) malloc(5120 * sizeof(type_t));
 	type_t *B_fc1 = (type_t *) malloc(64 * sizeof(type_t));
 	type_t *W_fc2 = (type_t *) malloc(384 * sizeof(type_t));
 	type_t *B_fc2 = (type_t *) malloc(6 * sizeof(type_t));
 	float *O_final = (float *) calloc(N_CLASSES, sizeof(float));

 	// --- Load and Convert Trained Weights ---
 	int all_weights_loaded = 1;
 	all_weights_loaded &= load_and_convert_weights("W1.bin", W1, M1 * C1 * R1);
 	all_weights_loaded &= load_and_convert_weights("B1.bin", B1, M1);
 	all_weights_loaded &= load_and_convert_weights("W_fc1.bin", W_fc1, 5120);
 	all_weights_loaded &= load_and_convert_weights("B_fc1.bin", B_fc1, 64);
 	all_weights_loaded &= load_and_convert_weights("W_fc2.bin", W_fc2, 384);
 	all_weights_loaded &= load_and_convert_weights("B_fc2.bin", B_fc2, 6);
 	if (!all_weights_loaded) { return EXIT_FAILURE; }

 	// --- Load Real Test Data ---
 	std::vector<float> test_features;
 	std::vector<int> test_labels;
 	int num_samples = load_test_data("../../../dataset/test_windowed_channel3_windowsize10_class6.dat", test_features, test_labels, 30);
 	if (num_samples == 0) {
 	 	printf("FATAL: No samples loaded from test.dat. Exiting.\n");
 	 	return EXIT_FAILURE;
 	}

 	// --- Load Normalization Stats and Apply to Test Data ---
 	double train_mean = 0.0;
 	double train_std_dev = 1.0;
 	std::ifstream stats_file("norm_stats.txt");
 	if (stats_file.is_open()) {
 	 	stats_file >> train_mean;
 	 	stats_file >> train_std_dev;
 	 	stats_file.close();
 	 	printf("Successfully loaded stats from norm_stats.txt\n");
 	} else {
 	 	printf("ERROR: Could not open norm_stats.txt. Using default values.\n");
 	}
 	printf("Applying normalization (mean=%.3f, std_dev=%.3f) to %zu features...\n", train_mean, train_std_dev, test_features.size());
 	for (auto& val : test_features) {{
 	 	val = (val - train_mean) / train_std_dev;
 	}}
 	
 	// --- Run Inference on All Test Samples ---
 	int correct_predictions = 0;
 	printf("Running inference on %d test samples...\n", num_samples);
 	for (int i = 0; i < num_samples; ++i) {
 	 	// 1. Get a pointer to the start of the current sample's features
 	 	float* current_sample_features = &test_features[i * FEATURES_PER_SAMPLE];
 	 	// 2. Zero-out the entire input buffer to handle padding
 	 	memset(I1, 0, INPUT_BUFFER_SIZE * sizeof(type_t));
 	 	// 3. Copy and convert the available features into the start of the buffer
 	 	for (size_t j = 0; j < FEATURES_PER_SAMPLE; ++j) {
 	 	 	I1[j] = current_sample_features[j];
 	 	}

 	 	// 4. Perform CNN inference
 	 	cnn(I1, W1, B1, W_fc1, B_fc1, W_fc2, B_fc2, O_final);

 	 	// 5. Find the predicted class
 	 	float max_val = O_final[0];
 	 	int predicted_class = 0;
 	 	for (int k = 1; k < 6; k++) {
 	 	 	if (O_final[k] > max_val) {
 	 	 	 	max_val = O_final[k];
 	 	 	 	predicted_class = k;
 	 	 	}
 	 	}
 	 	// 6. Compare with the true label
 	 	int true_label = test_labels[i];
 	 	if (predicted_class == true_label) {
 	 	 	correct_predictions++;
 	 	}
 	}

 	// --- Report Final Accuracy ---
 	float accuracy = (float)correct_predictions / num_samples * 100.0f;
 	printf("\n--- HLS Model Verification Result ---\n");
 	printf("Correctly Classified: %d / %d\n", correct_predictions, num_samples);
 	printf("Accuracy: %.2f%%\n", accuracy);
 	printf("-------------------------------------\n");

 	// Free allocated memory
 	if(I1) free(I1);
 	if(W1) free(W1);
 	if(B1) free(B1);
 	if(W_fc1) free(W_fc1);
 	if(B_fc1) free(B_fc1);
 	if(W_fc2) free(W_fc2);
 	if(B_fc2) free(B_fc2);
 	if(O_final) free(O_final);

 	// Return success code (0) for C-sim pass, or failure (1) if needed
 	return EXIT_SUCCESS;
}