#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>
#include "conv_tb3.h"
#include "conv3.h"

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
 	float *O_final = (float *) calloc(N_CLASSES, sizeof(float));


 	// --- Load Real Test Data ---
 	std::vector<float> test_features;
 	std::vector<int> test_labels;
 	int num_samples = load_test_data("test_windowed_channel3_windowsize10_class6.dat", test_features, test_labels, 30);
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
 	 	cnn(I1, O_final);

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
 	if(O_final) free(O_final);

 	// Return success code (0) for C-sim pass, or failure (1) if needed
 	return EXIT_SUCCESS;
}