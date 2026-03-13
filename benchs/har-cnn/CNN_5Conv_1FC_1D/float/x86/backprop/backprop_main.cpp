#include "backprop.h"
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <numeric>
using namespace std;

// --- DATA LOADING FUNCTION ---
#include <fstream>
#include <sstream>
#include <vector>
long load_data(const std::string& file_path, std::vector<dtype_t>& X_out, std::vector<dtype_t>& Y_out_one_hot, std::vector<size_t>& Y_out_raw, int N_CLASSES, int num_features) {
    std::ifstream file(file_path);
    if (!file.is_open()) { std::cerr << "ERROR: Could not open file: " << file_path << std::endl; return 0; }
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
        std::vector<dtype_t> current_features;
        while (std::getline(ss, feature_val_str, ',')) { current_features.push_back(std::stof(feature_val_str)); }
        if (current_features.size() != num_features) {
            std::cerr << "WARNING: Mismatch in feature count. Expected " << num_features << ", got " << current_features.size() << ". Skipping sample." << std::endl;
            continue;
        }
        X_out.insert(X_out.end(), current_features.begin(), current_features.end());
        int label = std::stoi(label_str);
        Y_out_raw.push_back(label);
        std::vector<dtype_t> one_hot(N_CLASSES, 0.0f);
        if (label >= 0 && label < N_CLASSES) { one_hot[label] = 1.0f; }
        Y_out_one_hot.insert(Y_out_one_hot.end(), one_hot.begin(), one_hot.end());
        num_samples++;
    }
    std::cout << "Successfully loaded " << num_samples << " samples from " << file_path << std::endl;
    return num_samples;
}
// --- WEIGHT SAVING FUNCTION ---
#include <fstream>

void save_weights(const std::string& file_path, const dtype_t* data, size_t num_elements) {
    std::ofstream out_file(file_path, std::ios::binary);
    if (!out_file.is_open()) {
        std::cerr << "ERROR: Could not open file for writing: " << file_path << std::endl;
        return;
    }
    // Write the raw bytes of the array to the file
    out_file.write(reinterpret_cast<const char*>(data), num_elements * sizeof(dtype_t));
    out_file.close();
    std::cout << "Saved " << num_elements << " elements to " << file_path << std::endl;
}

// --- DIAGNOSTIC HELPER FUNCTION ---
#include <numeric> // For std::inner_product

// Calculates the L2 Norm (magnitude) of a vector of floats.
dtype_t calculate_l2_norm(const dtype_t* data, size_t size) {
    dtype_t sum_sq = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        sum_sq += data[i] * data[i];
    }
    return sqrt(sum_sq);
}

// --- DATA NORMALIZATION FUNCTIONS ---
#include <numeric> // For std::accumulate

// 1. Calculates mean and stddev from a given dataset (should be TRAINING data)
void calculate_stats(const std::vector<dtype_t>& features, double& mean, double& std_dev) {
 	if (features.empty()) {
 	 	mean = 0.0;
 	 	std_dev = 1.0;
 	 	return;
 	}

 	// --- THIS IS THE FIX ---
 	double sum = 0.0;
 	// --- END FIX ---

 	sum = std::accumulate(features.begin(), features.end(), 0.0);
 	mean = sum / features.size();
 	double sq_sum = 0.0;
 	for(const auto& val : features) {
 	 	sq_sum += (val - mean) * (val - mean);
 	}
 	std_dev = std::sqrt(sq_sum / features.size());
 	std::cout << "Calculated Stats from Training Data: Mean=" << mean << ", StdDev=" << std_dev << std::endl;
}

// 2. Applies pre-calculated mean and stddev to a dataset
void apply_normalization(std::vector<dtype_t>& features, const double mean, const double std_dev) {
 	if (features.empty()) return;
 	if (std_dev > 1e-6) {
 	 	for(auto& val : features) {
 	 	 	val = (val - mean) / std_dev;
 	 	}
 	} else {
 	 	std::cout << "Warning: Standard deviation is near zero. Skipping normalization." << std::endl;
 	}
}

// --- ADAM OPTIMIZER FUNCTION ---
// Implements the Adam weight update rule.
void adam_update(dtype_t* W, const dtype_t* dW, dtype_t* m, dtype_t* v, size_t size,
                 dtype_t lr, dtype_t beta1, dtype_t beta2, dtype_t epsilon, size_t t) {
    // t is the 1-based timestep for bias correction
    dtype_t beta1_t = powf(beta1, t);
    dtype_t beta2_t = powf(beta2, t);

    for (size_t i = 0; i < size; ++i) {
        // Update biased first and second moment estimates
        m[i] = beta1 * m[i] + (1.0f - beta1) * dW[i];
        v[i] = beta2 * v[i] + (1.0f - beta2) * (dW[i] * dW[i]);

        // Compute bias-corrected moment estimates
        dtype_t m_hat = m[i] / (1.0f - beta1_t);
        dtype_t v_hat = v[i] / (1.0f - beta2_t);

        // Update weights
        W[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

int main() {
 	srand(time(0));
 	const dtype_t LEARNING_RATE = 0.0001f;
 	const dtype_t ADAM_BETA1 = 0.9f;
 	const dtype_t ADAM_BETA2 = 0.999f;
 	const dtype_t ADAM_EPSILON = 1e-8f;
 	const size_t NUM_EPOCHS = 5;
 	const size_t BATCH_SIZE = 32;
 	const std::string TRAIN_DATA_FILE = "../../../dataset/train_windowed_channel3_windowsize10_class6.dat";
 	const std::string TEST_DATA_FILE = "../../../dataset/test_windowed_channel3_windowsize10_class6.dat";

 	// --- Data Loading ---
 	std::vector<dtype_t> X_train_vec, Y_train_vec_one_hot, X_test_vec, Y_test_vec_one_hot;
 	std::vector<size_t> Y_train_vec_raw, Y_test_vec_raw;
 	long N_TRAIN_SAMPLES = load_data(TRAIN_DATA_FILE, X_train_vec, Y_train_vec_one_hot, Y_train_vec_raw, 6, 30);
 	long N_TEST_SAMPLES = load_data(TEST_DATA_FILE, X_test_vec, Y_test_vec_one_hot, Y_test_vec_raw, 6, 30);
 	if (N_TRAIN_SAMPLES == 0) { cerr << "FATAL ERROR: No training data loaded." << endl; return 1; }

 	// --- Correct Data Normalization ---
 	double train_mean, train_std_dev;
 	calculate_stats(X_train_vec, train_mean, train_std_dev);
 	cout << "Saving normalization stats to norm_stats.txt..." << endl;
 	std::ofstream stats_file("norm_stats.txt");
 	if (stats_file.is_open()) {
 	 	stats_file << train_mean << std::endl;
 	 	stats_file << train_std_dev << std::endl;
 	 	stats_file.close();
 	} else {
 	 	cerr << "ERROR: Could not open norm_stats.txt for writing." << endl;
 	}
 	apply_normalization(X_train_vec, train_mean, train_std_dev);
 	cout << "Applying same stats to test data..." << std::endl;
 	apply_normalization(X_test_vec, train_mean, train_std_dev);

 	// --- Get Raw Data Pointers ---
 	dtype_t* in_data_train = X_train_vec.data();
 	size_t* out_data_train_raw = Y_train_vec_raw.data();
 	dtype_t* in_data_test = X_test_vec.data();
 	size_t* out_data_test_raw = Y_test_vec_raw.data();

 	// --- Allocate Memory for CONV Layers ---
 	dtype_t W1[M1 * C1 * R1]; dtype_t B1[M1];
 	dtype_t dW1[M1 * C1 * R1]; dtype_t dB1[M1];
 	dtype_t m_W1[M1 * C1 * R1]; dtype_t v_W1[M1 * C1 * R1];
 	dtype_t m_B1[M1]; dtype_t v_B1[M1];
 	dtype_t O1[L1_FMAP_SIZE];
 	dtype_t W2[M2 * C2 * R2]; dtype_t B2[M2];
 	dtype_t dW2[M2 * C2 * R2]; dtype_t dB2[M2];
 	dtype_t m_W2[M2 * C2 * R2]; dtype_t v_W2[M2 * C2 * R2];
 	dtype_t m_B2[M2]; dtype_t v_B2[M2];
 	dtype_t O2[L2_FMAP_SIZE];
 	dtype_t W3[M3 * C3 * R3]; dtype_t B3[M3];
 	dtype_t dW3[M3 * C3 * R3]; dtype_t dB3[M3];
 	dtype_t m_W3[M3 * C3 * R3]; dtype_t v_W3[M3 * C3 * R3];
 	dtype_t m_B3[M3]; dtype_t v_B3[M3];
 	dtype_t O3[L3_FMAP_SIZE];
 	dtype_t W4[M4 * C4 * R4]; dtype_t B4[M4];
 	dtype_t dW4[M4 * C4 * R4]; dtype_t dB4[M4];
 	dtype_t m_W4[M4 * C4 * R4]; dtype_t v_W4[M4 * C4 * R4];
 	dtype_t m_B4[M4]; dtype_t v_B4[M4];
 	dtype_t O4[L4_FMAP_SIZE];
 	dtype_t W5[M5 * C5 * R5]; dtype_t B5[M5];
 	dtype_t dW5[M5 * C5 * R5]; dtype_t dB5[M5];
 	dtype_t m_W5[M5 * C5 * R5]; dtype_t v_W5[M5 * C5 * R5];
 	dtype_t m_B5[M5]; dtype_t v_B5[M5];
 	dtype_t O5[L5_FMAP_SIZE];
 	dtype_t O5_pool[L5_POOL_FMAP_SIZE];

 	// --- Allocate Memory for FC Layers ---
 	dtype_t W_fc1[960]; dtype_t B_fc1[6];
 	dtype_t dW_fc1[960]; dtype_t dB_fc1[6];
 	dtype_t m_W_fc1[960]; dtype_t v_W_fc1[960];
 	dtype_t m_B_fc1[6]; dtype_t v_B_fc1[6];
 	dtype_t O_fc1[6];

 	// --- Allocate Memory for Input & Intermediate Gradients ---
 	dtype_t padded_input[C1 * H1];
 	dtype_t dI_fc1[160];
 	dtype_t dI1[C1*H1*(GLOBAL_DIM==2?H1:1)];
 	dtype_t dI5[L4_FMAP_SIZE];
 	dtype_t dI4[L3_FMAP_SIZE];
 	dtype_t dI3[L2_FMAP_SIZE];
 	dtype_t dI2[L1_FMAP_SIZE];

 	// --- Allocate Memory for MaxPool Indices ---
 	int max_indices5[L5_POOL_FMAP_SIZE];

 	cout << "Initializing weights using He initialization..." << endl;
 	float w_bound_1 = sqrt(6.0f / 9);
 	for (size_t j = 0; j < M1*C1*R1; ++j) W1[j] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * w_bound_1;
 	for (size_t j = 0; j < M1; ++j) B1[j] = 0.0f;
 	float w_bound_2 = sqrt(6.0f / 18);
 	for (size_t j = 0; j < M2*C2*R2; ++j) W2[j] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * w_bound_2;
 	for (size_t j = 0; j < M2; ++j) B2[j] = 0.0f;
 	float w_bound_3 = sqrt(6.0f / 36);
 	for (size_t j = 0; j < M3*C3*R3; ++j) W3[j] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * w_bound_3;
 	for (size_t j = 0; j < M3; ++j) B3[j] = 0.0f;
 	float w_bound_4 = sqrt(6.0f / 48);
 	for (size_t j = 0; j < M4*C4*R4; ++j) W4[j] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * w_bound_4;
 	for (size_t j = 0; j < M4; ++j) B4[j] = 0.0f;
 	float w_bound_5 = sqrt(6.0f / 72);
 	for (size_t j = 0; j < M5*C5*R5; ++j) W5[j] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * w_bound_5;
 	for (size_t j = 0; j < M5; ++j) B5[j] = 0.0f;
 	float w_bound_fc1 = sqrt(6.0f / 160);
 	for (size_t j = 0; j < 960; ++j) W_fc1[j] = ((float)rand()/(float)RAND_MAX * 2.0f - 1.0f) * w_bound_fc1;
 	for (size_t j = 0; j < 6; ++j) B_fc1[j] = 0.0f;

 	cout << "Initializing Adam optimizer state..." << endl;
 	memset(m_W1, 0, sizeof(m_W1)); memset(v_W1, 0, sizeof(v_W1));
 	memset(m_B1, 0, sizeof(m_B1)); memset(v_B1, 0, sizeof(v_B1));
 	memset(m_W2, 0, sizeof(m_W2)); memset(v_W2, 0, sizeof(v_W2));
 	memset(m_B2, 0, sizeof(m_B2)); memset(v_B2, 0, sizeof(v_B2));
 	memset(m_W3, 0, sizeof(m_W3)); memset(v_W3, 0, sizeof(v_W3));
 	memset(m_B3, 0, sizeof(m_B3)); memset(v_B3, 0, sizeof(v_B3));
 	memset(m_W4, 0, sizeof(m_W4)); memset(v_W4, 0, sizeof(v_W4));
 	memset(m_B4, 0, sizeof(m_B4)); memset(v_B4, 0, sizeof(v_B4));
 	memset(m_W5, 0, sizeof(m_W5)); memset(v_W5, 0, sizeof(v_W5));
 	memset(m_B5, 0, sizeof(m_B5)); memset(v_B5, 0, sizeof(v_B5));
 	memset(m_W_fc1, 0, sizeof(m_W_fc1)); memset(v_W_fc1, 0, sizeof(v_W_fc1));
 	memset(m_B_fc1, 0, sizeof(m_B_fc1)); memset(v_B_fc1, 0, sizeof(v_B_fc1));

 	cout << "Starting training for " << NUM_EPOCHS << " epochs..." << endl;
 	std::vector<size_t> indices(N_TRAIN_SAMPLES); std::iota(indices.begin(), indices.end(), 0);
 	std::mt19937 g(rand());
 	size_t t = 0;

 	for (size_t epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
 	 	std::shuffle(indices.begin(), indices.end(), g);
 	 	long total_batches = (N_TRAIN_SAMPLES + BATCH_SIZE - 1) / BATCH_SIZE; int batch_count = 0;
 	 	for (long i = 0; i < N_TRAIN_SAMPLES; i += BATCH_SIZE) {
 	 	 	size_t current_batch_size = (i + BATCH_SIZE <= N_TRAIN_SAMPLES) ? BATCH_SIZE : (N_TRAIN_SAMPLES - i);
 	 	 	memset(dW1, 0, sizeof(dW1)); memset(dB1, 0, sizeof(dB1));
 	 	 	memset(dW2, 0, sizeof(dW2)); memset(dB2, 0, sizeof(dB2));
 	 	 	memset(dW3, 0, sizeof(dW3)); memset(dB3, 0, sizeof(dB3));
 	 	 	memset(dW4, 0, sizeof(dW4)); memset(dB4, 0, sizeof(dB4));
 	 	 	memset(dW5, 0, sizeof(dW5)); memset(dB5, 0, sizeof(dB5));
 	 	 	memset(dW_fc1, 0, sizeof(dW_fc1)); memset(dB_fc1, 0, sizeof(dB_fc1));

 	 	 	for (size_t sample_idx = 0; sample_idx < current_batch_size; ++sample_idx) {
 	 	 	 	size_t shuffled_index = indices[i + sample_idx];
 	 	 	 	dtype_t* current_sample_features = in_data_train + shuffled_index * 30;
 	 	 	 	size_t* current_sample_LABEL = out_data_train_raw + shuffled_index;
 	 	 	 	memset(padded_input, 0, sizeof(padded_input));
 	 	 	 	memcpy(padded_input, current_sample_features, 30 * sizeof(dtype_t));
 	 	 	 	cnn_fwd(padded_input, W1, B1, O1, W2, B2, O2, W3, B3, O3, W4, B4, O4, W5, B5, O5, O5_pool, W_fc1, B_fc1, O_fc1, max_indices5);
 	 	 	 	cnn_bwd(padded_input, current_sample_LABEL, O1, O2, O3, O4, O5, O5_pool, O_fc1, W1, dW1, dB1, W2, dW2, dB2, W3, dW3, dB3, W4, dW4, dB4, W5, dW5, dB5, W_fc1, dW_fc1, dB_fc1, dI_fc1, dI1, dI5, dI4, dI3, dI2, max_indices5);
 	 	 	}

 	 	 	t++;
 	 	 	adam_update(W_fc1, dW_fc1, m_W_fc1, v_W_fc1, 960, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(B_fc1, dB_fc1, m_B_fc1, v_B_fc1, 6, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(W5, dW5, m_W5, v_W5, M5*C5*R5, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(B5, dB5, m_B5, v_B5, M5, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(W4, dW4, m_W4, v_W4, M4*C4*R4, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(B4, dB4, m_B4, v_B4, M4, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(W3, dW3, m_W3, v_W3, M3*C3*R3, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(B3, dB3, m_B3, v_B3, M3, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(W2, dW2, m_W2, v_W2, M2*C2*R2, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(B2, dB2, m_B2, v_B2, M2, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(W1, dW1, m_W1, v_W1, M1*C1*R1, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	adam_update(B1, dB1, m_B1, v_B1, M1, LEARNING_RATE, ADAM_BETA1, ADAM_BETA2, ADAM_EPSILON, t);
 	 	 	batch_count++; if (batch_count % 50 == 0) { printf("Epoch %zu, Batch %d / %ld...\r", epoch + 1, batch_count, total_batches); fflush(stdout); }
 	 	} // End of batch loop

 	 	// --- EVALUATION on Test Set ---
 	 	if (N_TEST_SAMPLES > 0) {
 	 	 	int correct_predictions = 0;
 	 	 	for (long j = 0; j < N_TEST_SAMPLES; ++j) {
 	 	 	 	dtype_t* current_test_features = in_data_test + j * 30;
 	 	 	 	size_t true_label = out_data_test_raw[j];
 	 	 	 	memset(padded_input, 0, sizeof(padded_input));
 	 	 	 	memcpy(padded_input, current_test_features, 30 * sizeof(dtype_t));
 	 	 	 	cnn_fwd(padded_input, W1, B1, O1, W2, B2, O2, W3, B3, O3, W4, B4, O4, W5, B5, O5, O5_pool, W_fc1, B_fc1, O_fc1, max_indices5);
 	 	 	 	int predicted_class = 0; dtype_t max_score = O_fc1[0];
 	 	 	 	for (size_t k = 1; k < N_CLASSES; ++k) { if (O_fc1[k] > max_score) { max_score = O_fc1[k]; predicted_class = k; } }
 	 	 	 	if (predicted_class == true_label) { correct_predictions++; }
 	 	 	}
 	 	 	float test_accuracy = (float)correct_predictions / N_TEST_SAMPLES * 100.0f;
 	 	 	dtype_t grad_norm = calculate_l2_norm(dW_fc1, 960);
 	 	 	dtype_t weight_norm = calculate_l2_norm(W_fc1, 960);
 	 	 	printf("\nEpoch %zu | Test Accuracy: %.2f%% | Grad Norm: %e | Weight Norm: %.4f\n", epoch + 1, test_accuracy, grad_norm, weight_norm);
 	 	}
 	} // End of epoch loop

 	cout << "\nTraining finished." << endl;

 	// --- CALIBRATION: Find max absolute weight value ---
 	float max_abs_value = 0.0f;
 	// Helper lambda to check an array
 	auto find_max = [&](const dtype_t* arr, size_t size) {
 	 	for (size_t i = 0; i < size; ++i) {
 	 	 	if (fabs(arr[i]) > max_abs_value) {
 	 	 	 	max_abs_value = fabs(arr[i]);
 	 	 	}
 	 	}
 	};
 	find_max(W1, M1 * C1 * R1);
 	find_max(B1, M1);
 	find_max(W2, M2 * C2 * R2);
 	find_max(B2, M2);
 	find_max(W3, M3 * C3 * R3);
 	find_max(B3, M3);
 	find_max(W4, M4 * C4 * R4);
 	find_max(B4, M4);
 	find_max(W5, M5 * C5 * R5);
 	find_max(B5, M5);
 	find_max(W_fc1, 960);
 	find_max(B_fc1, 6);
 	cout << "CALIBRATION_INFO: Maximum absolute weight value is: " << max_abs_value << endl;
 	cout << "Saving final model weights..." << endl;
 	save_weights("W1.bin", W1, M1 * C1 * R1);
 	save_weights("B1.bin", B1, M1);
 	save_weights("W2.bin", W2, M2 * C2 * R2);
 	save_weights("B2.bin", B2, M2);
 	save_weights("W3.bin", W3, M3 * C3 * R3);
 	save_weights("B3.bin", B3, M3);
 	save_weights("W4.bin", W4, M4 * C4 * R4);
 	save_weights("B4.bin", B4, M4);
 	save_weights("W5.bin", W5, M5 * C5 * R5);
 	save_weights("B5.bin", B5, M5);
 	save_weights("W_fc1.bin", W_fc1, 960);
 	save_weights("B_fc1.bin", B_fc1, 6);

 	return 0;
}