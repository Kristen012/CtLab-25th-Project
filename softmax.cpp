#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<float> read_data(const std::string& file_path) {
    std::ifstream file(file_path);
    std::vector<float> data;
    float value;
    
    while (file >> value) {
        data.push_back(value);
    }

    return data;
}

std::vector<float> softmax(const std::vector<float>& x) {
    std::vector<float> ans;
    int size = 1024 * 768;
    int chunk_size = 768;

    for (int i = 0; i < size; i += chunk_size) {
        // float max_val = *std::max_element(x.begin() + i, x.begin() + i + chunk_size);
        float sum_exp = 0.0, max_val = x[i], pre_max = x[i];
        for (int j = 0; j < chunk_size; ++j) {
            pre_max = max_val;
            max_val = std::max(max_val, x[i + j]);
            sum_exp = sum_exp * std::exp(pre_max - max_val) + std::exp(x[i + j] - max_val);
            if(i == 0){
                printf("pre_max: %f max_val: %f \n",pre_max, max_val);
            }
        }
        if(i == 0){
            printf("%f\n",sum_exp);
        }
        for (int j = 0; j < chunk_size; ++j) {
            
            ans.push_back(std::exp(x[i + j] - max_val) / sum_exp);
        }
    }
    return ans;
}

void save_data(const std::string& file_path, const std::vector<float>& data) {
    std::ofstream file(file_path);
    for (float val : data) {
        file << val << "\n";
    }
}

int main() {
    std::string input_file = "random_floats.txt";
    std::string output_file = "expected_softmax_online_float.txt";

    std::vector<float> data = read_data(input_file);
    std::vector<float> calculated_softmax = softmax(data);
    save_data(output_file, calculated_softmax);

    return 0;
}
