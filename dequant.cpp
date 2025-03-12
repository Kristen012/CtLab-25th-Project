#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <string>
#include <limits>
// #include <cuda_fp16.h>
#include <time.h>
using namespace std;

// Function to load CSV data into a 2D vector
vector<vector<int>> loadCSV_int(const string& filename) {
    vector<vector<int>> data;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        vector<int> row;
        string value;
        // cout << line;
        while (getline(ss, value, ',')) {
            row.push_back(static_cast<int> (stod(value)));
            // cout << static_cast<int> (stod(value)) << " ";
        }

        data.push_back(row);
    }
    
    return data;
}
vector<vector<int>> loadCSV_intp(const string& filename) {
    vector<vector<int>> data;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        vector<int> row;
        string value;
        while (getline(ss, value, ',')) {
            row.push_back(stoi(value));
        }
        data.push_back(row);
    }
    
    return data;
}

vector<vector<float>> loadCSV_float(const string& filename) {
    vector<vector<float>> data;
    ifstream file(filename);
    string line;
    
    while (getline(file, line)) {
        stringstream ss(line);
        vector<float> row;
        string value;
        
        while (getline(ss, value, ',')) {
            row.push_back(stof(value));
        }
        
        data.push_back(row);
    }
    
    return data;
}

// Function to reshape a 2D vector into another 2D vector with given dimensions
// vector<vector<int>> reshape(const vector<vector<int>>& input, int new_rows, int new_cols) {
//     vector<vector<int>> reshaped(new_rows, vector<int>(new_cols));
//     int current_row = 0, current_col = 0;
    
//     for (const auto& row : input) {
//         for (const auto& val : row) {
//             reshaped[current_row][current_col] = val;
//             current_col++;
//             if (current_col == new_cols) {
//                 current_col = 0;
//                 current_row++;
//             }
//         }
//     }
    
//     return reshaped;
// }
void writeToCSV(const std::vector<std::vector<int>>& data, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (const auto& row : data) {
            for (size_t i = 0; i < row.size(); ++i) {
                file << row[i];
                if (i < row.size() - 1) {
                    file << ",";  // Add comma between values, but not after the last one
                }
            }
            file << "\n";  // Newline at the end of each row
        }
        file.close();
        std::cout << "File saved successfully!" << std::endl;
    } else {
        std::cerr << "Unable to open file" << std::endl;
    }
}
vector<vector<float>> dequantize(int bits, int group_size, const vector<vector<int>>& qzeros, const vector<vector<float>>& scales, const vector<vector<int>>& wf, const vector<vector<int>>& qweight, int file_index) {
    // Initialize zeros vector
    int num_rows = qzeros.size();
    int num_cols = qzeros[0].size() * (32 / bits);
    vector<int> sh = {0,4,8,12,16,20,24,28};
    vector<vector<int>> zeros(num_rows, vector<int>(num_cols));
    for (size_t i = 0; i < qzeros.size(); ++i) {
        for (size_t j = 0; j < qzeros[i].size(); ++j) {
            for (int k = 0; k < 32 / bits; ++k) {
                // int shift = wf[0][k];
                zeros[i][j * (32 / bits) + k] = (((qzeros[i][j] >> sh[k]) + 1) & ((1 << bits) - 1));
            }
        }
    }
    
    // Initialize weight vector
    num_rows = qweight.size() * (32 / bits);
    num_cols = qweight[0].size();
    vector<vector<int>> weight(num_rows, vector<int>(num_cols));

    for (size_t i = 0; i < qweight.size(); ++i) {
        for (size_t j = 0; j < qweight[i].size(); ++j) {
            for (int k = 0; k < 32 / bits; ++k) {
                // int shift = wf[0][k];
                weight[i* (32 / bits) + k][j] = ((qweight[i][j] >> sh[k]) & ((1 << bits) - 1));
                // if(i==0 && j==0) cout << weight[0][k];
            }
        }
    }
    // for(int i=0;i<8;i++){
        // for(int j=0;j<8;j++){
        //     cout << hex << weight[0][j];
        // }
    // cout << "\n";
    // }
    // for(int j=0;j<16;j++){
    //     cout << hex << zeros[0][j];
    // }
    // cout << "\n";
    // cout << "qweight: " << hex << weight[0][1];
    // cout << "; zero: " << hex << zeros[0][1];
    // cout << "; scale: " << scales[0][1] << "\n"; 

    // cout << "\n";
    // cout << "qweight: " << hex << weight[0][1] ;
    // cout << "; zero: " << hex << zeros[0][1];
    // cout << "; scale: " << scales[0][1] << "\n"; 
    // string reshape_name = "reshape_qweight/reshape_weight_" + to_string(file_index) + ".csv";
    // writeToCSV(weight, reshape_name);
    // Dequantize
    vector<vector<float>> dequantized_weight(weight.size(), vector<float>(weight[0].size()));

    for (size_t i = 0; i < weight.size(); ++i) {
        for (size_t j = 0; j < weight[i].size(); ++j) {
            dequantized_weight[i][j] = scales[i/group_size][j] * (weight[i][j] - zeros[i/group_size][j]);
        }
    }
    return dequantized_weight;
}

pair<float, float> compare_with_ans(const vector<vector<float>>& result, const string& ans_file) {
    vector<vector<float>> ans = loadCSV_float(ans_file);
    
    float avg_diff = 0.0;
    float percentage_diff = 0.0;
    float max_diff = -numeric_limits<float>::infinity();

    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = 0; j < result[i].size(); ++j) {
            float diff = abs(result[i][j] - ans[i][j]);
            avg_diff += diff;
            // if(ans[i][j] != 0) percentage_diff += diff / ans[i][j];
            max_diff = max(max_diff, diff);
        }
    }

    avg_diff /= (result.size() * result[0].size());
    percentage_diff /= (result.size() * result[0].size());
    cout << "percentage_diff: " << percentage_diff << endl;
    return {avg_diff, max_diff};
}

int main() {
    int bits = 4;
    int group_size = 128;

    for (int i = 1; i <= 48; ++i) {
        string qzeros_file = "quantized_weight/qzeros_" + to_string(i) + ".csv";
        string scales_file = "quantized_weight/scales_" + to_string(i) + ".csv";
        string wf_file = "quantized_weight/wf_" + to_string(i) + ".csv";
        string qweight_file = "quantized_weight/qweight_" + to_string(i) + ".csv";

        clock_t start, end;
        double cpu_time_used;
        
        vector<vector<int>> qzeros = loadCSV_int(qzeros_file);
        vector<vector<float>> scales = loadCSV_float(scales_file);
        vector<vector<int>> wf = loadCSV_int(wf_file);
        vector<vector<int>> qweight = loadCSV_int(qweight_file);
        // start = clock();

        vector<vector<float>> result = dequantize(bits, group_size, qzeros, scales, wf, qweight, i);

        // end = clock();
        // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        // cout << "time: " << cpu_time_used << endl;

        string ans_file = "weight/weight_" + to_string(i) + ".csv";
        pair<float, float> diffs = compare_with_ans(result, ans_file);
        float avg_diff = diffs.first;
        float max_diff = diffs.second;
        cout << "avg_diff for weight_" << i << ".csv is " << avg_diff << endl;
        cout << "max_diff for weight_" << i << ".csv is " << max_diff << endl;

        // if (avg_diff < 1e-4) {
        //     cout << "Result for weight_" << i << ".csv is correct." << endl;
        // } else {
        //     cout << "Result for weight_" << i << ".csv does not match." << endl;
        // }
    }

    return 0;
}
