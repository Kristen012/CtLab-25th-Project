#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>
#include <time.h>

// Function to load 1D weights from a txt file and reshape them into 2D
// std::vector<float> load1DWeights(const std::string& filePath) {
//     std::vector<float> weights;
//     std::ifstream inFile(filePath);

//     if (inFile.is_open()) {
//         float value;
//         while (inFile >> value) { // Read each value into the vector
//             weights.push_back(value);
//         }
//         inFile.close();
//     } else {
//         std::cerr << "Unable to open file " << filePath << std::endl;
//     }

//     return weights;
// }

// std::vector<std::vector<float>> loadWeightsFromTXT(const std::string& filePath, int rows, int cols) {
//     std::vector<float> flat_weights = load1DWeights(filePath);
//     std::vector<std::vector<float>> reshaped_weights(rows, std::vector<float>(cols));

//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             reshaped_weights[i][j] = flat_weights[i * cols + j];
//         }
//     }

//     return reshaped_weights;
// }

// Function to save result as a flat 1D array in output.txt
void saveToTXT(const std::vector<std::vector<float>>& data, const std::string& filePath) {
    std::ofstream outFile(filePath);
    if (outFile.is_open()) {
        outFile << std::fixed << std::setprecision(13); // Set fixed-point and precision to 13
        for (const auto& row : data) {
            for (const auto& value : row) {
                outFile << value << "\n"; // Write each value on a new line
            }
        }
        outFile.close();
    } else {
        std::cerr << "Unable to open file " << filePath << std::endl;
    }
}

std::vector<std::vector<float>> sumWPEAndWTE(const std::vector<int>& tokenized_input,
                                             const std::vector<std::vector<float>>& wte,
                                             const std::vector<std::vector<float>>& wpe,int index) {
    int embedding_size = wte[0].size();
    int sequence_length = tokenized_input.size();

    std::vector<std::vector<float>> result(sequence_length, std::vector<float>(embedding_size, 0.0f));

    for (int i = 0; i < sequence_length; ++i) {
        int token_id = tokenized_input[i];
        for (int j = 0; j < embedding_size; ++j) {
            result[i][j] = wte[token_id][j] + wpe[index+i][j];
        }
    }

    return result;
}

std::vector<int> readTokenizedInput(const std::string& filename) {
    std::vector<int> tokenized_input;
    std::ifstream file(filename);
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            std::stringstream ss(line);
            int token_id;
            while (ss >> token_id) {
                tokenized_input.push_back(token_id);
                if (ss.peek() == ',') ss.ignore();
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }

    return tokenized_input;
}

// Function to save result as a flat 1D array in output.txt
void saveToTXT(const std::vector<std::vector<float>>& data, std::vector<float>& Array_ln_Data_in1) {
    // std::ofstream outFile(filePath);
    // if (outFile.is_open()) {
        // outFile << std::fixed << std::setprecision(13); // Set fixed-point and precision to 13
        for (const auto& row : data) {
            for (const auto& value : row) {
                Array_ln_Data_in1.push_back(value);
                // std::cout << "WPE_WTE_add" << value << std::endl;
                // outFile << value << "\n"; // Write each value on a new line
            }
        }
        // outFile.close();
    // } else {
        // std::cerr << "Unable to open file " << filePath << std::endl;
    // }
}
double WPE_WTE_add(std::vector<int>& input_vec, std::vector<float>& Array_ln_Data_in1, int index) {

    double total_time = 0;
    time_t start_time = clock();
    Array_ln_Data_in1.clear();
//    for (int i = 0; i<input_vec.size(); i++) {
//        std::cout << i << ": " << input_vec[i] << std::endl;
//    }

    int embedding_size = 768;
    int num_wte_tokens = 50257;
    int num_wpe_positions = 1024;

    // Load weights from the txt files (now 1D and reshaped into 2D)
    // std::vector<std::vector<float>> wte = loadWeightsFromTXT("/home/ywtang23/Data/weight/transformer.wte.weight.txt", num_wte_tokens, embedding_size);
    // std::vector<std::vector<float>> wpe = loadWeightsFromTXT("/home/ywtang23/Data/weight/transformer.wpe.weight.txt", num_wpe_positions, embedding_size);
    extern std::vector<std::vector<float>> wte;
    extern std::vector<std::vector<float>> wpe;
    // File paths for input files
    // std::vector<std::string> input_files = {
    //     ".\\input_ids_before_embedding\\iter_1.txt",
    //     ".\\input_ids_before_embedding\\iter_2.txt",
    //     ".\\input_ids_before_embedding\\iter_3.txt"
    // };

    // Process each input file
    // int index = 0;
    // for (const auto& input_file : input_files) {
        // Read tokenized input
        // std::vector<int> tokenized_input = readTokenizedInput(input_file);

        // Perform the sum operation between WTE and WPE
        // printf("index : %d\n",index);
    std::vector<std::vector<float>> result = sumWPEAndWTE(input_vec, wte, wpe,index);

    index+=(int)result.size();
        // Generate output file name based on input file
        // std::string output_file = input_file.substr(input_file.find_last_of("\\") + 1);
        // output_file = "output_" + output_file; // Example: output_iter_1.txt

        // Save the result to output file
    saveToTXT(result, Array_ln_Data_in1);
    time_t end_time = clock();
    total_time += ((double)(end_time-start_time) / CLOCKS_PER_SEC);
    // }
//    cout << "Done WPE WTE ADD" << endl;
    return total_time;

}
