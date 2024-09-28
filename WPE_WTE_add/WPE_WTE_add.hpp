#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

// Function to load 1D weights from a txt file and reshape them into 2D
std::vector<float> load1DWeights(const std::string& filePath) {
    std::vector<float> weights;
    std::ifstream inFile(filePath);

    if (inFile.is_open()) {
        float value;
        while (inFile >> value) { // Read each value into the vector
            weights.push_back(value);
        }
        inFile.close();
    } else {
        std::cerr << "Unable to open file " << filePath << std::endl;
    }

    return weights;
}

std::vector<std::vector<float>> loadWeightsFromTXT(const std::string& filePath, int rows, int cols) {
    std::vector<float> flat_weights = load1DWeights(filePath);
    std::vector<std::vector<float>> reshaped_weights(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            reshaped_weights[i][j] = flat_weights[i * cols + j];
        }
    }

    return reshaped_weights;
}

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
                                             const std::vector<std::vector<float>>& wpe) {
    int embedding_size = wte[0].size();
    int sequence_length = tokenized_input.size();
    
    std::vector<std::vector<float>> result(sequence_length, std::vector<float>(embedding_size, 0.0f));

    for (int i = 0; i < sequence_length; ++i) {
        int token_id = tokenized_input[i];
        for (int j = 0; j < embedding_size; ++j) {
            result[i][j] = wte[token_id][j] + wpe[i][j];
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
