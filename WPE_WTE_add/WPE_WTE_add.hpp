#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <string>
#include <iomanip>

std::vector<std::vector<double>> loadWeights(const std::string& filename, int embedding_size) {
    std::ifstream file(filename);
    std::vector<std::vector<double>> weights;
    std::string line;
    
    // Ignore the first line (header)
    std::getline(file, line); 
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        
        weights.push_back(row);
    }
    
    return weights;
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

std::vector<std::vector<double>> sumWPEAndWTE(const std::vector<int>& tokenized_input, 
                                             const std::vector<std::vector<double>>& wte, 
                                             const std::vector<std::vector<double>>& wpe) {
    int embedding_size = wte[0].size();
    int sequence_length = tokenized_input.size();
    
    std::vector<std::vector<double>> result(sequence_length, std::vector<double>(embedding_size, 0.0f));

    for (int i = 0; i < sequence_length; ++i) {
        int token_id = tokenized_input[i];
        for (int j = 0; j < embedding_size; ++j) {
            result[i][j] = wte[token_id][j] + wpe[i][j];
        }
    }
    
    return result;
}

void saveToCSV(const std::vector<std::vector<double>>& data, const std::string& filename) {
    std::ofstream file(filename);

    // Set the desired precision for output
    file << std::fixed << std::setprecision(15); // Adjust the precision value as needed

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            file << row[i];
            if (i < row.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
}
