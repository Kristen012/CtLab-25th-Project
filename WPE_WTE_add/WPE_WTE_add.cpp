#include "host.hpp"

int main() {
    int embedding_size = 768;

    std::vector<std::vector<double>> wte = loadWeights("transformer.wte.weight.csv", embedding_size);
    std::vector<std::vector<double>> wpe = loadWeights("transformer.wpe.weight.csv", embedding_size);

    std::vector<int> tokenized_input = readTokenizedInput("input_ids_before_embedding.txt");

    int num_tokens = wte.size();               
    /*
    WTE Shape: [50257, 768]
    WPE Shape: [1024, 768]
    */
    std::cout << "WTE Shape: [" << num_tokens << ", " << wte[0].size() << "]" << std::endl;
    std::cout << "WPE Shape: [" << wpe.size() << ", " << wpe[0].size() << "]" << std::endl;
    std::cout << "INPUT Shape: [" << tokenized_input.size() << "]" << std::endl;
    std::cout << "WPE Shape: [" << wpe.size() << ", " << wpe[0].size() << "]" << std::endl;

    std::cout << std::setprecision(20) << wpe[2][7] << std::endl; // Output with full precision
    
    std::vector<std::vector<double>> result = sumWPEAndWTE(tokenized_input, wte, wpe);

    std::cout << "Result Shape: [" << result.size() << ", " << result[0].size() << "]" << std::endl;

    saveToCSV(result, "output.csv");
    return 0;
}

