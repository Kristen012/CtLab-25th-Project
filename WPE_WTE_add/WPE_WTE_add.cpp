#include "WPE_WTE_add.hpp"

int main() {
    int embedding_size = 768;
    int num_wte_tokens = 50257;
    int num_wpe_positions = 1024;

    // Load weights from the txt files (now 1D and reshaped into 2D)
    std::vector<std::vector<float>> wte = loadWeightsFromTXT("transformer.wte.weight.txt", num_wte_tokens, embedding_size);
    std::vector<std::vector<float>> wpe = loadWeightsFromTXT("transformer.wpe.weight.txt", num_wpe_positions, embedding_size);

    // File paths for input files
    std::vector<std::string> input_files = {
        ".\\input_ids_before_embedding\\iter_1.txt",
        ".\\input_ids_before_embedding\\iter_2.txt",
        ".\\input_ids_before_embedding\\iter_3.txt"
    };

    // Process each input file
    int index = 0;
    for (const auto& input_file : input_files) {
        // Read tokenized input
        std::vector<int> tokenized_input = readTokenizedInput(input_file);

        // Perform the sum operation between WTE and WPE
        // printf("index : %d\n",index);
        std::vector<std::vector<float>> result = sumWPEAndWTE(tokenized_input, wte, wpe,index);        

        index+=(int)result.size();
        // Generate output file name based on input file
        std::string output_file = input_file.substr(input_file.find_last_of("\\") + 1);
        output_file = "output_" + output_file; // Example: output_iter_1.txt

        // Save the result to output file
        saveToTXT(result, output_file);
    }

    return 0;
}