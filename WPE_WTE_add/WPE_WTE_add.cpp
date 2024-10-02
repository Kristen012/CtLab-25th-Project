// #include "WPE_WTE_add.hpp"

// int main() {
//     int embedding_size = 768;
//     int num_wte_tokens = 50257;
//     int num_wpe_positions = 1024;

//     // Load weights from the txt files (now 1D and reshaped into 2D)
//     std::vector<std::vector<float>> wte = loadWeightsFromTXT("transformer.wte.weight.txt", num_wte_tokens, embedding_size);
//     std::vector<std::vector<float>> wpe = loadWeightsFromTXT("transformer.wpe.weight.txt", num_wpe_positions, embedding_size);

//     // Read tokenized input
//     std::vector<int> tokenized_input = readTokenizedInput("input_ids_before_embedding.txt");

//     // // Print shapes
//     // std::cout << "WTE Shape: [" << wte.size() << ", " << wte[0].size() << "]" << std::endl;
//     // std::cout << "WPE Shape: [" << wpe.size() << ", " << wpe[0].size() << "]" << std::endl;
//     // std::cout << "INPUT Shape: [" << tokenized_input.size() << "]" << std::endl;
    
//     // // Perform the sum operation between WTE and WPE
//     std::vector<std::vector<float>> result = sumWPEAndWTE(tokenized_input, wte, wpe);

//     // // Print result shape
//     // std::cout << "Result Shape: [" << result.size() << ", " << result[0].size() << "]" << std::endl;

//     // Save the result to output.txt
//     saveToTXT(result, "output.txt");
// }
