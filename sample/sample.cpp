#include "sample.hpp"

int main()
{
    int batch_size = 1;
    int embedding_size = 50257;
    std::string sample_input_file_path = "/home/fenyu/vlsi/activations/sample_input/iter_3.txt";
    int iter = get_iteration_number(sample_input_file_path);
    std::cout << "iter: " << iter << std::endl;
    auto sample_input = load_csv_1d_to_3d(sample_input_file_path, batch_size, embedding_size);

    // Load input_ids data
    std::string input_ids_file_path = "";
    if (iter == 1)
    {
        input_ids_file_path = "/home/fenyu/vlsi/activations/input_ids_before_embedding/iter_1.txt";
    }
    else
    {
        std::string base_file = "sample_output";
        input_ids_file_path = get_file_by_iteration(base_file, iter - 1);
    }
    auto input_ids = load_csv(input_ids_file_path, ',');

    std::cout << "sample_input shape: " << sample_input.size() << " x " << sample_input[0].size() << " x " << sample_input[0][0].size() << std::endl;

    std::vector<std::vector<float>> next_token_logits = extract_last_token_logits(sample_input);
    save_tensor_to_file(next_token_logits, "next_token_logits.txt"); // same

    const int top_k = 40;
    TopKLogitsWarper topk_warper(top_k);
    auto scores_processed = topk_warper(next_token_logits);
    save_tensor_to_file(scores_processed, "scores_processed.txt"); // same with sample_score

    std::vector<float> probs = softmax(scores_processed[0]);

    // Get index of the max value
    auto max_it = std::max_element(probs.begin(), probs.end());
    int next_token = std::distance(probs.begin(), max_it);

    std::cout << "Next token: " << next_token << std::endl;

    input_ids.back().push_back(static_cast<float>(next_token));
    std::string base_file = "sample_output";
    std::string file_path = get_file_by_iteration(base_file, iter);
    save_tensor_to_file(input_ids, file_path);

    return 0;
}
