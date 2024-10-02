#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iomanip>
#include <numeric>
#include <sstream>

class TopKLogitsWarper
{
public:
    TopKLogitsWarper(int top_k, float filter_value = -std::numeric_limits<float>::infinity(), int min_tokens_to_keep = 1)
    {
        if (top_k <= 0)
        {
            throw std::invalid_argument("`top_k` has to be a strictly positive integer.");
        }
        this->top_k = std::max(top_k, min_tokens_to_keep);
        this->filter_value = filter_value;
    }

    std::vector<std::vector<float>> operator()(const std::vector<std::vector<float>> &scores)
    {
        int num_columns = scores[0].size();
        int top_k = std::min(this->top_k, num_columns);

        std::vector<std::vector<float>> scores_processed(scores.size(), std::vector<float>(num_columns, 0));

        for (size_t i = 0; i < scores.size(); ++i)
        {
            // Get top-k values and their indices
            std::vector<float> row = scores[i];
            std::vector<int> indices(row.size());
            std::iota(indices.begin(), indices.end(), 0);

            std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                              [&](int a, int b)
                              { return row[a] > row[b]; });

            float threshold = row[indices[top_k - 1]];

            // Filter out values below the top-k threshold
            for (int j = 0; j < num_columns; ++j)
            {
                if (row[j] < threshold)
                {
                    scores_processed[i][j] = filter_value;
                }
                else
                {
                    scores_processed[i][j] = row[j];
                }
            }
        }

        return scores_processed;
    }

private:
    int top_k;
    float filter_value;
};

// std::vector<std::vector<float>> load_csv(const std::string &file_path, char delimiter)
// {
//     std::ifstream file(file_path);
//     std::vector<std::vector<float>> data;
//     std::string line;

//     while (std::getline(file, line))
//     {
//         std::stringstream ss(line);
//         std::string value;
//         std::vector<float> row;

//         while (std::getline(ss, value, delimiter))
//         {
//             row.push_back(std::stof(value));
//         }
//         data.push_back(row);
//     }
//     return data;
// }

std::vector<int> load_csv(const std::string &file_path, char delimiter)
{
    std::ifstream file(file_path);
    std::vector<int> data;
    std::string line;

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        while (std::getline(ss, value, delimiter))
        {
            data.push_back(std::stof(value));
        }
    }
    return data;
}

std::vector<std::vector<std::vector<float>>> load_csv_1d_to_3d(const std::string &file_path, int batch_size, int embedding_size)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    int total_lines = 0;
    std::string line;
    while (std::getline(file, line))
    {
        total_lines++;
    }

    int sequence_length = total_lines / (batch_size * embedding_size);
    if (sequence_length * batch_size * embedding_size != total_lines)
    {
        throw std::runtime_error("Inconsistent data: the number of lines is not divisible by batch_size * embedding_size.");
    }
    file.clear();
    file.seekg(0, std::ios::beg);

    std::vector<std::vector<std::vector<float>>> data_3d(batch_size, std::vector<std::vector<float>>(sequence_length, std::vector<float>(embedding_size)));

    int current_batch = 0, current_sequence = 0, current_embedding = 0;

    while (std::getline(file, line))
    {
        float value = std::stof(line);

        data_3d[current_batch][current_sequence][current_embedding] = value;

        current_embedding++;
        if (current_embedding == embedding_size)
        {
            current_embedding = 0;
            current_sequence++;
            if (current_sequence == sequence_length)
            {
                current_sequence = 0;
                current_batch++;
            }
        }

        if (current_batch == batch_size)
        {
            break;
        }
    }

    return data_3d;
}

// void save_tensor_to_file(const std::vector<std::vector<float>> &data, const std::string &file_path)
// {
//     std::ofstream file(file_path);
//     if (file.is_open())
//     {
//         file << std::fixed << std::setprecision(8);
//         for (const auto &row : data)
//         {
//             for (const auto &value : row)
//             {
//                 file << value << "\n";
//             }
//         }
//         file.close();
//         std::cout << "Data saved to " << file_path << " (one number per line)\n";
//     }
//     else
//     {
//         std::cerr << "Unable to open file: " << file_path << std::endl;
//     }
// }

void save_tensor_to_file(const std::vector<int> &data, const std::string &file_path)
{
    std::ofstream file(file_path);
    if (file.is_open())
    {
        file << std::fixed << std::setprecision(8);
        for (const auto &value : data)
        {
            file << value << "\n"; 
        }
        file.close();
        std::cout << "Data saved to " << file_path << " (one number per line)\n";
    }
    else
    {
        std::cerr << "Unable to open file: " << file_path << std::endl;
    }
}


// Function to calculate softmax
std::vector<float> softmax(const std::vector<float> &scores)
{
    std::vector<float> exp_scores(scores.size());
    float sum_exp = 0;

    for (size_t i = 0; i < scores.size(); ++i)
    {
        exp_scores[i] = std::exp(scores[i]);
        sum_exp += exp_scores[i];
    }

    for (size_t i = 0; i < scores.size(); ++i)
    {
        exp_scores[i] /= sum_exp;
    }

    return exp_scores;
}

std::vector<std::vector<float>> extract_last_token_logits(const std::vector<float> &sample_input)
{
    int num_tokens = 50257;
    std::vector<std::vector<float>> next_token_logits;

    if (sample_input.size() >= num_tokens)
    {
        std::vector<float> last_tokens(sample_input.end() - num_tokens, sample_input.end());
        next_token_logits.push_back(last_tokens);
    }
    else
    {
        throw std::runtime_error("Insufficient tokens in sample_input");
    }

    return next_token_logits;
}
// std::vector<std::vector<float>> extract_last_token_logits(const std::vector<std::vector<std::vector<float>>> &sample_input)
// {
//     std::vector<std::vector<float>> next_token_logits;

//     for (const auto &row : sample_input)
//     {
//         next_token_logits.push_back(row.back());
//     }

//     return next_token_logits;
// }

int get_iteration_number(const std::string &file_path)
{
    size_t iter_pos = file_path.find("iter_");

    if (iter_pos != std::string::npos)
    {
        std::string iter_str = file_path.substr(iter_pos + 5);

        size_t end_pos = iter_str.find_first_not_of("0123456789");
        iter_str = iter_str.substr(0, end_pos);

        return std::stoi(iter_str);
    }
    return -1;
}

std::string get_file_by_iteration(const std::string &base_file_path, int iteration)
{
    std::string file_path = base_file_path + "_iter_" + std::to_string(iteration) + ".txt";

    return file_path;
}

std::vector<float> load_csv_1d(const std::string &file_path)
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + file_path);
    }

    std::vector<float> data_1d;
    std::string line;

    while (std::getline(file, line))
    {
        float value = std::stof(line);
        data_1d.push_back(value);   
    }

    return data_1d;
}

void sample(std::vector<float>& sample_input, std::vector<int>& input_ids, int iter)
{
    // int batch_size = 1;
    // int embedding_size = 50257;
    // std::string sample_input_file_path = "/home/fenyu/vlsi/activations/sample_input/iter_1.txt";
    // int iter = get_iteration_number(sample_input_file_path);
    // std::cout << "iter: " << iter << std::endl;
    // // auto sample_input = load_csv_1d_to_3d(sample_input_file_path, batch_size, embedding_size);
    // auto sample_input = load_csv_1d(sample_input_file_path);

    // Load input_ids data
    // std::string input_ids_file_path = "";
    // if (iter == 1)
    // {
    //     input_ids_file_path = "/home/fenyu/vlsi/activations/input_ids_before_embedding/iter_1.txt";
    // }
    // else
    // {
    //     std::string base_file = "sample_output";
    //     input_ids_file_path = get_file_by_iteration(base_file, iter - 1);
    // }
    // auto input_ids = load_csv(input_ids_file_path, ',');

    // std::cout << "sample_input shape: " << sample_input.size() << " x " << sample_input[0].size() << " x " << sample_input[0][0].size() << std::endl;
    std::cout << "sample_input shape: " << sample_input.size() << std::endl;

    std::vector<std::vector<float>> next_token_logits = extract_last_token_logits(sample_input);

    const int top_k = 40;
    TopKLogitsWarper topk_warper(top_k);
    auto scores_processed = topk_warper(next_token_logits);

    std::vector<float> probs = softmax(scores_processed[0]);

    // Get index of the max value
    auto max_it = std::max_element(probs.begin(), probs.end());
    int next_token = std::distance(probs.begin(), max_it);

    std::cout << "Next token: " << next_token << std::endl;

    input_ids.push_back(next_token);
    std::string base_file = "/home/ywtang23/Data/sample_output";
    std::string file_path = get_file_by_iteration(base_file, iter);
    save_tensor_to_file(input_ids, file_path);

    return;
}
