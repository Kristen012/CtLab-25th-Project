#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <regex>
#include <sstream>
#include <iomanip>
#include <utility>
#include <stdexcept>


class tokenizer_t {
private:
    std::map<std::string, int> encoder;
    std::map<int, std::string> decoder;
    std::vector<std::pair<std::string, std::string>> merge_ranks;
    std::regex regex_splitter;
    std::map<uint8_t, char32_t> byte_encoder;

    std::u32string utf8_to_utf32(const std::string& str);
    std::string utf32_to_utf8(const std::u32string& str);
    int get_pair_rank(const std::string& first, const std::string& second);
    std::map<uint8_t, char32_t> bytes_to_unicode();
    std::vector<std::string> bpe(const std::u32string& input);

public:
    tokenizer_t(const std::string& vocab_file, const std::string& merges_file);
    std::vector<int> tokenize(const std::string& text);
    std::vector<std::string> detokenize(const std::vector<int>& tokens);
};

void die(const std::string& message) {
    std::cerr << "Error: " << message << std::endl;
    exit(1);
}

std::u32string tokenizer_t::utf8_to_utf32(const std::string& str) {
    std::u32string result;
    for (size_t i = 0; i < str.size();) {
        uint32_t code_point = 0;
        unsigned char first = str[i];
        if ((first & 0x80) == 0) {
            code_point = first;
            ++i;
        } else if ((first & 0xE0) == 0xC0) {
            code_point = first & 0x1F;
            code_point <<= 6;
            code_point |= (str[++i] & 0x3F);
            ++i;
        } else if ((first & 0xF0) == 0xE0) {
            code_point = first & 0x0F;
            code_point <<= 6;
            code_point |= (str[++i] & 0x3F);
            code_point <<= 6;
            code_point |= (str[++i] & 0x3F);
            ++i;
        }
        result.push_back(code_point);
    }
    return result;
}

std::vector<std::string> tokenizer_t::bpe(const std::u32string& input) {
    std::vector<std::u32string> tokens(input.size());
    for (char32_t c : input) {
        tokens.push_back(std::u32string(1, c));
    }

    while (true) {
        std::pair<std::u32string, std::u32string> best_pair;
        int best_rank = -1;

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            int rank = get_pair_rank(utf32_to_utf8(tokens[i]), utf32_to_utf8(tokens[i + 1]));
            if (rank != -1 && (best_rank == -1 || rank < best_rank)) {
                best_pair = {tokens[i], tokens[i + 1]};
                best_rank = rank;
            }
        }

        if (best_rank == -1) break;

        std::vector<std::u32string> merged_tokens;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i < tokens.size() - 1 && tokens[i] == best_pair.first && tokens[i + 1] == best_pair.second) {
                merged_tokens.push_back(best_pair.first + best_pair.second);
                ++i;
            } else {
                merged_tokens.push_back(tokens[i]);
            }
        }
        tokens = std::move(merged_tokens);
    }

    std::vector<std::string> result;
    for (const auto& token : tokens) {
        result.push_back(utf32_to_utf8(token));
    }
    return result;
}

std::string tokenizer_t::utf32_to_utf8(const std::u32string& str) {
    std::string result;
    for (char32_t code_point : str) {
        if (code_point <= 0x7F) {
            result.push_back(static_cast<char>(code_point));
        } else if (code_point <= 0x7FF) {
            result.push_back(0xC0 | (code_point >> 6));
            result.push_back(0x80 | (code_point & 0x3F));
        } else if (code_point <= 0xFFFF) {
            result.push_back(0xE0 | (code_point >> 12));
            result.push_back(0x80 | ((code_point >> 6) & 0x3F));
            result.push_back(0x80 | (code_point & 0x3F));
        } else {
            result.push_back(0xF0 | (code_point >> 18));
            result.push_back(0x80 | ((code_point >> 12) & 0x3F));
            result.push_back(0x80 | ((code_point >> 6) & 0x3F));
            result.push_back(0x80 | (code_point & 0x3F));
        }
    }
    return result;
}

int tokenizer_t::get_pair_rank(const std::string& first, const std::string& second) {
    for (size_t i = 0; i < merge_ranks.size(); ++i) {
        if (merge_ranks[i].first == first && merge_ranks[i].second == second) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::map<uint8_t, char32_t> tokenizer_t::bytes_to_unicode() {
    std::vector<uint8_t> bs;
    for (int i = 33; i <= 126; ++i) bs.push_back(i);
    for (int i = 161; i <= 172; ++i) bs.push_back(i);
    for (int i = 174; i <= 255; ++i) bs.push_back(i);

    std::vector<char32_t> cs(bs.begin(), bs.end());
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) != bs.end()) continue;
        bs.push_back(b);
        cs.push_back(256 + n);
        ++n;
    }

    std::map<uint8_t, char32_t> result;
    for (size_t i = 0; i < bs.size(); ++i) {
        result[bs[i]] = cs[i];
    }
    return result;
}

tokenizer_t::tokenizer_t(const std::string& vocab_file, const std::string& merges_file) {
    std::ifstream vocab_stream(vocab_file);
    if (!vocab_stream.is_open()) {
        throw std::runtime_error("Unable to open vocab file: " + vocab_file);
    }

    std::string line;
    std::string token;
    int n = 0;
    while (std::getline(vocab_stream, line)) {
        if (n % 2 == 0) {
            token = line;
        } else {
            int id = std::stoi(line);
            encoder[token] = id;
            decoder[id] = token;
        }
        n++;
    }
    vocab_stream.close();

    std::ifstream merges_stream(merges_file);
    if (!merges_stream.is_open()) {
        throw std::runtime_error("Unable to open merges file: " + merges_file);
    }

    std::getline(merges_stream, line); 
    while (std::getline(merges_stream, line)) {
        if (line.empty()) break;
        size_t split_pos = line.find(' ');
        if (split_pos == std::string::npos) throw std::runtime_error("Invalid line in merges file: " + line);
        std::string first = line.substr(0, split_pos);
        std::string second = line.substr(split_pos + 1);
        merge_ranks.emplace_back(first, second);
    }
    merges_stream.close();

    std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^a-zA-Z\d\s]+|\s+(?!\S)|\s+)";
    regex_splitter = std::regex(pattern);
    byte_encoder = bytes_to_unicode();
}

std::vector<int> tokenizer_t::tokenize(const std::string& text) {
    std::vector<int> tokens;
    std::sregex_iterator iter(text.begin(), text.end(), regex_splitter);
    std::sregex_iterator end_iter;

    while (iter != end_iter) {
        std::string utf8_token = iter->str();
        std::u32string utf32_token;
        for (uint8_t b : utf8_token) {
            utf32_token += byte_encoder.at(b);
        }
        auto bpe_encoded = bpe(utf32_token);
        // puts("**");
        for (const auto& bpe_token : bpe_encoded) {
            if(encoder.find(bpe_token)==encoder.end())
                continue;
            int token_id = encoder.at(bpe_token);
            if (token_id != 201) tokens.push_back(token_id);
        }
        ++iter;
    }

    return tokens;
}

void saveToFile(const std::vector<int>& ids, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }
    for (const auto& id : ids) {
        outFile << id << "\n";
    }
}

int main() {
    tokenizer_t tokenizer("vocab.txt", "merges.txt");
    std::ifstream input_file("input.txt", std::ios::in | std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Error: Could not open input.txt" << std::endl;
        return 0;
    }
    std::string input_text((std::istreambuf_iterator<char>(input_file)),
                         std::istreambuf_iterator<char>());
    input_file.close();
    std::vector<int> tokens = tokenizer.tokenize(input_text);
    saveToFile(tokens,"output.txt");
    return 0;
}
