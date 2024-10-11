#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <locale>
#include <codecvt>
#include <algorithm>
using namespace std;
void print_b2u_mapping(const std::unordered_map < uint8_t, wchar_t > & b2u) {
    for (const auto & pair: b2u) {
        std::wcout << L"Debug: Byte value: " << static_cast < int > (pair.first) <<
            L", Unicode value: " << static_cast < int > (pair.second) << std::endl;
    }
}
void print_u2b_mapping(const std::unordered_map < wchar_t, uint8_t > & u2b) {
    for (const auto & pair: u2b) {
        std::wcout << L"Debug: Byte value: " << static_cast < int > (pair.first) <<
            L", Unicode value: " << static_cast < int > (pair.second) << std::endl;
    }
}

std::vector < int > readIndicesFromCSV(const std::string & filename) {
    std::vector < int > indices;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return indices;
    }

    std::string line;

    // Assuming there is only one row
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;

        // Read each comma-separated value
        while (std::getline(ss, value, ',')) {
            try {
                // Convert the value to an integer and add it to the vector
                indices.push_back(std::stoi(value));
            } catch (const std::invalid_argument & e) {
                std::cerr << "Invalid number: " << value << std::endl;
            }
        }
    }

    file.close();
    return indices;
}
void insert_range(std::unordered_map<uint8_t, wchar_t>* b2u, int start, int end) {
    for (int c = start; c <= end; c++) {
        b2u->insert({ uint8_t(c), wchar_t(c) });
    }
}

void bytes_to_unicode(std::unordered_map<uint8_t, wchar_t>* b2u, 
                      std::unordered_map<wchar_t, uint8_t>* u2b) {
    b2u->clear();
    insert_range(b2u, L'!', L'~');
    insert_range(b2u, L'¡', L'¬');
    insert_range(b2u, L'®', L'ÿ');

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b2u->find(uint8_t(b)) == b2u->end()) {
            b2u->insert({ uint8_t(b), wchar_t(256 + n) });
            n++;
        }
    }

    if (u2b != NULL) {
        u2b->clear();
        for (auto e : (*b2u)) {
            u2b->insert({ e.second, e.first });
        }
    }
}
void print_i2t(const std::unordered_map < int, std::string > & i2t) {
    for (const auto & pair: i2t) {
        std::cout << pair.first << " : " << pair.second << std::endl;
        std::cout << " : " << std::endl;
    }
}
void load_vocab(std::istream & ins, std::unordered_map < std::string, int > * t2i,
    std::unordered_map < int, std::string > * i2t) {
    t2i -> clear();
    i2t -> clear();

    std::string line;
    std::string token;
    int n = 0;
    while (std::getline(ins, line)) {
        if (n % 2 == 0) {
            token = line;
        } else {
            t2i -> insert({
                token,
                std::stoi(line)
            });
            i2t -> insert({
                std::stoi(line),
                token
            });
        }
        n++;
    }
}
std::wstring utf8_to_wstring(const std::string & str) {
    std::wstring_convert < std::codecvt_utf8 < wchar_t >> myconv;
    return myconv.from_bytes(str);
}
std::string decode(const std::vector < int > & ids,
    const std::unordered_map < wchar_t, uint8_t > & u2b,
        const std::unordered_map < int, std::string > & i2t) {
    // std::string concat;
    // std::wstring concat;

    // int index=0;
    //     std::setlocale(LC_ALL, "");

    // for (int id : ids) {
    //     std::wstring current_value = utf8_to_wstring(i2t.at(id)); // Convert each string to wstring
    //     concat =  concat+current_value; // Append value to concat
    //     std::wcout << L"Index " << index << L": " << current_value << std::endl; // Print the index and value
    //     std::wcout << concat << std::endl; // Print the current concat
    //     std::wcout << L"Length of concat: " << concat.length() << std::endl; // Print the length of concat
    //     index++;
    //     std::wcout << std::endl; // Add a new line for better readability
    // }
    //       std::unordered_map<int, uint8_t> int_map;

    // // Populate the new map
    // for (const auto& pair : u2b) {
    //     cout<<static_cast<int>(pair.first)<<'\n';
    //     int_map[static_cast<int>(pair.first)] = pair.second;
    // }

    std::string concat;
    std::vector < wchar_t > wchar_vector;

    // Iterate through each id, get the corresponding value, and convert to wchar_t
    std::wstring concatenated_wstr; // Create a variable to hold the concatenated wstring
    std::string r;
    for (int id: ids) {
        std::string value = i2t.at(id);
        // std::cout << value << std::endl; // Print each value
        // std::cout << value.size() << std::endl; // Print each value
        std::wstring w = utf8_to_wstring(value);
        // std::wcout << w << static_cast<int>(c)std::endl;

        // Append the current wstring to the concatenated variable
        // concatenated_wstr.append(w);
        for (wchar_t c: w) {
            // std::wcout<<c<<" "<<static_cast<int>(c)<<" ";
            // std::wcout << std::flush;
            if (u2b.count(c))
                r.push_back(char(u2b.at(c)));
            // if(static_cast<int>(c)!=13)
            //   r.push_back(char(int_map.at(static_cast<int>(c))));
        }
        // std::wcout << std::endl;
    }

    // Optionally print the concatenated wstring
    // std::wcout << L"Concatenated wstring: " << concatenated_wstr << std::endl;

    // Output the vector to validate
    // std::wcout << L"Vector of wchar_t: ";
    // for (wchar_t wc : wchar_vector) {
    //     std::wcout << wc;
    // }
    // std::wcout << std::endl;

    // cout<<"asfasfsaffasfageageag"<<'\n';
    // cout<<concat<<'\n';
    // std::setlocale(LC_ALL, "");

    // Assuming utf8_to_wstring is already defined
    // std::wstring w = utf8_to_wstring(concat);

    // Use std::wcout to print the wide string
    // std::wcout << w << std::endl;
    // std::string r;
    // for (wchar_t c : w) {
    //   r.push_back(char(u2b.at(c)));
    // }
    // return "";
    return r;
}

void Decode(std::vector<int> indices) {
    // Load vocab.json and create the decoder
    std::unordered_map < std::string, int > t2i;
    std::unordered_map < int, std::string > i2t;
    std::fstream vocab_txt("/home/ywtang23/Data/weight/vocab.txt", std::ios::in);
    load_vocab(vocab_txt, & t2i, & i2t);
    // cout << i2t.size() << "  "<< t2i.size() << endl;
    std::unordered_map < uint8_t, wchar_t > b2u;
    std::unordered_map < wchar_t, uint8_t > u2b;
    bytes_to_unicode( & b2u, & u2b);
    // std::wcout.imbue(std::locale(""));

    // Output the contents of u2b
    // std::vector<std::pair<wchar_t, uint8_t>> vec(u2b.begin(), u2b.end());

    // Sort the vector based on the numeric value of pair.first (wchar_t)
    // std::sort(vec.begin(), vec.end(), [](const std::pair<wchar_t, uint8_t>& a, const std::pair<wchar_t, uint8_t>& b) {
    //     return static_cast<int>(a.first) < static_cast<int>(b.first);
    // });

    // // Output the sorted vector
    // for (const auto& pair : vec) {
    //     std::wcout << L"Character (numeric): " << static_cast<int>(pair.first)
    //                << L", Byte: " << static_cast<int>(pair.second) << std::endl;
    // }

    // cout << (int) b2u.size()<< endl;

    // // Example indices to decode
    // std::vector < int > indices = readIndicesFromCSV("output_ids.csv");
    // for(auto x: indices)
    //     printf("%d\n",x);
    // // Output the result
    std::cout << "===== Decoded Result =====" << endl;
    std::string decoded_result = decode(indices, u2b, i2t);
    std::cout << decoded_result << std::endl;
    std::ofstream outfile("/home/ywtang23/Data/decoded_output.txt");
    if (outfile.is_open()) {
        outfile << decoded_result;
        outfile.close();
        std::cout << "Decoded result has been written to decoded_output.txt" << std::endl;
    } else {
        std::cerr << "Failed to open the file for writing" << std::endl;
    }

    return;
}
