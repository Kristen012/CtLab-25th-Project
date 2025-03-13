#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> readData(const string &filename) {
    vector<double> data;
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    double value;
    while (file >> value) {
        data.push_back(value);
    }
    file.close();
    return data;
}

void writeData(const string &filename, const vector<vector<double>> &data) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        exit(1);
    }
    for (const auto &row : data) {
        for (const auto &val : row) {
            file << val << "\n";
        }
    }
    file.close();
}

vector<vector<double>> matmul(const vector<vector<double>> &A, const vector<vector<double>> &B) {
    int m = A.size();
    int n = B[0].size();
    int k = A[0].size();

    cout << "matmul called with A shape: (" << m << ", " << k << "), B shape: (" << k << ", " << n << ")" << endl;

    vector<vector<double>> result(m, vector<double>(n, 0.0));

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int l = 0; l < k; ++l) {
                result[i][j] += A[i][l] * B[l][j];
            }
        }
    }

    return result;
}

void addBias(vector<vector<double>> &matrix, const vector<double> &bias) {
    int rows = matrix.size();
    int cols = matrix[0].size();

    cout << "addBias called with matrix shape: (" << rows << ", " << cols << "), bias shape: (" << bias.size() << ")" << endl;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] += bias[j];
        }
    }
}

double gelu(double x) {
    return 0.5 * x * (1 + tanh(sqrt(2.0 / 3.1415926535) * (x + 0.044715 * pow(x, 3))));
}

// Apply GELU activation to a matrix
void applyGelu(vector<vector<double>> &matrix) {
    for (auto &row : matrix) {
        for (auto &val : row) {
            val = gelu(val);
        }
    }
}

int main() {
    vector<double> weight_data = readData("h0_mlp_weight.txt");
    vector<double> bias_data = readData("h0_mlp_bias.txt");
    vector<double> input_data = readData("h0_mlp_input.txt");
    vector<double> weight_2_data = readData("h0_mlp_weight_2.txt");
    vector<double> bias_2_data = readData("h0_mlp_bias_2.txt");

    int S = 127, D_in = 768, D_hidden = 3072;
    vector<vector<double>> weight(D_in, vector<double>(D_hidden));
    vector<vector<double>> weight_2(D_hidden, vector<double>(D_in));
    vector<vector<double>> x(S, vector<double>(D_in));
    vector<double> bias(D_hidden);
    vector<double> bias_2(D_in);

    int idx = 0;
    for (int i = 0; i < D_in; ++i)
        for (int j = 0; j < D_hidden; ++j)
            weight[i][j] = weight_data[idx++];

    idx = 0;
    for (int i = 0; i < D_hidden; ++i)
        for (int j = 0; j < D_in; ++j)
            weight_2[i][j] = weight_2_data[idx++];

    idx = 0;
    for (int i = 0; i < S; ++i)
        for (int j = 0; j < D_in; ++j)
            x[i][j] = input_data[idx++];

    for (int i = 0; i < D_hidden; ++i)
        bias[i] = bias_data[i];

    for (int i = 0; i < D_in; ++i)
        bias_2[i] = bias_2_data[i];

    vector<vector<double>> result = matmul(x, weight);
    addBias(result, bias);

    applyGelu(result);

    result = matmul(result, weight_2);
    addBias(result, bias_2);

    writeData("mlp_result.txt", result);


    // cout << "Time to calculate first layer: "
    //      << chrono::duration<double>(end_layer1 - start_layer1).count() << " seconds" << endl;
    // cout << "Time to apply GELU activation: "
    //      << chrono::duration<double>(end_gelu - start_gelu).count() << " seconds" << endl;
    // cout << "Time to calculate second layer: "
    //      << chrono::duration<double>(end_layer2 - start_layer2).count() << " seconds" << endl;
    // cout << "Time to save result to file: "
    //      << chrono::duration<double>(end_saving - start_saving).count() << " seconds" << endl;
    // cout << "Total time taken: "
    //      << chrono::duration<double>(end_time - start_time).count() << " seconds" << endl;

    return 0;
}
