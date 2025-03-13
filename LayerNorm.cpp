#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace std::chrono;

void LayerNorm(float* data, float* G, float* B, float* out, int height, int width) {
    for(int i = 0; i < height; i++) {
        float sum = 0;
        float sumsq = 0;

        for(int j = 0; j < width; j++) {
            sum += data[i * width + j];
            sumsq += data[i * width + j] * data[i * width + j];
        }

        float mean = sum / width;
        float var = sumsq / width - mean * mean;
        float stddev = sqrt(var + 1e-5);

        for(int j = 0; j < width; j++) {
            out[i * width + j] = (data[i * width + j] - mean) / stddev * G[j] + B[j];
        }
    }
}

int main() {
    string filename = "flattened_data.txt";

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "can't open: " << filename << endl;
        return 1;
    }

    float* data = new float[512*768];
    string line;

    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        int index = 0;
        while (getline(ss, value, ',')) {
            data[index++] = stof(value);
        }
    }

    file.close();

    int HEIGHT = 128;
    int WIDTH = 768;
    float out[128*768];
    float G[768];
    float B[768];
    for(int i = 0; i < WIDTH; i++) {
        G[i] = 1;
        B[i] = 0;
    }

    LayerNorm(data, G, B, out, HEIGHT, WIDTH);

    return 0;
}
