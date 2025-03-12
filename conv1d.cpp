#include <iostream>
#include <vector>

#define DEPTH 1
#define HEIGHT 512
#define NX 768
#define NF 768
#define DEPTH_HEIGHT 512

#define DATA_SIZE_X 393216
#define DATA_SIZE_WEIGHT 589824
#define DATA_SIZE_BIAS 768
#define DATA_SIZE_RES 393216
#define a_max DEPTH_HEIGHT * NX
#define b_max NX * NF
#define o_max DEPTH_HEIGHT * NF

void reshape(const std::vector<float>& in, std::vector<float>& outStream, int size) {
    outStream.assign(in.begin(), in.begin() + size);
}

void compute_matmul(const std::vector<float>& xStream,
                    const std::vector<float>& weightStream,
                    const std::vector<float>& biasStream,
                    std::vector<float>& outStream) {
    std::vector<float> a(a_max);
    std::vector<float> b(b_max);
    std::vector<float> conv_result(o_max, 0.0f);
    std::vector<float> bias(NF);

    for (int i = 0; i < a_max; i++) {
        a[i] = xStream[i];
    }
    for (int i = 0; i < b_max; i++) {
        b[i] = weightStream[i];
    }
    for (int i = 0; i < NF; i++) {
        bias[i] = biasStream[i];
    }

    for (int i = 0; i < DEPTH_HEIGHT; i++) {
        for (int j = 0; j < NF; j++) {
            conv_result[i * NF + j] = bias[j];
        }
    }

    for (int i = 0; i < DEPTH_HEIGHT; i++) {
        int iNF = i * NF;
        for (int k = 0; k < NX; k++) {
            int kNF = k * NF;
            float aik = a[i * NX + k];
            for (int j = 0; j < NF; j++) {
                conv_result[iNF + j] += aik * b[kNF + j];
            }
        }
    }

    outStream.assign(conv_result.begin(), conv_result.begin() + o_max);
}

void store_result(std::vector<float>& out, const std::vector<float>& inStream, int size) {
    out.assign(inStream.begin(), inStream.begin() + size);
}

void krnl_conv1D(const std::vector<float>& x, 
                 const std::vector<float>& weight, 
                 const std::vector<float>& bias, 
                 std::vector<float>& out) {
    std::vector<float> xStream, weightStream, biasStream, outStream;

    reshape(x, xStream, DATA_SIZE_X);
    reshape(weight, weightStream, DATA_SIZE_WEIGHT);
    reshape(bias, biasStream, DATA_SIZE_BIAS);

    compute_matmul(xStream, weightStream, biasStream, outStream);

    store_result(out, outStream, DATA_SIZE_RES);
}

int main() {
    std::vector<float> x(DATA_SIZE_X);
    std::vector<float> weight(DATA_SIZE_WEIGHT);
    std::vector<float> bias(DATA_SIZE_BIAS);
    std::vector<float> out(DATA_SIZE_RES);

    for (int i = 0; i < DATA_SIZE_X; i++) {
        std::cin >> x[i];
    }

    for (int i = 0; i < DATA_SIZE_WEIGHT; i++) {
        std::cin >> weight[i];
    }

    for (int i = 0; i < DATA_SIZE_BIAS; i++) {
        std::cin >> bias[i];
    }

    krnl_conv1D(x, weight, bias, out);

    return 0;
}