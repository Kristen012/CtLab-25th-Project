#include <stdint.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_int.h>

// Define constants
#define NX 768
#define NF 3072
#define DEPTH_HEIGHT 128

#define DATA_SIZE_X (DEPTH_HEIGHT * NX)
#define DATA_SIZE_WEIGHT (NX * NF)
#define DATA_SIZE_BIAS 3072
#define DATA_SIZE_RES (DEPTH_HEIGHT * NX)
#define a_max (DEPTH_HEIGHT * NX)
#define b_max (NX * NF)
#define gelu_max (DEPTH_HEIGHT * NF)

// Define packed type for 2 floats in one 64-bit slot
typedef ap_uint<64> packed_t;

// Pack two floats into one 64-bit integer
packed_t pack_floats(float a, float b) {
#pragma HLS INLINE off
    packed_t packed;
    ap_uint<32> a_bits = *(ap_uint<32>*)&a;
    ap_uint<32> b_bits = *(ap_uint<32>*)&b;
    packed.range(31, 0) = a_bits;
    packed.range(63, 32) = b_bits;
    return packed;
}

// Unpack two floats from one 64-bit integer
void unpack_floats(packed_t packed, float& a, float& b) {
#pragma HLS INLINE off
    ap_uint<32> a_bits = packed.range(31, 0);
    ap_uint<32> b_bits = packed.range(63, 32);
    a = *(float*)&a_bits;
    b = *(float*)&b_bits;
}

// Load and reshape data to packed format
static void reshape(float* in, packed_t* out, int size) {
change_dim:
    for (int i = 0; i < size; i += 2) {
#pragma HLS PIPELINE II=1
        float val1 = in[i];
        float val2 = in[i + 1]; // Handle potential out-of-bounds access
        out[i / 2] = pack_floats(val1, val2);
    }
}

// First matrix multiplication and GELU activation
static void compute_matmul_1(packed_t* a, packed_t* b, packed_t* bias_local, packed_t* conv_result, int depth) {
plus_bias_1:
    for (int i = 0; i < DEPTH_HEIGHT; i++) {
        if (i < depth) {
            int iNF = i * NF;
            for (int j = 0; j < NF; j += 2) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                float bias_val1, bias_val2;
                unpack_floats(bias_local[j / 2], bias_val1, bias_val2);
                conv_result[(iNF + j) / 2] = pack_floats(bias_val1, bias_val2);
            }
        }
    }

cal_conv_result_1:
    for (int i = 0; i < DEPTH_HEIGHT; i++) {
        if (i < depth) {
            int iNF = i * NF;
            for (int k = 0; k < NX; k += 2) {
                float a1, a2;
                unpack_floats(a[(i * NX + k) / 2], a1, a2);
                for (int j = 0; j < NF; j += 2) {
#pragma HLS PIPELINE II=1 style=frp
#pragma HLS UNROLL factor=4
                    float c1, c2, b1, b2, b3, b4, d1, d2;
                    unpack_floats(conv_result[(iNF + j) / 2], c1, c2);
                    unpack_floats(b[(k * NF + j) / 2], b1, b2);
                    unpack_floats(b[((k + 1) * NF + j) / 2], b3, b4);
                    d1 = c1 + a1 * b1 + a2 * b3;
                    d2 = c2 + a1 * b2 + a2 * b4;
                    conv_result[(iNF + j) / 2] = pack_floats(d1, d2);
                }
            }
        }
    }
}

static void compute_GELU(packed_t* data, int size) {
cal_GELU:
    for (int i = 0; i < gelu_max; i += 2) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
        if (i < size) {
            float c1, c2, d1, d2;
            unpack_floats(data[i / 2], c1, c2);

            // Perform GELU on float values
            d1 = 0.5 * c1 * (1 + hls::tanh(0.7978845608 * (c1 + 0.044715 * c1 * c1 * c1)));
            d2 = 0.5 * c2 * (1 + hls::tanh(0.7978845608 * (c2 + 0.044715 * c2 * c2 * c2)));

            data[i / 2] = pack_floats(d1, d2);
        }
    }
}

// Second matrix multiplication
static void compute_matmul_2(packed_t* a, packed_t* b, packed_t* bias_local, packed_t* conv_result, int depth) {
plus_bias_2:
    for (int i = 0; i < DEPTH_HEIGHT; i++) {
        if (i < depth) {
            int iNX = i * NX;
            for (int j = 0; j < NX; j += 2) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=4
                float bias_val1, bias_val2;
                unpack_floats(bias_local[j / 2], bias_val1, bias_val2);
                conv_result[(iNX + j) / 2] = pack_floats(bias_val1, bias_val2);
            }
        }
    }

cal_conv_result_2:
    for (int i = 0; i < DEPTH_HEIGHT; i++) {
        if (i < depth) {
            int iNX = i * NX;
            for (int k = 0; k < NF; k += 2) {
                float a1, a2;
                unpack_floats(a[(i * NF + k) / 2], a1, a2);
                for (int j = 0; j < NX; j += 2) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=4
                    float c1, c2, b1, b2, b3, b4, d1, d2;
                    unpack_floats(conv_result[(iNX + j) / 2], c1, c2);
                    unpack_floats(b[(k * NX + j) / 2], b1, b2);
                    unpack_floats(b[((k + 1) * NX + j) / 2], b3, b4);
                    d1 = c1 + a1 * b1 + a2 * b3;
                    d2 = c2 + a1 * b2 + a2 * b4;
                    conv_result[(iNX + j) / 2] = pack_floats(d1, d2);
                }
            }
        }
    }
}

// Store result back to output
static void store_result(float* out, packed_t* in, int size) {
mem_wr:
    for (int i = 0; i < size; i += 2) {
#pragma HLS PIPELINE II=1
        float val1, val2;
        unpack_floats(in[i / 2], val1, val2);
        out[i] = val1;
        out[i + 1] = val2; // Handle potential out-of-bounds access
    }
}

extern "C" {
void krnl_MLP(float* x, float* weight, float* bias, float* weight_2, float* bias_2, int depth, float* out) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 depth = DATA_SIZE_X max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem1 depth = DATA_SIZE_WEIGHT max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = bias offset = slave bundle = gmem2 depth = DATA_SIZE_BIAS max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = weight_2 offset = slave bundle = gmem4 depth = DATA_SIZE_WEIGHT max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = bias_2 offset = slave bundle = gmem5 depth = NX max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem7 depth = gelu_max max_write_burst_length = 256
#pragma HLS INTERFACE s_axilite port = x
#pragma HLS INTERFACE s_axilite port = weight
#pragma HLS INTERFACE s_axilite port = bias
#pragma HLS INTERFACE s_axilite port = weight_2
#pragma HLS INTERFACE s_axilite port = bias_2
#pragma HLS INTERFACE s_axilite port = depth
#pragma HLS INTERFACE s_axilite port = out
#pragma HLS INTERFACE s_axilite port = return

    // Define packed buffers
    packed_t a[a_max / 2];
    packed_t b[b_max / 2];
    packed_t conv_result_1[gelu_max / 2];
    packed_t conv_result_2[DATA_SIZE_RES / 2];
    packed_t bias_local[DATA_SIZE_BIAS / 2];

#pragma HLS ARRAY_PARTITION variable=conv_result_1 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=conv_result_2 cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=a cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=b cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=bias_local cyclic factor=4 dim=1

#pragma HLS bind_storage variable=conv_result_1 type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=conv_result_2 type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=a type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=b type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=bias_local type=RAM_T2P impl=uram

    reshape(x, a, depth * NX);
    reshape(weight, b, DATA_SIZE_WEIGHT);
    reshape(bias, bias_local, DATA_SIZE_BIAS);

    compute_matmul_1(a, b, bias_local, conv_result_1, depth);

    compute_GELU(conv_result_1, depth * NF);

    reshape(weight_2, b, DATA_SIZE_WEIGHT);
    reshape(bias_2, bias_local, NX);

    compute_matmul_2(conv_result_1, b, bias_local, conv_result_2, depth);

    store_result(out, conv_result_2, depth * NX);
}
}
