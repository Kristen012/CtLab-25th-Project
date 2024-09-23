#include <stdint.h>
#include <hls_math.h>
#include <ap_int.h>
#include <ap_fixed.h>
#include <hls_half.h>
#include "krnl_helper.h"

// Load input data into local arrays using packed fp24
static void load_input(float* in, packed_fp24_t* local_in, int size) {
mem_rd:
    for (int i = 0; i < size; i += 3) {
#pragma HLS PIPELINE II=1
        fp24_t val1 = static_cast<fp24_t>(in[i]);
        fp24_t val2 = static_cast<fp24_t>(in[i+1]);
        fp24_t val3 = static_cast<fp24_t>(in[i+2]);
        local_in[i / 3] = pack_fp24(val1, val2, val3);
    }
}

// Perform matrix-vector multiplication using packed fp24
static void compute_mul(packed_fp24_t* X, packed_fp24_t* W, packed_fp24_t* result, int depth) {
init_result:
    for (int i = 0; i < MAX_DEPTH; i++) {
        if (i < depth) {
            int iN0 = i * N0;
            for (int j = 0; j < N0; j += 3) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=8
                result[(iN0 + j) / 3] = pack_fp24(0.0f, 0.0f, 0.0f);
            }
        }
    }

execute_mul:
    for (int i = 0; i < MAX_DEPTH; i++) {
        if (i < depth) {
            int iN0 = i * N0;
            for (int k = 0; k < K0; k += 3) {
                fp24_t x1, x2, x3;
                unpack_fp24(X[(i * K0 + k) / 3], x1, x2, x3);
                for (int j = 0; j < N0; j += 3) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=4
                    fp24_t c1, c2, c3, w1, w2, w3, w4, w5, w6, w7, w8, w9;
                    fp24_t d1, d2, d3;
                    unpack_fp24(result[(iN0 + j) / 3], c1, c2, c3);
                    unpack_fp24(W[(k * N0 + j) / 3], w1, w2, w3); // Unpack weights corresponding to j
                    unpack_fp24(W[((k + 1) * N0 + j) / 3], w4, w5, w6); // Unpack weights corresponding to j + 1
                    unpack_fp24(W[((k + 2) * N0 + j) / 3], w7, w8, w9); // Unpack weights corresponding to j + 1

                    d1 = c1 + x1 * w1 + x2 * w4 + x3 * w7;
                    d2 = c2 + x1 * w2 + x2 * w5 + x3 * w8;
                    d3 = c3 + x1 * w3 + x2 * w6 + x3 * w9;

                    result[(iN0 + j) / 3] = pack_fp24(d1, d2, d3);
                }
            }
        }
    }

}

// Store the result back to fp24 precision after unpacking
static void store_result(float* out, packed_fp24_t* local_out, int size) {
mem_wr:
    for (int i = 0; i < size; i += 3) {
#pragma HLS PIPELINE II=1
        fp24_t val1, val2, val3;

        unpack_fp24(local_out[i / 3], val1, val2, val3);

        out[i] = val1.to_half();
        out[i + 1] = val2.to_half();
        out[i + 2] = val3.to_half();
    }
}

extern "C" {
void krnl_linear_2(float* x, float* w, float* out, int depth) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem1 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem2 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=depth bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Define local buffers using packed fp24
    packed_fp24_t local_x[MAX_DEPTH * K0 / 3];
    packed_fp24_t local_w[(K0 * N0) / 3];
    packed_fp24_t local_result[MAX_DEPTH * N0 / 3];

#pragma HLS ARRAY_PARTITION variable=local_x cyclic factor=3 dim=1
#pragma HLS ARRAY_PARTITION variable=local_w cyclic factor=6 dim=1
#pragma HLS ARRAY_PARTITION variable=local_result cyclic factor=6 dim=1

#pragma HLS bind_storage variable=local_x type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=local_w type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=local_result type=RAM_T2P impl=uram

#pragma HLS DATAFLOW
    load_input(x, local_x, depth * K0);
    load_input(w, local_w, K0 * N0);

    compute_mul(local_x, local_w, local_result, depth);

    store_result(out, local_result, depth * N0);
}
}

