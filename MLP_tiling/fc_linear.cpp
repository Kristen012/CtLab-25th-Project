#include <stdint.h>
#include <hls_math.h>
#include <ap_int.h>
#include <hls_half.h>
#include <ap_fixed.h>

#define fc_K0 768
#define fc_N0 258
#define MAX_SIZE (fc_K0 * fc_N0)
#define MAX_DEPTH 32

typedef ap_uint<64> packed_t;
typedef half half_t;

// Pack two float values into one 64-bit integer
packed_t pack_floats(float a, float b) {
#pragma HLS INLINE off
    packed_t packed;
    ap_uint<32> a_bits = *(ap_uint<32>*)&a;
    ap_uint<32> b_bits = *(ap_uint<32>*)&b;
    packed.range(31, 0) = a_bits;
    packed.range(63, 32) = b_bits;
    return packed;
}

// Unpack two float values from one 64-bit integer
void unpack_floats(packed_t packed, float& a, float& b) {
#pragma HLS INLINE off
    ap_uint<32> a_bits = packed.range(31, 0);
    ap_uint<32> b_bits = packed.range(63, 32);
    a = *(float*)&a_bits;
    b = *(float*)&b_bits;
}

// Load input data into local arrays using packed floats
static void load_input(float* in, packed_t* local_in, int size) {
mem_rd:
    for (int i = 0; i < size; i += 2) {
#pragma HLS PIPELINE II=1
        float val1 = in[i];
        float val2 = in[i + 1];
        local_in[i / 2] = pack_floats(val1, val2);
    }
}

// Perform matrix-vector multiplication using packed floats
static void compute_mul(packed_t* X, packed_t* W, packed_t* B, packed_t* result, int depth) {
init_result:
    for (int i = 0; i < MAX_DEPTH; i++) {
        if (i < depth) {
            int iN0 = i * fc_N0;
            for (int j = 0; j < fc_N0; j += 2) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=16
                float bias_val1, bias_val2;
                unpack_floats(B[j / 2], bias_val1, bias_val2);
                result[(iN0 + j) / 2] = pack_floats(bias_val1, bias_val2);
            }
        }
    }

execute_mul:
    for (int i = 0; i < MAX_DEPTH; i++) {
        if (i < depth) {
            int iN0 = i * fc_N0;
            for (int k = 0; k < fc_K0; k += 2) {
                float x1, x2;
                unpack_floats(X[(i * fc_K0 + k) / 2], x1, x2);
                for (int j = 0; j < fc_N0; j += 2) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL factor=2
                    float c1, c2, w1, w2, w3, w4, d1, d2;
                    unpack_floats(result[(iN0 + j) / 2], c1, c2);
                    unpack_floats(W[(k * fc_N0 + j) / 2], w1, w2); // Unpack weights corresponding to j
                    unpack_floats(W[((k + 1) * fc_N0 + j) / 2], w3, w4); // Unpack weights corresponding to j + 1

                    d1 = c1 + x1 * w1 + x2 * w3;
                    d2 = c2 + x1 * w2 + x2 * w4;

                    result[(iN0 + j) / 2] = pack_floats(d1, d2);
                }
            }
        }
    }

compute_gelu:
	for (int i = 0; i < MAX_DEPTH * fc_N0; i += 2) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
		if (i < depth * fc_N0) {
				float c1, c2, d1, d2;
				unpack_floats(result[i / 2], c1, c2);

				d1 = 0.5 * c1 * (1 + hls::tanh(0.7978845608 * (c1 + 0.044715 * c1 * c1 * c1)));
				d2 = 0.5 * c2 * (1 + hls::tanh(0.7978845608 * (c2 + 0.044715 * c2 * c2 * c2)));

				result[i / 2] = pack_floats(d1, d2);
		}
	}
}

// Store the result back to half precision after unpacking
static void store_result(float* out, packed_t* local_out, int size) {
mem_wr:
    for (int i = 0; i < size; i += 2) {
#pragma HLS PIPELINE II=1
        float val1, val2;

        unpack_floats(local_out[i / 2], val1, val2);

        out[i] = val1;
        out[i + 1] = val2;
    }
}

extern "C" {

void krnl_MLP_fc_linear(float* x, float* w, float* b, float* out, int depth) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=w offset=slave bundle=gmem1 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem2 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem3 depth=4096 max_read_burst_length=256
#pragma HLS INTERFACE s_axilite port=x bundle=control
#pragma HLS INTERFACE s_axilite port=w bundle=control
#pragma HLS INTERFACE s_axilite port=b bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=depth bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // Define local buffers using packed floats
    packed_t local_x[MAX_DEPTH * fc_K0 / 2];
    packed_t local_w[(fc_K0 * fc_N0) / 2];
    packed_t local_b[fc_N0 / 2];
    packed_t local_result[MAX_DEPTH * fc_N0 / 2];

#pragma HLS ARRAY_PARTITION variable=local_x cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=local_w cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_b cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=local_result cyclic factor=8 dim=1

#pragma HLS bind_storage variable=local_x type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=local_w type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=local_b type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=local_result type=RAM_T2P impl=uram

//#pragma HLS DATAFLOW
    load_input(x, local_x, depth * fc_K0);
    load_input(w, local_w, fc_K0 * fc_N0);
    load_input(b, local_b, fc_N0);

    compute_mul(local_x, local_w, local_b, local_result, depth);

    store_result(out, local_result, depth * fc_N0);
}

}
