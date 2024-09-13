#include <stdint.h>
#include <ap_int.h>
#include <hls_math.h>
#include <stdio.h>

#define MAX_DEPTH 128
#define MAX_WIDTH 768
#define MAX_DATA_SIZE MAX_DEPTH * MAX_WIDTH
#define CHUNK_SIZE 1
#define roll 4

static ap_uint<72> temp_input[MAX_WIDTH / 2];
static ap_uint<72> in1_local[MAX_WIDTH / 2];
static ap_uint<72> out_local[MAX_WIDTH / 2];

static float mean, var, stddev, tmp_sum, tmp_sumsq, norm, sum, sumsq, norm_a, norm_b;
static float a, b, c, d;
static ap_uint<32> a_bits, b_bits;
static ap_uint<72> packed;

static void init_temp_input() {
    #pragma HLS bind_storage variable=temp_input type=RAM_T2P impl=uram
    #pragma HLS ARRAY_PARTITION variable=temp_input cyclic factor=4 dim=1
    #pragma HLS bind_storage variable=in1_local type=RAM_T2P impl=uram
    #pragma HLS ARRAY_PARTITION variable=in1_local cyclic factor=4 dim=1
    #pragma HLS bind_storage variable=out_local type=RAM_T2P impl=uram
    #pragma HLS ARRAY_PARTITION variable=out_local cyclic factor=4 dim=1
}

ap_uint<72> pack_floats(float a, float b) {
    #pragma HLS INLINE off
    a_bits = *(ap_uint<32> *)&a;
    b_bits = *(ap_uint<32> *)&b;
    packed.range(31, 0) = a_bits;
    packed.range(63, 32) = b_bits;
    return packed;
}

void unpack_floats(ap_uint<72> packed, float &a, float &b) {
    #pragma HLS INLINE off
    a_bits = packed.range(31, 0);
    b_bits = packed.range(63, 32);
    a = *(float *)&a_bits;
    b = *(float *)&b_bits;
}

static void load_input_chunk(float *in, ap_uint<72> *local_mem, int start, int size) {
mem_rd:
    for (int i = 0; i < size; i += 2) {
        #pragma HLS PIPELINE II=1
        local_mem[i / 2] = pack_floats(in[start + i], in[start + i + 1]);
    }
}

static void load_GB(float* in_1, float* local_mem_1, float* in_2, float* local_mem_2, int size) {
mem_rd2:
    for (int i = 0; i < size; i++) {
        #pragma HLS PIPELINE II=1
        local_mem_1[i] = in_1[i];
        local_mem_2[i] = in_2[i];
    }
}

static void store_result_chunk(float *out, ap_uint<72> *local_mem, int start, int size) {
mem_wr:
    for (int i = 0; i < size; i += 2) {
        #pragma HLS PIPELINE II=1
        unpack_floats(local_mem[i / 2], a, b);
        out[start + i] = a;
        out[start + i + 1] = b;
    }
}

static void compute_norm_chunk(ap_uint<72> *in1_local, ap_uint<72> *out_local, int width, float *G, float *B) {
    #pragma HLS ARRAY_PARTITION variable=G cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=B cyclic factor=4 dim=1

    sum = 0;
    sumsq = 0;

    load_temp_input:
    for (int j = 0; j < width; j += 2) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL skip_exit_check factor=4
        temp_input[j / 2] = in1_local[j / 2];
    }

    compute_norm_mean:
    for (int j = 0; j < width; j += roll) {
        #pragma HLS PIPELINE II=1
        unpack_floats(temp_input[j / 2], a, b);
        unpack_floats(temp_input[(j + 2) / 2], c, d);
        tmp_sum = a + b + c + d;
        sum += tmp_sum;
        tmp_sumsq = a * a + b * b + c * c + d * d;
        sumsq += tmp_sumsq;
    }

    mean = sum / width;
    var = sumsq / width - mean * mean;
    stddev = hls::sqrt(var + 0.00001);

    compute_norm_normalize:
    for (int j = 0; j < width; j += 2) {
        #pragma HLS PIPELINE II=1
        #pragma HLS UNROLL skip_exit_check factor=4
        unpack_floats(temp_input[j / 2], a, b);
        norm_a = (a - mean) / stddev * G[j] + B[j];
        norm_b = (b - mean) / stddev * G[j + 1] + B[j + 1];
        out_local[j / 2] = pack_floats(norm_a, norm_b);
    }
}

extern "C" {

void layer_norm(float *in1, float *out, float *g, float *b, int depth) {
    #pragma HLS INTERFACE m_axi port=in1 offset=slave bundle=gmem0 depth=4096 max_read_burst_length=128 num_read_outstanding=16 num_write_outstanding=16 max_write_burst_length=128
    #pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=4096 max_read_burst_length=128 num_read_outstanding=16 num_write_outstanding=16 max_write_burst_length=128
    #pragma HLS INTERFACE s_axilite port=in1 bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=depth bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS INTERFACE m_axi port=g offset=slave bundle=gmem2 depth=768 max_read_burst_length=128 num_read_outstanding=16 num_write_outstanding=16 max_write_burst_length=128
    #pragma HLS INTERFACE m_axi port=b offset=slave bundle=gmem3 depth=768 max_read_burst_length=128 num_read_outstanding=16 num_write_outstanding=16 max_write_burst_length=128
    #pragma HLS INTERFACE s_axilite port=g bundle=control
    #pragma HLS INTERFACE s_axilite port=b bundle=control

    float G_local[MAX_WIDTH];
    float B_local[MAX_WIDTH];

    #pragma HLS ARRAY_PARTITION variable=G_local cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=B_local cyclic factor=4 dim=1

    #pragma HLS bind_storage variable=G_local type=RAM_T2P impl=uram
    #pragma HLS bind_storage variable=B_local type=RAM_T2P impl=uram

    int size = depth * MAX_WIDTH;

    init_temp_input();

    #pragma HLS dataflow
    load_GB(g, G_local, b, B_local, MAX_WIDTH);

    process_depth_chunks:
    for (int i = 0; i < depth; i += CHUNK_SIZE) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=MAX_DEPTH / CHUNK_SIZE

        #pragma HLS ARRAY_PARTITION variable=in1_local cyclic factor=4 dim=1
        #pragma HLS ARRAY_PARTITION variable=out_local cyclic factor=4 dim=1

        #pragma HLS bind_storage variable=in1_local type=RAM_T2P impl=uram
        #pragma HLS bind_storage variable=out_local type=RAM_T2P impl=uram

        load_input_chunk(in1, in1_local, i * MAX_WIDTH, CHUNK_SIZE * MAX_WIDTH);

        for (int j = 0; j < CHUNK_SIZE; j++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=CHUNK_SIZE
            compute_norm_chunk(&in1_local[j * MAX_WIDTH / 2], &out_local[j * MAX_WIDTH / 2], MAX_WIDTH, G_local, B_local);
        }

        store_result_chunk(out, out_local, i * MAX_WIDTH, CHUNK_SIZE * MAX_WIDTH);
    }
}
}
