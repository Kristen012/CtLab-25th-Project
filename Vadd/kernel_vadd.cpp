// Includes
#include <stdint.h>
#include <hls_stream.h>

#define DATA_SIZE 128*768

// TRIPCOUNT identifier
const int c_size = DATA_SIZE;

static void load_input(float* in, hls::stream<float>& inStream, int size) {
mem_rd:
    for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = c_size
        inStream << in[i];
    }
}

static void compute_add(hls::stream<float>& in1_stream,
                        hls::stream<float>& in2_stream,
                        hls::stream<float>& out_stream,
                        int size) {
execute_add:
    for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = c_size
        out_stream << (in1_stream.read() + in2_stream.read());
    }
}

static void store_result(float* out, hls::stream<float>& out_stream, int size) {
mem_wr:
    for (int i = 0; i < size; i++) {
#pragma HLS LOOP_TRIPCOUNT min = 0 max = c_size
        out[i] = out_stream.read();
    }
}

extern "C" {

void krnl_vadd(float* in1, float* in2, float* out, int size) {
#pragma HLS INTERFACE m_axi port = in1 offset = slave bundle = gmem10 depth = 4096 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = in2 offset = slave bundle = gmem19 depth = 4096 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem11 depth = 4096 max_read_burst_length = 256
// #pragma HLS INTERFACE s_axilite port=in1 bundle=control
// #pragma HLS INTERFACE s_axilite port=in2 bundle=control
// #pragma HLS INTERFACE s_axilite port=out bundle=control

    static hls::stream<float> in1_stream("input_stream_1");
    static hls::stream<float> in2_stream("input_stream_2");
    static hls::stream<float> out_stream("output_stream");

#pragma HLS STREAM variable=in1_stream depth=4096
#pragma HLS STREAM variable=in2_stream depth=4096
#pragma HLS STREAM variable=out_stream depth=4096

#pragma HLS dataflow
    load_input(in1, in1_stream, size);
    load_input(in2, in2_stream, size);
    compute_add(in1_stream, in2_stream, out_stream, size);
    store_result(out, out_stream, size);
}
}
