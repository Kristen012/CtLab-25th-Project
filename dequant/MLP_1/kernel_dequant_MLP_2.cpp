#include <stdint.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_int.h>
#include <hls_half.h>
#include "hls_print.h"
// Define constants
#define NX 768
#define NF 3072
#define DEPTH_HEIGHT 128
#define GROUP_SIZE 128
#define BLOCK_SIZE (512/4)

#define DATA_SIZE_X (DEPTH_HEIGHT * NX)
#define DATA_SIZE_WEIGHT (NX * NF)
#define DATA_SIZE_BIAS 3072
#define DATA_SIZE_RES (DEPTH_HEIGHT * NX)
#define a_max (DEPTH_HEIGHT * NX)
#define b_max (NX * NF)
#define gelu_max (DEPTH_HEIGHT * NF)

// Define packed type for 2 floats in one 64-bit slot
typedef ap_uint<64> packed_t;

// Load and reshape data to packed format after dequanting the weights
static void reshape_weight(ap_uint<512>* qweight, ap_uint<512>* qzeros, float* scale, float* output) {
    ap_uint<512> zeros_buffer[NX / BLOCK_SIZE];
    half scale_buffer[NX];
    #pragma HLS bind_storage variable=zeros_buffer type=ram_t2p
    #pragma HLS bind_storage variable=scale_buffer type=ram_t2p
    #pragma HLS dataflow
    for (int i = 0; i < NF / GROUP_SIZE ; i++) {
        #pragma HLS dataflow
        for (int j = 0; j < NX / BLOCK_SIZE ; j++){
            #pragma HLS PIPELINE II = 1  style = frp
            zeros_buffer[j] = qzeros[i*(NX / BLOCK_SIZE) + j];
        }
        for (int j = 0; j < NX; j++){
            #pragma HLS PIPELINE II = 1  style = frp
            scale_buffer[j] = static_cast<half>(scale[i*NX + j]);
        }
        // i: # of groups
        // j: # of read times in a group
        // k index of data  in a read
        for(int j = 0; j < GROUP_SIZE * NX / BLOCK_SIZE ; j++){
            #pragma HLS PIPELINE II = 1  style = frp
            ap_uint<512> weight = qweight[i*DATA_SIZE_WEIGHT / BLOCK_SIZE / (NF / GROUP_SIZE) + j]; // i(group index): every 128*2304 element plus 1 = read 2304 times plus 1; j(index of loop in a group): read 1 times plus 1
            ap_uint<512> zero = zeros_buffer[(j * BLOCK_SIZE % NX) / BLOCK_SIZE];
            for (int k = 0; k < BLOCK_SIZE ;k++){
                #pragma HLS loop_flatten
                ap_uint<4> int4_weight = weight.range((k + 1) * 4 - 1, k * 4);
                ap_uint<4> int4_zero = zero.range((k + 1) * 4 - 1, k * 4) + 1;
                output[i * (GROUP_SIZE * NX) + j * BLOCK_SIZE + k] = static_cast<float>((int4_weight - int4_zero) * (scale_buffer[(j * BLOCK_SIZE + k) % NX]));
            }
        }
    }
}


extern "C" {
void krnl_dequant(ap_uint<512>* qweight, ap_uint<512>* qzeros, float* scale, float* out, int size) {
// #pragma HLS INTERFACE m_axi port = size bundle = gmem0 depth = 1
#pragma HLS INTERFACE m_axi port = qweight bundle = gmem0 depth = (96*3072)
#pragma HLS INTERFACE m_axi port = qzeros bundle = gmem1 depth = (6*24)
#pragma HLS INTERFACE m_axi port = scale bundle = gmem2 depth = (6*3072)
#pragma HLS INTERFACE m_axi port = out bundle = gmem3 depth = (768*2304)
#pragma HLS INTERFACE s_axilite port=qweight bundle=control
#pragma HLS INTERFACE s_axilite port=qzeros bundle=control
#pragma HLS INTERFACE s_axilite port=scale bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=size bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control
    
	reshape_weight(qweight, qzeros, scale, out);
}
}