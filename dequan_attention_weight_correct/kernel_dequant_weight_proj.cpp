// Includes
#include <stdint.h>
#include <hls_math.h>
#include <hls_half.h>
#include <ap_int.h>
#include "hls_print.h"

#define S_MAX 128
#define EMB 768
#define IO_MAX 1024
#define X_SIZE 98304

#define NUM_HEAD 12
#define HEAD_DIM 64

#define DATA_SIZE_ATTN_WEIGHT_TILED 49152
#define DATA_SIZE_ATTN_BIAS_TILED 64
#define DATA_SIZE_PROJ_WEIGHT 589824
#define DATA_SIZE_PROJ_BIAS 768

#define K_V_MAX 8192 // 128*64
#define QK_RES_MAX 16384 // 128*128
#define ONE_HEAD 192

#define NF 768
#define NX 768
#define BLOCK_SIZE 128
#define GROUP_SIZE 128
#define DATA_SIZE_PROJ_WEIGHT 589824
#define DATA_SIZE_PROJ_BIAS 768

#define I_BOUND (NX / GROUP_SIZE)
#define J_BOUND (GROUP_SIZE * NF / BLOCK_SIZE)
#define ZREO_BUFFER_SIZE (NF / BLOCK_SIZE)
#define DATA_SIZE_WEIGHT (NX * NF)
#define I_INDEX (DATA_SIZE_WEIGHT / BLOCK_SIZE / (NX / GROUP_SIZE))

static void reshape(float* in, float* out_buf, int size) {
#pragma HLS inline off
change_dim:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out_buf[i] = in[i];
    }
}
static void compute_attn(float* qkv_res_buf,
                        float* c_proj_weight_buf,
                        float* c_proj_bias_buf,
                        int s,
                        float* out_buf) {

#pragma HLS DATAFLOW
    ap_uint<512> zeros_buffer[ZREO_BUFFER_SIZE];
    half scale_buffer[NF];
    #pragma HLS bind_storage variable=zeros_buffer type=ram_t2p
    #pragma HLS bind_storage variable=scale_buffer type=ram_t2p
    for (int i = 0; i < I_BOUND ; i++) {
        #pragma HLS dataflow
        for (int j = 0; j < ZREO_BUFFER_SIZE ; j++){
            #pragma HLS PIPELINE II = 1  style = frp
            zeros_buffer[j] = qzeros[i*ZREO_BUFFER_SIZE + j];
        }
        for (int j = 0; j < NF; j++){
            #pragma HLS PIPELINE II = 1  style = frp
            scale_buffer[j] = static_cast<half>(scale[i*NF + j]);
        }
        for(int j = 0; j < J_BOUND ; j++){
            #pragma HLS PIPELINE II = 1  style = frp
            ap_uint<512> weight = qweight[i*I_INDEX + j]; // i(group index): every 128*2304 element plus 1 = read 2304 times plus 1; j(index of loop in a group): read 1 times plus 1
            ap_uint<512> zero = zeros_buffer[(j * BLOCK_SIZE % NF) / BLOCK_SIZE];
            for (int k = 0; k < BLOCK_SIZE ;k++){
                #pragma HLS loop_flatten
                ap_uint<4> int4_weight = weight.range((k + 1) * 4 - 1, k * 4);
                ap_uint<4> int4_zero = zero.range((k + 1) * 4 - 1, k * 4) + 1;
                float dequant_weight = (static_cast<float>((int4_weight - int4_zero) * (scale_buffer[(j * BLOCK_SIZE + k) % NF])));
                out_buf[(i * (GROUP_SIZE * NF) + j * BLOCK_SIZE + k)] = dequant_weight_int;
            }
        }
    }
}


static void store_result(float* buf, float* out, int size) {
#pragma HLS inline off
mem_wr:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = static_cast<half>(buf[i]);
    }
}

extern "C" {

void krnl_c_proj(float* qkv_res, float* c_proj_weight, float* c_proj_bias, int s, float* out) {
// #pragma HLS INTERFACE m_axi port = qkv_res offset = slave bundle = gmem0 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = c_proj_weight offset = slave bundle = gmem1 depth = DATA_SIZE_PROJ_WEIGHT // inout
// #pragma HLS INTERFACE m_axi port = c_proj_bias offset = slave bundle = gmem2 depth = DATA_SIZE_PROJ_BIAS // inout
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem3


    // float qkv_res_buf[X_SIZE];
    float c_proj_weight_buf[DATA_SIZE_PROJ_WEIGHT];
    // float c_proj_bias_buf[DATA_SIZE_PROJ_BIAS];
    float out_buf[X_SIZE];

// #pragma HLS bind_storage variable=qkv_res_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=c_proj_weight_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=out_buf type=RAM_T2P impl=bram

#pragma HLS ARRAY_RESHAPE variable=c_proj_weight_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=c_proj_bias_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=out_buf type=cyclic factor=8


#pragma HLS DATAFLOW

    // reshape(qkv_res, qkv_res_buf, s*EMB);
    reshape(c_proj_weight, c_proj_weight_buf, DATA_SIZE_PROJ_WEIGHT);
    // reshape(c_proj_bias, c_proj_bias_buf, DATA_SIZE_PROJ_BIAS);

    compute_attn(c_proj_weight_buf, s, out_buf);

    store_result(out_buf, out, s*EMB);

}
}
