// Includes
#include <stdint.h>
#include <hls_math.h>
#include <hls_half.h>
#include "hls_print.h"

#define S_MAX 128
#define EMB 768
#define IO_MAX 1024
#define X_SIZE 98304

#define NUM_HEAD 12
#define HEAD_DIM 64
#define MAX_CACHE_SIZE 65536 // 1024*64

#define DATA_SIZE_ATTN_WEIGHT_TILED 49152
#define DATA_SIZE_ATTN_BIAS_TILED 64
#define DATA_SIZE_PROJ_WEIGHT 589824
#define DATA_SIZE_PROJ_BIAS 768

#define K_V_MAX 8192 // 128*64
#define QK_RES_MAX 16384 // 128*128
#define ONE_HEAD 192

#define NF 64
#define NX 768
#define BLOCK_SIZE 128
#define GROUP_SIZE 128

#define I_BOUND (NX / GROUP_SIZE / 2)
#define J_BOUND (GROUP_SIZE * NF / BLOCK_SIZE)
#define DATA_SIZE_WEIGHT (NX * NF)
#define I_INDEX (GROUP_SIZE * NF / BLOCK_SIZE)

typedef ap_uint<512> intpacked_t;

static void reshape(float* in, float* out_buf, int size) {
#pragma HLS inline off
change_dim:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out_buf[i] = in[i];
    }
}
static void reshape_weight(intpacked_t* qweight, intpacked_t* qzeros, float* scale, float* out_buf, int size) {
#pragma HLS inline off
#pragma HLS dataflow
change_dim:
    intpacked_t zeros_buffer;
    half scale_buffer[128];
    #pragma HLS bind_storage variable=scale_buffer type=ram_t2p impl=bram
    #pragma HLS ARRAY_PARTITION variable=scale_buffer cyclic factor=4 dim=1
    for (int i = 0; i < I_BOUND ; i++) {
        zeros_buffer = qzeros[i];
        for (int j = 0; j < 128; j++){
            #pragma HLS PIPELINE II = 1
            scale_buffer[j] = static_cast<half>(scale[i*2*NF + j]);
        }
        for(int j = 0; j < J_BOUND ; j++){
            ap_uint<512> weight = qweight[(2*i)*I_INDEX + j]; // i(group index): every 128*2304 element plus 1 = read 2304 times plus 1; j(index of loop in a group): read 1 times plus 1
            ap_uint<256> zeros = zeros_buffer.range(255,0);
            for (int k = 0; k < BLOCK_SIZE ;k++){
                #pragma HLS UNROLL factor=4 skip_exit_check
                #pragma HLS PIPELINE II = 3
                ap_uint<4> int4_weight = weight.range((k + 1) * 4 - 1, k * 4);
                ap_uint<4> int4_zero = zeros.range(((k % 64) + 1) * 4 - 1, (k % 64) * 4) + 1;
                float dequant_weight = (static_cast<float>((int4_weight - int4_zero) * (scale_buffer[k % 64])));
                out_buf[((2*i) * (GROUP_SIZE * NF) + j * BLOCK_SIZE + k)] = dequant_weight;
                hls::print("==============");
                hls::print("weight index: %d", ((2*i) * (GROUP_SIZE * NF) + j * BLOCK_SIZE + k));
                hls::print("dequant_weight: %f", out_buf[((2*i) * (GROUP_SIZE * NF) + j * BLOCK_SIZE + k)]);
            }
        }
        for(int j = 0; j < J_BOUND ; j++){
            ap_uint<512> weight = qweight[(2*i+1)*I_INDEX + j]; // i(group index): every 128*2304 element plus 1 = read 2304 times plus 1; j(index of loop in a group): read 1 times plus 1
            ap_uint<256> zeros = zeros_buffer.range(511,256);
            for (int k = 0; k < BLOCK_SIZE ;k++){
                #pragma HLS UNROLL factor=4 skip_exit_check
                #pragma HLS PIPELINE II = 3
                ap_uint<4> int4_weight = weight.range((k + 1) * 4 - 1, k * 4);
                ap_uint<4> int4_zero = zeros.range(((k % 64) + 1) * 4 - 1, (k % 64) * 4) + 1;
                float dequant_weight = (static_cast<float>((int4_weight - int4_zero) * (scale_buffer[64 + (k % 64)])));
                out_buf[((2*i+1) * (GROUP_SIZE * NF) + j * BLOCK_SIZE + k)] = dequant_weight;
                hls::print("==============");
                hls::print("weight index: %d", ((2*i+1) * (GROUP_SIZE * NF) + j * BLOCK_SIZE + k));
                hls::print("dequant_weight: %f", out_buf[((2*i+1) * (GROUP_SIZE * NF) + j * BLOCK_SIZE + k)]);
            }
        }
    }
}
static void compute_attn(float* x_buf,
                        float* bias_tiling_q_buf,
                        float* bias_tiling_k_buf,
                        float* bias_tiling_v_buf,
                        float* w_tiling_q_buf,
                        float* w_tiling_k_buf,
                        float* w_tiling_v_buf,
                        float* k_tiling_buf,
                        float* v_tiling_buf,
                        int s,
                        int iter,
                        int host_wi,
                        float* out_buf,
                        float* cur_k_buf,
                        float* cur_v_buf) {

    float q_attn[K_V_MAX];
    float qk_result[QK_RES_MAX];

#pragma HLS bind_storage variable=q_attn type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=qk_result type=RAM_T2P impl=bram

// #pragma HLS ARRAY_RESHAPE variable=q_attn type=cyclic factor=8
// #pragma HLS ARRAY_RESHAPE variable=qk_result type=cyclic factor=8


    int query_s = (iter != 0) ? 1 : s;
    int cur_s = s;

init_q_attn_cur_k_attn_cur_c_attn:
    int idx_b = host_wi*ONE_HEAD;
    for (int i = 0; i<query_s; i++) {
        for (int j = 0; j<HEAD_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
#pragma HLS loop_flatten
            int idx = i*HEAD_DIM + j;
            q_attn[idx] = bias_tiling_q_buf[j];
            cur_k_buf[idx] = bias_tiling_k_buf[j];
            cur_v_buf[idx] = bias_tiling_v_buf[j];
        }
    }

compute_q_attn_cur_k_attn_cur_c_attn:
    for (int i = 0; i<query_s; i++) {
        for (int k = 0; k<EMB; k++) {
            for (int j = 0; j<HEAD_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
#pragma HLS loop_flatten
            float x_buf_tmp = x_buf[i*EMB + k];
                int idx_w = k*HEAD_DIM + j;
                int idx_attn = i*HEAD_DIM + j;
                q_attn[idx_attn] += x_buf_tmp * w_tiling_q_buf[idx_w];
                cur_k_buf[idx_attn] += x_buf_tmp * w_tiling_k_buf[idx_w];
                cur_v_buf[idx_attn] += x_buf_tmp * w_tiling_v_buf[idx_w];
            }
        }
    }

stack_past_k_v:
    int offset = 0;
    if (iter != 0) {
        cur_s = s+1;
        offset = s*HEAD_DIM;
    }
    for (int i = 0; i<query_s; i++) {
        for (int j = 0; j<HEAD_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
#pragma HLS loop_flatten
        int idx_i = i*HEAD_DIM;
            int idx_k_v = offset + idx_i + j;
            int idx_cur = idx_i + j;
            k_tiling_buf[idx_k_v] = cur_k_buf[idx_cur];
            v_tiling_buf[idx_k_v] = cur_v_buf[idx_cur];
        }
    }
qk_matmul:
    int qk_s = query_s*cur_s;
    // if(iter != 0) qk_s = cur_s;
    // else qk_s = s*s;
init_qk_result:
    for (int i = 0; i<qk_s; i++) {
#pragma HLS PIPELINE II=1
// #pragma HLS UNROLL factor=4
        qk_result[i] = 0;
    }
compute_qk_result:
    for (int i = 0; i<query_s; i++) {
        for (int j = 0; j<cur_s; j++) {
            for (int k = 0; k<HEAD_DIM; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
#pragma HLS loop_flatten
            int idx_i = i*cur_s;
                int idx_q = i*HEAD_DIM + k;
                qk_result[idx_i + j] += q_attn[idx_q] * k_tiling_buf[k + j*HEAD_DIM];
            }
        }
    }
    for (int i = 0; i<qk_s; i++) {
#pragma HLS PIPELINE II=1
// #pragma HLS UNROLL factor=4
        qk_result[i] *= 0.125;
    }

    if (iter == 0) {
mask:
        for (int i = 0; i<s; i++) {
            for (int j = i+1; j<s; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS loop_flatten
// #pragma HLS UNROLL factor=4
                qk_result[i*s + j] = -10000000;
            }
        }
    }

softmax:
    for (int j = 0; j<query_s; j++) {
        float sum;
        float buffer[1024];
#pragma HLS bind_storage variable=buffer type=ram_t2p
        sum = 0;
        int js = j*cur_s;
        for (int i = 0; i<cur_s; i++) {
#pragma HLS PIPELINE II=1 style = frp
// #pragma HLS UNROLL factor=4
            buffer[i] = qk_result[js + i];
            sum = sum + hls::exp(buffer[i]);
        }
        for (int i = 0; i<cur_s; i++) {
#pragma HLS PIPELINE II=1 style = frp
// #pragma HLS UNROLL factor=4
            qk_result[js + i] = hls::exp(buffer[i]) / sum;
        }
    }
init_qkv:
    int qkv_size = query_s*HEAD_DIM;
    int qkv_offset = host_wi*query_s*HEAD_DIM;
    for (int i = 0; i<qkv_size; i++) {
#pragma HLS PIPELINE II=1
// #pragma HLS UNROLL factor=4
        out_buf[qkv_offset + i] = 0;
    }
compute_qkv:
    for (int i = 0; i<query_s; i++) {
        for (int k = 0; k<cur_s; k++) {
            // int idx_qk = i*cur_s + k;

            for (int j = 0; j<HEAD_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
#pragma HLS loop_flatten
            int idx_v = k*HEAD_DIM;
            float qk_res_tmp = qk_result[i*cur_s + k];
                out_buf[qkv_offset + i*HEAD_DIM + j] += qk_res_tmp * v_tiling_buf[idx_v + j];
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

void krnl_attn(float* x,
                intpacked_t* w_tiling_q_qwieght, intpacked_t*w_tiling_q_zeros, float* w_tiling_q_scale,
                intpacked_t* w_tiling_k_qwieght, intpacked_t*w_tiling_k_zeros, float* w_tiling_k_scale,
                intpacked_t* w_tiling_v_qwieght, intpacked_t*w_tiling_v_zeros, float* w_tiling_v_scale,
                float* bias_tiling_q, float* bias_tiling_k, float* bias_tiling_v,
                float* k_tiling, float* v_tiling,
                int s, int iter, int host_wi, float* out, float* cur_k, float* cur_v,
                float* w_tiling_q_buf, float* w_tiling_k_buf, float* w_tiling_v_buf) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = w_tiling_q_qwieght offset = slave bundle = gmem1 depth = DATA_SIZE_ATTN_WEIGHT_TILED/64
#pragma HLS INTERFACE m_axi port = w_tiling_q_zeros offset = slave bundle = gmem2 depth = DATA_SIZE_ATTN_WEIGHT_TILED/64/128
#pragma HLS INTERFACE m_axi port = w_tiling_q_scale offset = slave bundle = gmem3 depth = DATA_SIZE_ATTN_WEIGHT_TILED/128
#pragma HLS INTERFACE m_axi port = w_tiling_k_qwieght offset = slave bundle = gmem4 depth = DATA_SIZE_ATTN_WEIGHT_TILED/64
#pragma HLS INTERFACE m_axi port = w_tiling_k_zeros offset = slave bundle = gmem5 depth = DATA_SIZE_ATTN_WEIGHT_TILED/64/128
#pragma HLS INTERFACE m_axi port = w_tiling_k_scale offset = slave bundle = gmem6 depth = DATA_SIZE_ATTN_WEIGHT_TILED/128
#pragma HLS INTERFACE m_axi port = w_tiling_v_qwieght offset = slave bundle = gmem7 depth = DATA_SIZE_ATTN_WEIGHT_TILED/64
#pragma HLS INTERFACE m_axi port = w_tiling_v_zeros offset = slave bundle = gmem8 depth = DATA_SIZE_ATTN_WEIGHT_TILED/64/128
#pragma HLS INTERFACE m_axi port = w_tiling_v_scale offset = slave bundle = gmem9 depth = DATA_SIZE_ATTN_WEIGHT_TILED/128
#pragma HLS INTERFACE m_axi port = bias_tiling_q offset = slave bundle = gmem10 depth = DATA_SIZE_ATTN_BIAS_TILED
#pragma HLS INTERFACE m_axi port = bias_tiling_k offset = slave bundle = gmem11 depth = DATA_SIZE_ATTN_BIAS_TILED
#pragma HLS INTERFACE m_axi port = bias_tiling_v offset = slave bundle = gmem12 depth = DATA_SIZE_ATTN_BIAS_TILED
#pragma HLS INTERFACE m_axi port = k_tiling offset = slave bundle = gmem13 depth = K_V_MAX // inout
#pragma HLS INTERFACE m_axi port = v_tiling offset = slave bundle = gmem14 depth = K_V_MAX // inout
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem15 // inout
#pragma HLS INTERFACE m_axi port = w_tiling_q_buf offset = slave bundle = gmem16 // inout
#pragma HLS INTERFACE m_axi port = w_tiling_k_buf offset = slave bundle = gmem17 // inout
#pragma HLS INTERFACE m_axi port = w_tiling_v_buf offset = slave bundle = gmem18 // inout


    float x_buf[X_SIZE];
    float bias_tiling_q_buf[DATA_SIZE_ATTN_BIAS_TILED];
    float bias_tiling_k_buf[DATA_SIZE_ATTN_BIAS_TILED];
    float bias_tiling_v_buf[DATA_SIZE_ATTN_BIAS_TILED];
    // float w_tiling_q_buf[DATA_SIZE_ATTN_WEIGHT_TILED];
    // float w_tiling_k_buf[DATA_SIZE_ATTN_WEIGHT_TILED];
    // float w_tiling_v_buf[DATA_SIZE_ATTN_WEIGHT_TILED];
    float k_tiling_buf[MAX_CACHE_SIZE];
    float v_tiling_buf[MAX_CACHE_SIZE];
    float cur_k_buf[K_V_MAX];
    float cur_v_buf[K_V_MAX];
    float out_buf[X_SIZE];

#pragma HLS bind_storage variable=x_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=w_tiling_q_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=w_tiling_k_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=w_tiling_v_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=k_tiling_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=v_tiling_buf type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=cur_k_buf type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=cur_v_buf type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=out_buf type=RAM_T2P impl=bram

#pragma HLS ARRAY_RESHAPE variable=x_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=w_tiling_q_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=w_tiling_k_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=w_tiling_v_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=k_tiling_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=v_tiling_buf type=cyclic factor=8
#pragma HLS ARRAY_RESHAPE variable=out_buf type=cyclic factor=8

    int cache_out_size = (iter != 0) ? HEAD_DIM : s*HEAD_DIM;
    int query_s = (iter != 0) ? 1 : s;
    int out_size = query_s*EMB;
// #pragma HLS DATAFLOW
    reshape(x, x_buf, query_s*EMB);
    reshape(bias_tiling_q, bias_tiling_q_buf, DATA_SIZE_ATTN_BIAS_TILED);
    reshape(bias_tiling_k, bias_tiling_k_buf, DATA_SIZE_ATTN_BIAS_TILED);
    reshape(bias_tiling_v, bias_tiling_v_buf, DATA_SIZE_ATTN_BIAS_TILED);
    reshape_weight(w_tiling_q_qwieght, w_tiling_q_zeros, w_tiling_q_scale, w_tiling_q_buf, DATA_SIZE_ATTN_WEIGHT_TILED);
    reshape_weight(w_tiling_k_qwieght, w_tiling_k_zeros, w_tiling_k_scale, w_tiling_k_buf, DATA_SIZE_ATTN_WEIGHT_TILED);
    reshape_weight(w_tiling_v_qwieght, w_tiling_v_zeros, w_tiling_v_scale, w_tiling_v_buf, DATA_SIZE_ATTN_WEIGHT_TILED);
    hls::print("start deq");
    // reshape_weight(w_tiling_q_qwieght, w_tiling_q_zeros, w_tiling_q_scale, deq_q, DATA_SIZE_ATTN_WEIGHT_TILED);
    // reshape_weight(w_tiling_k_qwieght, w_tiling_k_zeros, w_tiling_k_scale, deq_k, DATA_SIZE_ATTN_WEIGHT_TILED);
    // reshape_weight(w_tiling_v_qwieght, w_tiling_v_zeros, w_tiling_v_scale, deq_v, DATA_SIZE_ATTN_WEIGHT_TILED);
    reshape(k_tiling, k_tiling_buf, s*HEAD_DIM);
    reshape(v_tiling, v_tiling_buf, s*HEAD_DIM);

    compute_attn(x_buf, bias_tiling_q_buf, bias_tiling_k_buf, bias_tiling_v_buf,
                w_tiling_q_buf, w_tiling_k_buf, w_tiling_v_buf,
                k_tiling_buf, v_tiling_buf,
                s, iter, host_wi, out_buf, cur_k_buf, cur_v_buf);

    store_result(out_buf, out, out_size);
    store_result(cur_k_buf, cur_k, cache_out_size); // 存12次
    store_result(cur_v_buf, cur_v, cache_out_size); // 存12次
    // hls::print("=======q_weight=======");
    // for (int i = 0; i < DATA_SIZE_ATTN_WEIGHT_TILED; i++){
    //     hls::print("index: %d",i);
    //     hls::print("q_dequant_weight: %f", w_tiling_q_buf[i]);
    // }
    // hls::print("=======k_weight=======");
    // for (int i = 0; i < DATA_SIZE_ATTN_WEIGHT_TILED; i++){
    //     hls::print("index: %d",i);
    //     hls::print("k_dequant_weight: %f", w_tiling_k_buf[i]);
    // }
    // hls::print("=======v_weight=======");
    // for (int i = 0; i < DATA_SIZE_ATTN_WEIGHT_TILED; i++){
    //     hls::print("index: %d",i);
    //     hls::print("v_dequant_weight: %f", w_tiling_v_buf[i]);
    // }
}
}