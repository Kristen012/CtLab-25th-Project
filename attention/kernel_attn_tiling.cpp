// Includes
#include <stdint.h>
#include <hls_math.h>

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

#define DATA_SIZE_PROJ_WEIGHT 589824
#define DATA_SIZE_PROJ_BIAS 768

static void reshape(float* in, float* out_buf, int size) {
#pragma HLS inline off
change_dim:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out_buf[i] = in[i];
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
            int idx = i*HEAD_DIM + j;
            q_attn[idx] = bias_tiling_q_buf[j];
            cur_k_buf[idx] = bias_tiling_k_buf[j];
            cur_v_buf[idx] = bias_tiling_v_buf[j];
        }
    }

compute_q_attn_cur_k_attn_cur_c_attn:
    for (int i = 0; i<query_s; i++) {
        for (int k = 0; k<EMB; k++) {
            // int idx_x = i*EMB + k;
            float x_buf_tmp = x_buf[i*EMB + k];
            for (int j = 0; j<HEAD_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
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
        int idx_i = i*HEAD_DIM;
        for (int j = 0; j<HEAD_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
            int idx_k_v = offset + idx_i + j;
            int idx_cur = idx_i + j;
            k_tiling_buf[idx_k_v] = cur_k_buf[idx_cur];
            v_tiling_buf[idx_k_v] = cur_v_buf[idx_cur];
        }
    }
qk_matmul:
    int qk_s;
    if(iter != 0) qk_s = cur_s;
    else qk_s = s*s;
init_qk_result:
    for (int i = 0; i<qk_s; i++) {
#pragma HLS PIPELINE II=1
// #pragma HLS UNROLL factor=4
        qk_result[i] = 0;
    }
compute_qk_result:
    for (int i = 0; i<query_s; i++) {
        for (int j = 0; j<cur_s; j++) {
            int idx_i = i*cur_s;
            for (int k = 0; k<HEAD_DIM; k++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
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
            int idx_v = k*HEAD_DIM;
            float qk_res_tmp = qk_result[i*cur_s + k];
            for (int j = 0; j<HEAD_DIM; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=8
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
        out[i] = buf[i];
    }
}

extern "C" {

void krnl_attn(float* x, float* w_tiling_q, float* w_tiling_k, float* w_tiling_v,
                float* bias_tiling_q, float* bias_tiling_k, float* bias_tiling_v,
                float* k_tiling, float* v_tiling,
                int s, int iter, int host_wi, float* out, float* cur_k, float* cur_v) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = w_tiling_q offset = slave bundle = gmem1 depth = DATA_SIZE_ATTN_WEIGHT_TILED
#pragma HLS INTERFACE m_axi port = w_tiling_k offset = slave bundle = gmem2 depth = DATA_SIZE_ATTN_WEIGHT_TILED
#pragma HLS INTERFACE m_axi port = w_tiling_v offset = slave bundle = gmem3 depth = DATA_SIZE_ATTN_WEIGHT_TILED
#pragma HLS INTERFACE m_axi port = bias_tiling_q offset = slave bundle = gmem4 depth = DATA_SIZE_ATTN_BIAS_TILED
#pragma HLS INTERFACE m_axi port = bias_tiling_k offset = slave bundle = gmem4 depth = DATA_SIZE_ATTN_BIAS_TILED
#pragma HLS INTERFACE m_axi port = bias_tiling_v offset = slave bundle = gmem4 depth = DATA_SIZE_ATTN_BIAS_TILED
#pragma HLS INTERFACE m_axi port = k_tiling offset = slave bundle = gmem5 depth = K_V_MAX // inout
#pragma HLS INTERFACE m_axi port = v_tiling offset = slave bundle = gmem6 depth = K_V_MAX // inout
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem9 // inout


    float x_buf[X_SIZE];
    float bias_tiling_q_buf[DATA_SIZE_ATTN_BIAS_TILED];
    float bias_tiling_k_buf[DATA_SIZE_ATTN_BIAS_TILED];
    float bias_tiling_v_buf[DATA_SIZE_ATTN_BIAS_TILED];
    float w_tiling_q_buf[DATA_SIZE_ATTN_WEIGHT_TILED];
    float w_tiling_k_buf[DATA_SIZE_ATTN_WEIGHT_TILED];
    float w_tiling_v_buf[DATA_SIZE_ATTN_WEIGHT_TILED];
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

// #pragma HLS ARRAY_RESHAPE variable=x_buf type=cyclic factor=8
// #pragma HLS ARRAY_RESHAPE variable=w_tiling_q_buf type=cyclic factor=8
// #pragma HLS ARRAY_RESHAPE variable=w_tiling_k_buf type=cyclic factor=8
// #pragma HLS ARRAY_RESHAPE variable=w_tiling_v_buf type=cyclic factor=8
// #pragma HLS ARRAY_RESHAPE variable=k_tiling_buf type=cyclic factor=8
// #pragma HLS ARRAY_RESHAPE variable=v_tiling_buf type=cyclic factor=8
// #pragma HLS ARRAY_RESHAPE variable=out_buf type=cyclic factor=8

    int cache_out_size = (iter != 0) ? HEAD_DIM : s*HEAD_DIM;
    int query_s = (iter != 0) ? 1 : s;
    int out_size = query_s*EMB;
// #pragma HLS DATAFLOW

    reshape(x, x_buf, query_s*EMB);
    reshape(bias_tiling_q, bias_tiling_q_buf, DATA_SIZE_ATTN_BIAS_TILED);
    reshape(bias_tiling_k, bias_tiling_k_buf, DATA_SIZE_ATTN_BIAS_TILED);
    reshape(bias_tiling_v, bias_tiling_v_buf, DATA_SIZE_ATTN_BIAS_TILED);
    reshape(w_tiling_q, w_tiling_q_buf, DATA_SIZE_ATTN_WEIGHT_TILED);
    reshape(w_tiling_k, w_tiling_k_buf, DATA_SIZE_ATTN_WEIGHT_TILED);
    reshape(w_tiling_v, w_tiling_v_buf, DATA_SIZE_ATTN_WEIGHT_TILED);
    reshape(k_tiling, k_tiling_buf, s*HEAD_DIM);
    reshape(v_tiling, v_tiling_buf, s*HEAD_DIM);

    compute_attn(x_buf, bias_tiling_q_buf, bias_tiling_k_buf, bias_tiling_v_buf,
                w_tiling_q_buf, w_tiling_k_buf, w_tiling_v_buf,
                k_tiling_buf, v_tiling_buf,
                s, iter, host_wi, out_buf, cur_k_buf, cur_v_buf);

    store_result(out_buf, out, out_size);
    store_result(cur_k_buf, cur_k, cache_out_size); // 存12次
    store_result(cur_v_buf, cur_v, cache_out_size); // 存12次

}
}
