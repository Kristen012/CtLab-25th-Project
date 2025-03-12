#include <iostream>
#include <vector>
#include <cmath>

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

void reshape(const std::vector<float>& in, std::vector<float>& out, int size) {
    out.assign(in.begin(), in.begin() + size);
}

void compute_attn(std::vector<float>& x_buf,
                  std::vector<float>& bias_tiling_q_buf,
                  std::vector<float>& bias_tiling_k_buf,
                  std::vector<float>& bias_tiling_v_buf,
                  std::vector<float>& w_tiling_q_buf,
                  std::vector<float>& w_tiling_k_buf,
                  std::vector<float>& w_tiling_v_buf,
                  int s,
                  int iter,
                  int host_wi,
                  std::vector<float>& out_buf) {

    std::vector<float> q_attn(K_V_MAX, 0);
    std::vector<float> cur_k_buf(K_V_MAX, 0);
    std::vector<float> cur_v_buf(K_V_MAX, 0);
    std::vector<float> qk_result(QK_RES_MAX, 0);

    int query_s = s;
    int cur_s = s;

    for (int i = 0; i < query_s; i++) {
        for (int j = 0; j < HEAD_DIM; j++) {
            int idx = i * HEAD_DIM + j;
            q_attn[idx] = bias_tiling_q_buf[j];
            cur_k_buf[idx] = bias_tiling_k_buf[j];
            cur_v_buf[idx] = bias_tiling_v_buf[j];
        }
    }

    for (int i = 0; i < query_s; i++) {
        for (int k = 0; k < EMB; k++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                int idx_w = k * HEAD_DIM + j;
                int idx_attn = i * HEAD_DIM + j;
                float x_buf_tmp = x_buf[i * EMB + k];
                q_attn[idx_attn] += x_buf_tmp * w_tiling_q_buf[idx_w];
                cur_k_buf[idx_attn] += x_buf_tmp * w_tiling_k_buf[idx_w];
                cur_v_buf[idx_attn] += x_buf_tmp * w_tiling_v_buf[idx_w];
            }
        }
    }

    int qk_s = query_s * cur_s;
    for (int i = 0; i < query_s; i++) {
        for (int j = 0; j < cur_s; j++) {
            for (int k = 0; k < HEAD_DIM; k++) {
                qk_result[i * cur_s + j] += q_attn[i * HEAD_DIM + k] * cur_k_buf[k + j * HEAD_DIM];
            }
        }
    }

    for (int i = 0; i < qk_s; i++) {
        qk_result[i] *= 0.125;
    }

    if (iter == 0) {
        for (int i = 0; i < s; i++) {
            for (int j = i + 1; j < s; j++) {
                qk_result[i * s + j] = -10000000;
            }
        }
    }

    for (int j = 0; j < query_s; j++) {
        float sum = 0;
        float buffer[1024];
        int js = j*cur_s;
        for (int i = 0; i < cur_s; i++) {
            buffer[i] = qk_result[js + i];
            sum += exp(buffer[i]);
        }
        for (int i = 0; i < cur_s; i++) {
            qk_result[js + i] = exp(buffer[i]) / sum;
        }
    }

    for (int i = 0; i < query_s * HEAD_DIM; i++) {
        out_buf[host_wi * query_s * HEAD_DIM + i] = 0;
    }

    for (int i = 0; i < query_s; i++) {
        for (int k = 0; k < cur_s; k++) {
            for (int j = 0; j < HEAD_DIM; j++) {
                out_buf[host_wi * query_s * HEAD_DIM + i * HEAD_DIM + j] += qk_result[i * cur_s + k] * cur_v_buf[k * HEAD_DIM + j];
            }
        }
    }
}

void compute_c_proj(std::vector<float>& qkv_res_buf,
                    std::vector<float>& c_proj_weight_buf,
                    std::vector<float>& c_proj_bias_buf,
                    int s,
                    std::vector<float>& out_buf) {

    for (int i = 0; i < s; i++) {
        for (int j = 0; j < EMB; j++) {
            out_buf[i * EMB + j] = c_proj_bias_buf[j];
        }
    }

    for (int k = 0; k < NUM_HEAD; k++) {
        for (int i = 0; i < s; i++) {
            for (int m = 0; m < HEAD_DIM; m++) {
                for (int j = 0; j < EMB; j++) {
                    int idx_o = i * EMB;
                    int qkv_idx = k * s * HEAD_DIM + i * HEAD_DIM + m;
                    float qkv_res_buf_tmp = qkv_res_buf[qkv_idx];
                    int weight_idx = (k * HEAD_DIM + m) * EMB + j;
                    out_buf[idx_o + j] += qkv_res_buf_tmp * c_proj_weight_buf[weight_idx];
                }
            }
        }
    }
}

int main() {
    std::vector<float> x(X_SIZE, 1.0f);
    std::vector<float> w_tiling_q(DATA_SIZE_ATTN_WEIGHT_TILED, 1.0f);
    std::vector<float> w_tiling_k(DATA_SIZE_ATTN_WEIGHT_TILED, 1.0f);
    std::vector<float> w_tiling_v(DATA_SIZE_ATTN_WEIGHT_TILED, 1.0f);
    std::vector<float> bias_tiling_q(DATA_SIZE_ATTN_BIAS_TILED, 1.0f);
    std::vector<float> bias_tiling_k(DATA_SIZE_ATTN_BIAS_TILED, 1.0f);
    std::vector<float> bias_tiling_v(DATA_SIZE_ATTN_BIAS_TILED, 1.0f);
    std::vector<float> out_buf(X_SIZE, 0.0f);

    int s = 128, iter = 0, host_wi = 0;
    compute_attn(x, bias_tiling_q, bias_tiling_k, bias_tiling_v, w_tiling_q, w_tiling_k, w_tiling_v,
                    s, iter, host_wi, out_buf);

    compute_c_proj(out_buf, w_tiling_q, bias_tiling_q, s, out_buf);

    return 0;
}