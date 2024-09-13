#include <iostream>
#include <fstream>
#include <vector>
#include <string>
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

#define DATA_SIZE_PROJ_WEIGHT 589824
#define DATA_SIZE_PROJ_BIAS 768
#define MAX_CACHE_SIZE 65536 // 1024*64

using namespace std;
// float query[97536+1];
// float q_tmp[97536+1];
float golden[98304+1];
float weight[1769472+1]; // 768*2304
float bias[2304+1];
float x_buf[X_SIZE+5];
float bias_tiling_q_buf[DATA_SIZE_ATTN_BIAS_TILED+5];
float bias_tiling_k_buf[DATA_SIZE_ATTN_BIAS_TILED+5];
float bias_tiling_v_buf[DATA_SIZE_ATTN_BIAS_TILED+5];
float w_tiling_q_buf[DATA_SIZE_ATTN_WEIGHT_TILED+5];
float w_tiling_k_buf[DATA_SIZE_ATTN_WEIGHT_TILED+5];
float w_tiling_v_buf[DATA_SIZE_ATTN_WEIGHT_TILED+5];
float c_proj_weight_buf[DATA_SIZE_PROJ_WEIGHT+5];
float c_proj_bias_buf[DATA_SIZE_PROJ_BIAS+5];
float k_tiling_buf[MAX_CACHE_SIZE+5];
float v_tiling_buf[MAX_CACHE_SIZE+5];
float out_buf[X_SIZE+5];
float q_attn[K_V_MAX+5];
float cur_k_attn[K_V_MAX+5];
float cur_v_attn[K_V_MAX+5];
float qk_result[QK_RES_MAX+5];
float qkv_res[X_SIZE+5];

float k_cache[12][786432]; // 12, 1024, 64
float v_cache[12][786432]; // 12, 1024, 64
int idx_cache = 0;
void upd_kv_cache(float* cur_k, float* cur_v, int block, int s, int head) {
	// int query_s = (iter != 0) ? 1 : s;
	// int max_i = query_s*64;
	for (int i = 0; i<s*64; i++) {
		k_cache[block][idx_cache] = cur_k[i];
		v_cache[block][idx_cache] = cur_v[i];
        idx_cache++;
        // cout << cur_k[i] << endl;
	}

}
int main() {
    // read q, k into a single 1d array
    for (int iter = 0; iter < 2; iter++) {
        int cnt_w = 0;
        int cnt_b = 0;
        int cnt_x = 0;
        int cnt_golden = 0;
        int cnt_proj_w = 0;
        int cnt_proj_b = 0;
        vector<string> file_name_iter_1 = {"ln1_out_iter_1_block_1.txt", "transformer.h.0.attn.c_attn.weight_split_head.txt", "transformer.h.0.attn.c_attn.bias_split_head.txt", "transformer.h.0.attn.c_proj.bias.txt", "transformer.h.0.attn.c_proj.weight.txt", "attn_output_iter_1_block_1.txt"};
        vector<string> file_name_iter_2 = {"ln1_out_iter_2_block_1.txt", "transformer.h.0.attn.c_attn.weight_split_head.txt", "transformer.h.0.attn.c_attn.bias_split_head.txt", "transformer.h.0.attn.c_proj.bias.txt", "transformer.h.0.attn.c_proj.weight.txt", "attn_cproj_output_iter_2_block_1.txt"};
        vector<string> file_name = (iter == 0) ? file_name_iter_1 : file_name_iter_2;
        for(int i = 0; i<file_name.size(); i++) {
            ifstream infile(file_name[i]);
            if(!infile) {
                cerr << "unable to open file " << file_name[i] << endl;
                return 1;
            }

            string line;
            
            while (getline(infile, line)) {
                if(i == 0) {
                    x_buf[cnt_x++] = stof(line);
                }
                else if(i == 1) {
                    weight[cnt_w++] = stof(line);
                }
                else if(i == 2){
                    bias[cnt_b++] = stof(line);
                } 
                else if(i == 3) {
                    c_proj_bias_buf[cnt_proj_b++] = stof(line);
                }
                else if(i == 4) {
                    c_proj_weight_buf[cnt_proj_w++] = stof(line);
                }
                else {
                    golden[cnt_golden++] = stof(line);
                }
            }
            infile.close();
        }        

        // 要在krnl外初始化
        // for (int i = 0; i<12*127*64; i++) qkv_res[i] = 0;
        int s = 127;
        int query_s = (iter != 0) ? 1 : s;    
        
        int head = (iter == 0) ? 12 : 1;
        for (int host_wi = 0; host_wi < 12; host_wi++) {
            if(iter == 1) {
                for (int i = 0; i<s*64; i++) {
                    k_tiling_buf[i] = k_cache[0][host_wi*s*64 + i];
                    v_tiling_buf[i] = v_cache[0][host_wi*s*64 + i];
                    // cout << k_tiling_buf[i] << endl;
                }
            }
            // int host_wi = 0;
            // 拿一個head，做一次完整的qkv
            for (int i = 0; i<768; i++) {
                for (int j = 0; j<64; j++) {
                    w_tiling_q_buf[i*64 + j] = weight[host_wi*192 + i*2304 + j];
                }
            }
            for (int i = 0; i<768; i++) {
                for (int j = 0; j<64; j++) {
                    w_tiling_k_buf[i*64 + j] = weight[host_wi*192 + i*2304 + 64 + j];
                }
            }
            for (int i = 0; i<768; i++) {
                for (int j = 0; j<64; j++) {
                    w_tiling_v_buf[i*64 + j] = weight[host_wi*192 + i*2304 + 128 + j];
                }
            }
            // cout << w_tiling_q[0] << " " << w_tiling_q[1] << endl;
            // cout << w_tiling_k[0] << " " << w_tiling_k[1] << endl;
            // cout << w_tiling_v[0] << " " << w_tiling_v[1] << endl;

            // bias
            for (int i = 0; i<64; i++) {
                bias_tiling_q_buf[i] = bias[host_wi*192 + i];
                bias_tiling_k_buf[i] = bias[host_wi*192 + 64 + i];
                bias_tiling_v_buf[i] = bias[host_wi*192 + 128 + i];
            }
            // cout << bias_tiling_q[0] << " " << bias_tiling_q[1] << endl;
            // cout << bias_tiling_k[0] << " " << bias_tiling_k[1] << endl;
            // cout << bias_tiling_v[0] << " " << bias_tiling_v[1] << endl;
            
            
            int cur_s = s;

            // q_attn, k_attn, c_attn
                int idx_b = host_wi*ONE_HEAD;
            for (int i = 0; i<query_s; i++) {
                for (int j = 0; j<HEAD_DIM; j++) {
                    int idx = i*HEAD_DIM + j;
                    q_attn[idx] = bias_tiling_q_buf[j];
                    cur_k_attn[idx] = bias_tiling_k_buf[j];
                    cur_v_attn[idx] = bias_tiling_v_buf[j];
                }
            }
            for (int i = 0; i<query_s; i++) {
                for (int k = 0; k<EMB; k++) {
                    int idx_x = i*EMB + k;
                    for (int j = 0; j<HEAD_DIM; j++) {
                        int idx_w = k*HEAD_DIM + j;
                        int idx_attn = i*HEAD_DIM + j;
                        q_attn[idx_attn] += x_buf[idx_x] * w_tiling_q_buf[idx_w];
                        cur_k_attn[idx_attn] += x_buf[idx_x] * w_tiling_k_buf[idx_w];
                        cur_v_attn[idx_attn] += x_buf[idx_x] * w_tiling_v_buf[idx_w];
                    }
                }
            }
            int offset = 0;
            if (iter != 0) {
                cur_s = s+1;
                offset = s*HEAD_DIM;
            }
            for (int i = 0; i<query_s; i++) {
                int idx_i = i*HEAD_DIM;
                for (int j = 0; j<HEAD_DIM; j++) {
                    int idx_k_v = offset + idx_i + j;
                    int idx_cur = idx_i + j;
                    k_tiling_buf[idx_k_v] = cur_k_attn[idx_cur];
                    v_tiling_buf[idx_k_v] = cur_v_attn[idx_cur];
                }
            }
            // cout << v_tiling_buf[0] << endl;
            upd_kv_cache(cur_k_attn, cur_v_attn, 0, cur_s, host_wi);
            // if(iter == 1) { // false
            //     for (int i = 0; i<127*64; i++) cout << k_tiling_buf[i] << endl;
            // }
            int qk_s;
            if(iter != 0) qk_s = cur_s;
            else qk_s = s*s;
            for (int i = 0; i<qk_s; i++) qk_result[i] = 0; 
            for (int i = 0; i<query_s; i++) {
                for (int k = 0; k<HEAD_DIM; k++) {
                    int idx_q = i*HEAD_DIM + k;
                    int idx_i = i*cur_s;
                    for (int j = 0; j<cur_s; j++) {
                        qk_result[idx_i + j] += q_attn[idx_q] * k_tiling_buf[k + j*HEAD_DIM];
                    }
                }
            }
            for (int i = 0; i<qk_s; i++) {
                qk_result[i] *= 0.125;
            }
            // if(iter == 1) { // false
            //     for (int i = 0; i<qk_s; i++) {
            //         // cout << "host_wi: " << host_wi << endl;
            //         cout << qk_result[i] << endl;
            //     }
            // }
            
            if(iter == 0) {
                for (int i = 0; i<127; i++) {
                    for (int j = i+1; j<127; j++) {
                        qk_result[i*127 + j] = -INFINITY;
                    }
                }
            }

            for (int j = 0; j<query_s; j++) {
                float sum;
                float buffer[S_MAX+1];
                sum = 0;
                int js = j*cur_s;
                for (int i = 0; i<cur_s; i++) {
                    buffer[i] = qk_result[js + i];
                    sum = sum + exp(buffer[i]);
                }
                for (int i = 0; i<cur_s; i++) {
                    qk_result[js + i] = exp(buffer[i]) / sum;
                }
            }

            int qkv_size = query_s*HEAD_DIM;
            int qkv_offset = host_wi*query_s*HEAD_DIM;
            for (int i = 0; i<qkv_size; i++) {
                qkv_res[qkv_offset + i] = 0;
            }
            for (int i = 0; i<query_s; i++) {
                for (int k = 0; k<cur_s; k++) {
                    int idx_qk = i*cur_s + k;
                    int idx_v = k*HEAD_DIM;
                    for (int j = 0; j<HEAD_DIM; j++) {
                        qkv_res[qkv_offset + i*HEAD_DIM + j] += qk_result[idx_qk] * v_tiling_buf[idx_v + j];
                    }
                }
            }
        }

        for (int i = 0; i<query_s; i++) {
            for (int j = 0; j<EMB; j++) {
                out_buf[i*EMB + j] = c_proj_bias_buf[j];
            }
        }
        for (int k = 0; k<NUM_HEAD; k++) {
            for (int i = 0; i<query_s; i++) {
                int idx_o = i*EMB;
                for (int m = 0; m<HEAD_DIM; m++) {
                    int qkv_idx = k*query_s*HEAD_DIM + i*HEAD_DIM + m;
                    for (int j = 0; j<EMB; j++) {
                        int weight_idx = (k*HEAD_DIM + m)*EMB + j;
                        out_buf[idx_o + j] += qkv_res[qkv_idx] * c_proj_weight_buf[weight_idx];
                    }
                }
            }
        }
        // if(iter == 1){
            cout << "=====ITER" << iter << "=====" << endl;
            
            for (int i = 0; i<768; i++) {
                if(abs(out_buf[i] - golden[i]) > 0.006)
                    cout << out_buf[i] << " " << golden[i] << endl;
            }               
        // }        
     
    }

    // float cnt = 0;
    // for (int i = 0; i<97536; i++) {
    //     cnt += abs(out[i] - golden[i]);
    // }
    // cnt /= 97536;
    // cout << cnt << endl;


    return 0;
}