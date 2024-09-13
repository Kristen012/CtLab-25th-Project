#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
// float query[97536+1];
// float q_tmp[97536+1];
float golden[98304+1];
float weight[1769472+1]; // 768*2304
float bias[2304+1];
// float c_attn[294912+1]; 
// float key[97536+1];
// float value[97536+1];
// float attn_result[292608+1];

// hls use
float x[98304+1]; // 128*768
float w_tiling_q[49152+1];
float w_tiling_k[49152+1];
float w_tiling_v[49152+1];
float bias_tiling_q[64+1];
float bias_tiling_k[64+1];
float bias_tiling_v[64+1];

float q_attn[8192+1];
float k_attn[8192+1];
float v_attn[8192+1];

float qk_result[16384+1];
float qkv_res[98304+1]; // 可以reuse x
float proj_weight[589824+1];
float proj_bias[768+1];
float out[98304+1]; // 可以reuse x

int main() {
    // read q, k into a single 1d array
    int cnt_w = 0;
    int cnt_b = 0;
    int cnt_x = 0;
    int cnt_golden = 0;
    int cnt_proj_w = 0;
    int cnt_proj_b = 0;
    vector<string> file_name = {"ln1_out_iter_1_block_1.txt", "transformer.h.0.attn.c_attn.weight_split_head.txt", "transformer.h.0.attn.c_attn.bias_split_head.txt", "transformer.h.0.attn.c_proj.bias.txt", "transformer.h.0.attn.c_proj.weight.txt", "softmax_output_iter_1_block_1.txt"};
    for(int i = 0; i<file_name.size(); i++) {
        ifstream infile(file_name[i]);
        if(!infile) {
            cerr << "unable to open file " << file_name[i] << endl;
            return 1;
        }

        string line;
        
        while (getline(infile, line)) {
            if(i == 0) {
                x[cnt_x++] = stof(line); 
            }
            else if(i == 1) {
                weight[cnt_w++] = stof(line);
            }
            else if(i == 2){
                bias[cnt_b++] = stof(line);
            } 
            else if(i == 3) {
                proj_bias[cnt_proj_b++] = stof(line);
            }
            else if(i == 4) {
                proj_weight[cnt_proj_w++] = stof(line);
            }
            else {
                golden[cnt_golden++] = stof(line);
            }
        }
        infile.close();
    }

    // 要在krnl外初始化
    for (int i = 0; i<12*127*64; i++) qkv_res[i] = 0;

    for (int host_wi = 0; host_wi < 1; host_wi++) {
        // int host_wi = 0;
        // 拿一個head，做一次完整的qkv
        for (int i = 0; i<768; i++) {
            for (int j = 0; j<64; j++) {
                w_tiling_q[i*64 + j] = weight[host_wi*192 + i*2304 + j];
            }
        }
        for (int i = 0; i<768; i++) {
            for (int j = 0; j<64; j++) {
                w_tiling_k[i*64 + j] = weight[host_wi*192 + i*2304 + 64 + j];
            }
        }
        for (int i = 0; i<768; i++) {
            for (int j = 0; j<64; j++) {
                w_tiling_v[i*64 + j] = weight[host_wi*192 + i*2304 + 128 + j];
            }
        }
        // cout << w_tiling_q[0] << " " << w_tiling_q[1] << endl;
        // cout << w_tiling_k[0] << " " << w_tiling_k[1] << endl;
        // cout << w_tiling_v[0] << " " << w_tiling_v[1] << endl;

        // bias
        for (int i = 0; i<64; i++) {
            bias_tiling_q[i] = bias[host_wi*192 + i];
            bias_tiling_k[i] = bias[host_wi*192 + 64 + i];
            bias_tiling_v[i] = bias[host_wi*192 + 128 + i];
        }
        // cout << bias_tiling_q[0] << " " << bias_tiling_q[1] << endl;
        // cout << bias_tiling_k[0] << " " << bias_tiling_k[1] << endl;
        // cout << bias_tiling_v[0] << " " << bias_tiling_v[1] << endl;

        // q_attn, k_attn, c_attn
        for (int i = 0; i<127; i++) {
            for (int j = 0; j<64; j++) {
                q_attn[i*64 + j] = bias_tiling_q[j];
                k_attn[i*64 + j] = bias_tiling_k[j];
                v_attn[i*64 + j] = bias_tiling_v[j];
            }
        }
        for (int i = 0; i<127; i++) {
            for (int k = 0; k<768; k++) {
                for (int j = 0; j<64; j++) {
                    int idx_attn = i*64 + j;
                    q_attn[idx_attn] += x[i*768+ k] * w_tiling_q[k*64 + j];
                    k_attn[idx_attn] += x[i*768+ k] * w_tiling_k[k*64 + j];
                    v_attn[idx_attn] += x[i*768+ k] * w_tiling_v[k*64 + j];
                }
            }
        }
        // for (int i = 0; i<127*64; i++) cout << k_attn[i] << endl;
        // qk matmul
        for (int i = 0; i<16384; i++) qk_result[i] = 0;
        for (int i = 0; i<127; i++) {
            for (int k = 0; k<64; k++) {
                for (int j = 0; j<127; j++) {
                    qk_result[i*127 + j] += q_attn[i*64 + k] * k_attn[k + j*64];
                }
            }
        }
        for (int i = 0; i<127*127; i++) {
            qk_result[i] *= 0.125;
        }
        for (int i = 0; i<127; i++) {
            for (int j = i+1; j<127; j++) {
                qk_result[i*127 + j] = -INFINITY;
            }
        }
        // softmax
        for (int j = 0; j < 127; j++) {
            float sum;
            float buffer[128];
            sum = 0;
            int js = j*127;
            for (int i = 0; i < 127; i++) {
                buffer[i] = qk_result[js + i];
                sum = sum + exp(buffer[i]);
            }
            for (int i = 0; i < 127; i++) {
                qk_result[js + i] = exp(buffer[i]) / sum;
            }
        }
        // qkv // stack to (12, s, 64)

        for (int j = 0; j<127; j++) {
            for (int m = 0; m<127; m++) {
                for (int k = 0; k<64; k++) {
                    qkv_res[host_wi*127*64 + j*64 + k] += qk_result[j*127 + m] * v_attn[m*64 + k];
                }
            }
        }            
    }
    // c_proj
    for (int i = 0; i<127; i++) {
        for (int j = 0; j<768; j++) {
            out[i*768 + j] = proj_bias[j];
        }
    }
    for (int k = 0; k<12; k++) {
        for (int i = 0; i<127; i++) {
            for (int m = 0; m<64; m++) {
                int qkv_idx = k*127*64 + i*64 + m;
                for (int j = 0; j<768; j++) {
                    int weight_idx = (k*64+m)*768 + j;
                    out[i*768 + j] += qkv_res[qkv_idx] * proj_weight[weight_idx];
                }
            }
        }
    }
    for (int i = 0; i<127*127; i++) {
        // if(abs(qkv_res[i] - golden[i]) > 0.006)
            cout << qk_result[i] << " " << golden[i] << endl;
    }        
    
    // float cnt = 0;
    // for (int i = 0; i<97536; i++) {
    //     cnt += abs(out[i] - golden[i]);
    // }
    // cnt /= 97536;
    // cout << cnt << endl;


    return 0;
}