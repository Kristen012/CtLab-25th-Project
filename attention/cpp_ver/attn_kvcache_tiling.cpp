#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
float golden[98304+1];
float weight[1769472+1]; // 768*2304
float bias[2304+1];
float past_key[97536+768+1];
float past_value[97536+1];
// hls use
float x[768+1]; // 128*768
float w_tiling_q[49152+1];
float w_tiling_k[49152+1];
float w_tiling_v[49152+1];
float bias_tiling_q[64+1];
float bias_tiling_k[64+1];
float bias_tiling_v[64+1];

float q_attn[64+1];
float cur_k_attn[64+1];
float cur_v_attn[64+1];

float k_tiling[65536+1]; // past + cur
float v_tiling[65536+1]; // past + cur

float qk_result[128+1];
float qkv_res[98304+1]; // 可以reuse x
float proj_weight[589824+1];
float proj_bias[768+1];
float out[768+1];

int main() {
    // read q, k into a single 1d array
    int cnt_q = 0;
    int cnt_k = 0;
    int cnt_v = 0;
    int cnt_cur_k = 0;
    int cnt_cur_v = 0;
    int cnt_golden = 0;
    int cnt_weight = 0;
    int cnt_bias = 0;
    int cnt_x = 0;
    int cnt_proj_w = 0;
    int cnt_proj_b = 0;
    vector<string> file_name = {"past_key_iter_2_block_1.txt", "past_value_iter_2_block_1.txt", "ln1_out_iter_2_block_1.txt", "transformer.h.0.attn.c_attn.weight_split_head.txt", "transformer.h.0.attn.c_attn.bias_split_head.txt", "transformer.h.0.attn.c_proj.weight.txt", "transformer.h.0.attn.c_proj.bias.txt", "attn_cproj_input_iter_2_block_1.txt"};

    
    for(int i = 0; i<file_name.size(); i++) {
        ifstream infile(file_name[i]);
        if(!infile) {
            cerr << "unable to open file" << endl;
            return 1;
        }

        string line;
        
        while (getline(infile, line)) {
            if(i == 0) {
                past_key[cnt_k] = stof(line);
                cnt_k++;
            }
            else if(i == 1){
                past_value[cnt_v] = stof(line);
                cnt_v++;
            }
            else if(i == 2) {
                x[cnt_x] = stof(line);
                cnt_x++;
            }
            else if(i == 3) {
                weight[cnt_weight] = stof(line);
                cnt_weight++;
            }
            else if(i == 4) {
                bias[cnt_bias] = stof(line);
                cnt_bias++;
            }
            else if(i == 5) {
                proj_weight[cnt_proj_w] = stof(line);
                cnt_proj_w++; 
            }
            else if(i == 6) {
                proj_bias[cnt_proj_b] = stof(line);
                cnt_proj_b++;
            }
            else {
                if(cnt_golden < 768) {
                    golden[cnt_golden] = stof(line);
                    cnt_golden++;
                }
            }
        }
        infile.close();
    }
    for (int i = 0; i<98304; i++) qkv_res[i] = 0;
    for (int host_wi = 0; host_wi < 1; host_wi++) {
        // int host_wi = 0;
        // 拿一個head，做一次完整的qkv
        // weight
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
        // bias
        for (int i = 0; i<64; i++) {
            bias_tiling_q[i] = bias[host_wi*192 + i];
            bias_tiling_k[i] = bias[host_wi*192 + 64 + i];
            bias_tiling_v[i] = bias[host_wi*192 + 128 + i];
        }
        // read past k, v
        for (int i = 0; i<127*64; i++) {
            k_tiling[i] = past_key[host_wi*127*64 + i];
            v_tiling[i] = past_value[host_wi*127*64 + i];
        }
        // krnl
        // q_attn, cur_k_attn, cur_c_attn
        for (int i = 0; i<1; i++) {
            for (int j = 0; j<64; j++) {
                q_attn[i*64 + j] = bias_tiling_q[j];
                cur_k_attn[i*64 + j] = bias_tiling_k[j];
                cur_v_attn[i*64 + j] = bias_tiling_v[j];
            }
        }
        for (int i = 0; i<1; i++) {
            for (int k = 0; k<768; k++) {
                for (int j = 0; j<64; j++) {
                    int idx_attn = i*64 + j;
                    q_attn[idx_attn] += x[i*768+ k] * w_tiling_q[k*64 + j];
                    cur_k_attn[idx_attn] += x[i*768+ k] * w_tiling_k[k*64 + j];
                    cur_v_attn[idx_attn] += x[i*768+ k] * w_tiling_v[k*64 + j];
                }
            }
        }
        // stack past k, v
        for (int i = 0; i<1; i++) { // new s
            for (int j = 0; j<64; j++) {
                k_tiling[127*64 + i*64 + j] = cur_k_attn[i*64 + j];
                v_tiling[127*64 + i*64 + j] = cur_v_attn[i*64 + j];
            }
        }
        // qk matmul
        for (int i = 0; i<128; i++) qk_result[i] = 0;
        for (int i = 0; i<1; i++) {
            for (int k = 0; k<64; k++) {
                for (int j = 0; j<128; j++) {
                    qk_result[i*128 + j] += q_attn[i*64 + k] * k_tiling[k + j*64];
                }
            }
        }
        for (int i = 0; i<128; i++) qk_result[i] *= 0.125;
        // softmax
        for (int j = 0; j < 1; j++) {
            float sum;
            float buffer[128+1];
            sum = 0;
            int js = j*128;
            for (int i = 0; i < 128; i++) {
                buffer[i] = qk_result[js + i];
                sum = sum + exp(buffer[i]);
            }
            for (int i = 0; i < 128; i++) {
                qk_result[js + i] = exp(buffer[i]) / sum;
            }
        }
        // qkv
        for (int i = 0; i<1; i++) {
            for (int k = 0; k<128; k++) {
                for (int j = 0; j<64; j++) {
                    qkv_res[host_wi*64 + i*64 + j] += qk_result[i*128 + k] * v_tiling[k*64 + j];
                }
            }
        }
        
    }
    // c_proj
    // for (int i = 0; i<1; i++) {
    //     for (int j = 0; j<768; j++) {
    //         out[i*768 + j] = proj_bias[j];
    //     }
    // }
    // for (int k = 0; k<12; k++) {
    //     for (int i = 0; i<1; i++) {
    //             for (int m = 0; m<64; m++) {
    //                 int qkv_idx = k*1*64 + i*64 + m;
    //                 for (int j = 0; j<768; j++) {
    //                     int weight_idx = (k*64+m)*768 + j;
    //                     out[i*768 + j] += qkv_res[qkv_idx] * proj_weight[weight_idx];
    //                 }
    //             }
    //         }
    // }
    // float cnt = 0;

    for (int i = 0; i<64; i++) {
        // cnt += abs(out[i] - golden[i]);
        // if(abs(qkv_res[i] - golden[i]) > 0.0006)
            cout << qk_result[i] << " " << golden[i] << endl;
    }
    // cnt /= 768;
    // cout << cnt << endl;
    
    return 0;
}