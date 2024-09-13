#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
float query[768+1];
// float q_tmp[97536+1];
float golden[768+1];
float key[97536+768+1];
float cur_key[768];
float past_key[97536+768+1]; // 做HLS要開最大
// float k_tmp[97536+1];

float value[97536+768+1];
float cur_value[768];
float past_value[97536+1]; // 做HLS要開最大
// float attn_result[292608+1];
float qk_result[1536+1]; // 12 * 128 // 12*(s+1)
float qkv_res[768+1];

float x[768+1];
float q_attn[97536+1];
float k_attn[97536+1];
float v_attn[97536+1];
float out[768+1];
float weight[1769472+1]; // 768*2304
float bias[2304+1];
float proj_weight[589824+1];
float proj_bias[768+1];
float c_attn[294912+1]; 

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
    vector<string> file_name = {"query_iter_2_block_1.txt", "past_key_iter_2_block_1.txt", "past_value_iter_2_block_1.txt", "key_iter_2_block_1.txt", "value_iter_2_block_1.txt", "ln1_out_iter_2_block_1.txt", "transformer.h.0.attn.c_attn.weight.txt", "transformer.h.0.attn.c_attn.bias.txt", "transformer.h.0.attn.c_proj.weight.txt", "transformer.h.0.attn.c_proj.bias.txt", "attn_cproj_output_iter_2_block_1.txt"};

    
    for(int i = 0; i<file_name.size(); i++) {
        ifstream infile(file_name[i]);
        if(!infile) {
            cerr << "unable to open file" << endl;
            return 1;
        }

        string line;
        
        while (getline(infile, line)) {
            if(i == 0) {
                query[cnt_q] = stof(line); 
                cnt_q++;
            }
            else if(i == 1) {
                past_key[cnt_k] = stof(line);
                cnt_k++;
            }
            else if(i == 2){
                past_value[cnt_v] = stof(line);
                cnt_v++;
            }
            else if(i == 3) {
                cur_key[cnt_cur_k] = stof(line);
                cnt_cur_k++;
            }
            else if(i == 4) {
                cur_value[cnt_cur_v] = stof(line);
                cnt_cur_v++;
            } 
            else if(i == 5) {
                x[cnt_x] = stof(line);
                cnt_x++;
            }
            else if(i == 6) {
                weight[cnt_weight] = stof(line);
                cnt_weight++;
            }
            else if(i == 7) {
                bias[cnt_bias] = stof(line);
                cnt_bias++;
            }
            else if(i == 8) {
                proj_weight[cnt_proj_w] = stof(line);
                cnt_proj_w++; 
            }
            else if(i == 9) {
                proj_bias[cnt_proj_b] = stof(line);
                cnt_proj_b++;
            }
            else {
                golden[cnt_golden] = stof(line);
                cnt_golden++;
            }
        }
        infile.close();
    }

    for (int j = 0; j<2304; j++) {
        c_attn[j] = bias[j];
    }
    for (int k = 0; k<768; k++) {
        for (int j = 0; j<2304; j++) {
            c_attn[j] += x[k] * weight[k*2304 + j];
        }
    }
    // for (int i = 0; i<2304; i++) cout << c_attn[i] << endl;
    int idx_q = 0;
    int idx_k = 0;
    int idx_v = 0;
    for(int j = 0; j<3; j++) {
            if(j == 0) {
                for (int k = 0; k<768; k++) {
                    q_attn[idx_q++] = c_attn[j*768 + k];
                }
            }
            else if(j == 1) {
                for (int k = 0; k<768; k++) {
                    k_attn[idx_k++] = c_attn[j*768 + k];
                }
            }
            else if(j == 2) {
                for (int k = 0; k<768; k++) {
                    v_attn[idx_v++] = c_attn[j*768 + k];
                }
            }
    }
    // float cnt = 0;
    // for (int i = 0; i<768; i++) {
    //     cnt += abs(q_attn[i] - query[i]);
    //     cnt += abs(v_attn[i] - cur_value[i]);
    //     cnt += abs(k_attn[i] - cur_key[i]);

    //     if(abs(q_attn[i] - query[i]) > 0.0001)
    //         cout << q_attn[i] << " " << query[i] << endl;
    // }
    // cnt /= 768;
    // cout << cnt << endl;
    // stack past k
    // 這個寫法對cur_key來說會多做，但整體來說存取比較連續
    int s = 127;
    int cur_s = 128;
    for (int i = 0; i<12; i++) {
        for (int k = 0; k<64; k++) {
            for (int j = 0; j<s; j++) {
                key[i*(s+1)*64 + j*64 + k] = past_key[i*s*64 + j*64 + k];
                key[i*(s+1)*64 + s*64 + k] = k_attn[i*64 + k];
            }
        }
    }

    // compute qk matmul, using no tmp array
    for (int i = 0; i<1536; i++) qk_result[i] = 0;
    for (int i = 0; i < 12; i++) {
        for (int j = 0; j<1; j++) {
            for (int m = 0; m<64; m++) {
                int idx_q = i*64 + j*768 + m; // 1, 12, 64 -> 12, 1, 64
                for (int k = 0; k<cur_s; k++) {
                    int idx_k = i*cur_s*64 + m + k*64; // 12, 128, 64 -> 12, 64, 128
                    qk_result[i*cur_s*1 + j*cur_s + k] += q_attn[idx_q] * key[idx_k]; // 12, 1, cur_s
                }
            }            
        }
    }
    for (int i = 0; i<1536; i++) {
        qk_result[i] *= 0.125;
    }


    // compute softmax
    for (int m = 0; m < 12; m++) {
        for (int j = 0; j < 1; j++) {
            float sum;
            float buffer[128];
            sum = 0;
            int js = j*128;
            for (int i = 0; i < 128; i++) {
                buffer[i] = qk_result[m * 1 * 128 + js + i];
                sum = sum + exp(buffer[i]);
            }
            for (int i = 0; i < 128; i++) {
                qk_result[m * 1 * 128 + js + i] = exp(buffer[i]) / sum;
            }
        }
    }

    // stack past v
    for (int i = 0; i<12; i++) {
        for (int j = 0; j<s; j++) {
            for (int k = 0; k<64; k++) {
                value[i*(s+1)*64 + j*64 + k] = past_value[i*s*64 + j*64 + k];
                value[i*(s+1)*64 + s*64 + k] = v_attn[i*64 + k];
            }
        }
    }

    // compute qkv
    for (int i = 0; i<768; i++) qkv_res[i] = 0;
    for (int i = 0; i < 12; i++) {
        for (int k = 0; k<cur_s; k++) {
            int idx_qk = i*cur_s + k; // 12, 1, 128
            for (int m = 0; m<64; m++) {
                int idx_v = i*64*cur_s + k*64 + m; // 12, 128, 64
                qkv_res[i*64 + m] += qk_result[idx_qk] * value[idx_v];
            }
        }
    }
    // c_proj
    for (int i = 0; i<1; i++) {
        for (int j = 0; j<768; j++) {
            out[i*768 + j] = proj_bias[j];
        }
    }
    for (int k = 0; k<12; k++) {
        for (int i = 0; i<1; i++) {
                for (int m = 0; m<64; m++) {
                    int qkv_idx = k*1*64 + i*64 + m;
                    for (int j = 0; j<768; j++) {
                        int weight_idx = (k*64+m)*768 + j;
                        out[i*768 + j] += qkv_res[qkv_idx] * proj_weight[weight_idx];
                    }
                }
            }
    }
    float cnt = 0;

    for (int i = 0; i<768; i++) {
        cnt += abs(out[i] - golden[i]);
        if(abs(out[i] - golden[i]) > 0.0001)
            cout << out[i] << " " << golden[i] << endl;
    }
    cnt /= 768;
    cout << cnt << endl;
    
    return 0;
}