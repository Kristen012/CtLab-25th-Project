/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.h
Purpose: VITIS vector addition
Revision History: January 28, 2016

*******************************************************************************
Copyright (C) 2019 XILINX, Inc.

This file contains confidential and proprietary information of Xilinx, Inc. and
is protected under U.S. and international copyright and other intellectual
property laws.

DISCLAIMER
This disclaimer is not a license and does not grant any rights to the materials
distributed herewith. Except as otherwise provided in a valid license issued to
you by Xilinx, and to the maximum extent permitted by applicable law:
(1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX
HEREBY DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, OR
FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
in contract or tort, including negligence, or under any other theory of
liability) for any loss or damage of any kind or nature related to, arising under
or in connection with these materials, including for any direct, or any indirect,
special, incidental, or consequential loss or damage (including loss of data,
profits, goodwill, or any type of loss or damage suffered as a result of any
action brought by a third party) even if such damage or loss was reasonably
foreseeable or Xilinx had been advised of the possibility of the same.

CRITICAL APPLICATIONS
Xilinx products are not designed or intended to be fail-safe, or for use in any
application requiring fail-safe performance, such as life-support or safety
devices or systems, Class III medical devices, nuclear facilities, applications
related to the deployment of airbags, or any other applications that could lead
to death, personal injury, or severe property or environmental damage
(individually and collectively, "Critical Applications"). Customer assumes the
sole risk and liability of any use of Xilinx products in Critical Applications,
subject only to applicable laws and regulations governing limitations on product
liability.

THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT
ALL TIMES.

*******************************************************************************/

#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#define K0 768
#define N0 426
#define K1 1
#define N1 118

#include <iostream>
#include <cstring>
#include <vector>
#include <CL/cl2.hpp>

using namespace std;
static const int LN_DATA_DEPTH = 128;
static const int LN_DATA_WIDTH = 768;
static const int LN_DATA_SIZE = LN_DATA_DEPTH * LN_DATA_WIDTH;

static const int DATA_SIZE_C_ATTN_WEIGHT = 1769472;
static const int DATA_SIZE_C_ATTN_WEIGHT_TILED = 49152;
static const int DATA_SIZE_C_ATTN_BIAS = 2304;
static const int DATA_SIZE_C_ATTN_BIAS_TILED = 64;
static const int DATA_SIZE_C_PROJ_WEIGHT = 589824;
static const int DATA_SIZE_C_PROJ_BIAS = 768;
static const int DATA_SIZE_ATTN_OUT = 768*128; // TODO
static const int DATA_SIZE_ATTN_CACHE_TILED = 65536; // TODO
static const int DATA_SIZE_ATTN_CUR_K_V = 8192;

static const int DATA_SIZE_VADD_OUT = 128*768; // TODO

static const int DATA_SIZE_MLP_WEIGHT = 768*3072;
static const int DATA_SIZE_MLP_BIAS_1 = 3072;
static const int DATA_SIZE_MLP_BIAS_2 = 768;
static const int DATA_SIZE_MLP_OUT = 128*768;

static const int DATA_SIZE_LAST_LINEAR_WEIGHT_WIDTH = 426*118;
static const int DATA_SIZE_LAST_LINEAR_ALL_WEIGHT = 768*426*118;
static const int DATA_SIZE_LAST_LINEAR_OUT = 128*426*118;
static const int DATA_SIZE_LL_WEIGHT = 768*(2*3*71);
static const int DATA_SIZE_LL_RES = 128*(2*3*71);
static const int DATA_SIZE_LL_MAX_DEPTH = 32;

static const int num_block = 12;


// Compute the size of array in bytes
size_t size_in_bytes_ln_inout = LN_DATA_SIZE * sizeof(int);
size_t size_in_bytes_ln_GandB = LN_DATA_WIDTH * sizeof(int);

size_t size_in_bytes_c_attn_weight_tiled = DATA_SIZE_C_ATTN_WEIGHT_TILED * sizeof(float);
size_t size_in_bytes_c_attn_bias_tiled = DATA_SIZE_C_ATTN_BIAS_TILED * sizeof(float);

size_t size_in_bytes_c_proj_weight = DATA_SIZE_C_PROJ_WEIGHT * sizeof(float);
size_t size_in_bytes_c_proj_bias = DATA_SIZE_C_PROJ_BIAS * sizeof(float);

size_t size_in_bytes_attn_cache_tiled = DATA_SIZE_ATTN_CACHE_TILED * sizeof(float);

size_t size_in_bytes_attn_out = DATA_SIZE_ATTN_OUT * sizeof(float);
size_t size_in_bytes_attn_cur_k_v = DATA_SIZE_ATTN_CUR_K_V * sizeof(float);

size_t size_in_bytes_vadd_out = DATA_SIZE_VADD_OUT * sizeof(float);

size_t size_in_bytes_mlp_weight = DATA_SIZE_MLP_WEIGHT * sizeof(float);
size_t size_in_bytes_mlp_bias_1 = DATA_SIZE_MLP_BIAS_1 * sizeof(float);
size_t size_in_bytes_mlp_bias_2 = DATA_SIZE_MLP_BIAS_2 * sizeof(float);
size_t size_in_bytes_mlp_out = DATA_SIZE_MLP_OUT * sizeof(float);

size_t size_in_bytes_ll_weight = DATA_SIZE_LL_WEIGHT * sizeof(float);
size_t size_in_bytes_ll_res = DATA_SIZE_LL_RES * sizeof(float);

enum DATA {
	ln_Data_in1,
	ln_Data_g, ln_Data_b,
	attn_Data_w_tiling_q, attn_Data_w_tiling_k, attn_Data_w_tiling_v,
	attn_Data_bias_tiling_q, attn_Data_bias_tiling_k, attn_Data_bias_tiling_v,
	attn_Data_k_tiling, attn_Data_v_tiling, attn_Data_c_proj_weight, attn_Data_c_proj_bias,
	ln2_Data_g, ln2_Data_b,
	mlp_Data_w_1, mlp_Data_w_2, mlp_Data_b_1, mlp_Data_b_2,
	lnf_Data_b, lnf_Data_g,
	ll_in, ll_w
};

cl::Buffer buffer_Data_ln_Data_in1;
cl::Buffer buffer_Data_ln_Data_g, buffer_Data_ln_Data_b, buffer_Data_ln_out;
cl::Buffer buffer_Data_attn_w_tiling_q, buffer_Data_attn_w_tiling_k, buffer_Data_attn_w_tiling_v,
	buffer_Data_attn_bias_tiling_q, buffer_Data_attn_bias_tiling_k, buffer_Data_attn_bias_tiling_v,
	buffer_Data_attn_k_tiling, buffer_Data_attn_v_tiling,
	buffer_attn_qkv_result, buffer_attn_cur_k, buffer_attn_cur_v;
cl::Buffer buffer_Data_attn_c_proj_weight, buffer_Data_attn_c_proj_bias;
cl::Buffer buffer_attn_out, buffer_vadd_out, buffer_mlp_out, buffer_layer_out;
cl::Buffer buffer_mlp_w1, buffer_mlp_b1, buffer_mlp_w2, buffer_mlp_b2;
cl::Buffer buffer_ll_head1_in, buffer_ll_head2_in, buffer_ll_head1_weight, buffer_ll_head1_res, buffer_ll_head2_weight, buffer_ll_head2_res;

float *ptr_ln_Data_in1;
float *ptr_ln_Data_g, *ptr_ln_Data_b, *ptr_ln_out;

float *ptr_attn_Data_w_tiling_q, *ptr_attn_Data_w_tiling_k, *ptr_attn_Data_w_tiling_v,
	*ptr_attn_Data_bias_tiling_q, *ptr_attn_Data_bias_tiling_k, *ptr_attn_Data_bias_tiling_v,
	*ptr_attn_Data_k_tiling, *ptr_attn_Data_v_tiling,
	*ptr_attn_Data_c_proj_weight, *ptr_attn_Data_c_proj_bias, *ptr_attn_qkv_result,
	*ptr_attn_cur_k, *ptr_attn_cur_v, *ptr_attn_out;

float *ptr_vadd_out;

float *ptr_mlp_w1, *ptr_mlp_b1, *ptr_mlp_w2, *ptr_mlp_b2, *ptr_mlp_out;

float *ptr_layer_out;

float *ptr_ll_Data_head1_in, *ptr_ll_Data_head1_weight, *ptr_ll_head1_res;

float *ptr_ll_Data_head2_in, *ptr_ll_Data_head2_weight, *ptr_ll_head2_res;

int s; // TODO

int kv_inner = 768432;

vector<vector<float>> k_cache(num_block, vector<float>(kv_inner));
vector<vector<float>> v_cache(num_block, vector<float>(kv_inner));
vector<float> k_tmp(kv_inner);
vector<float> v_tmp(kv_inner);

// float LL_RES[128 * DATA_SIZE_LAST_LINEAR_WEIGHT_WIDTH]; // 128 * N0 * N1

int idx_cache = 0;
vector<int> Array_tokenization_out(128);
vector<float> Array_ln_Data_in1(LN_DATA_SIZE);
vector<vector<float>> Array_ln_Data_g(num_block);
vector<vector<float>> Array_ln_Data_b(num_block);
vector<vector<float>> Array_attn_Data_w_tiling(num_block);
vector<vector<float>> Array_attn_Data_bias_tiling(num_block);
vector<vector<float>> Array_attn_Data_c_proj_weight(num_block);
vector<vector<float>> Array_attn_Data_c_proj_bias(num_block);
vector<vector<float>> Array_mlp_Data_w_1(num_block);
vector<vector<float>> Array_mlp_Data_w_2(num_block);
vector<vector<float>> Array_mlp_Data_b_1(num_block);
vector<vector<float>> Array_mlp_Data_b_2(num_block);
vector<vector<float>> Array_ln2_Data_g(num_block);
vector<vector<float>> Array_ln2_Data_b(num_block);
vector<float> Array_lnf_Data_g(LN_DATA_WIDTH);
vector<float> Array_lnf_Data_b(LN_DATA_WIDTH);
vector<float> Array_last_linear_w(DATA_SIZE_LAST_LINEAR_ALL_WEIGHT);
vector<float> LL_RES(128 * DATA_SIZE_LAST_LINEAR_WEIGHT_WIDTH);

vector<float> host_result(768*128);
vector<float> host_result_ll(50257*128);
vector<vector<float>> wte;
vector<vector<float>> wpe;

// float Array_ln_Data_in1;
// float **Array_ln_Data_g;
// float **Array_ln_Data_b;
// float **Array_attn_Data_w_tiling;
// float **Array_attn_Data_bias_tiling;
// float **Array_attn_Data_c_proj_weight;
// float **Array_attn_Data_c_proj_bias;
// float **Array_mlp_Data_w_1;
// float **Array_mlp_Data_w_2;
// float **Array_mlp_Data_b_1;
// float **Array_mlp_Data_b_2;
// float **Array_ln2_Data_g;
// float **Array_ln2_Data_b;
// float *Array_lnf_Data_g;
// float *Array_lnf_Data_b;
// float *Array_last_linear_w;

//Customized buffer allocation for 4K boundary alignment
template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

void upd_kv_cache(float* cur_k, float* cur_v, int block, int head, int iter) {
    // cur接在s*64的尾巴
    // 當前head之後的都放到tmp往後移
    if (iter != 0) {
        for (int i = 0; i<11-head; i++) {
            for (int j = 0; j<s; j++) {
                for (int k = 0; k<64; k++) {
                    k_tmp[i*s*64 + j*64 + k] = k_cache[block][head*(s+1)*64 + s*64 + i*s*64 + j*64 + k];
                    v_tmp[i*s*64 + j*64 + k] = v_cache[block][head*(s+1)*64 + s*64 + i*s*64 + j*64 + k];
                }
            }
        }
        for (int i = 0; i<64; i++) {
            k_cache[block][head*(s+1)*64 + s*64 + i] = cur_k[i];
            v_cache[block][head*(s+1)*64 + s*64 + i] = cur_v[i];
        }
        for (int i = 0; i<11-head; i++) {
            for (int j = 0; j<s; j++) {
                for (int k = 0; k<64; k++) {
                    k_cache[block][head*(s+1)*64 + (s+1)*64 + i*s*64 + j*64 + k] = k_tmp[i*s*64 + j*64 + k];
                    v_cache[block][head*(s+1)*64 + (s+1)*64 + i*s*64 + j*64 + k] = v_tmp[i*s*64 + j*64 + k];
                }
            }
        }
    }
    else {
        for (int i = 0; i<s*64; i++) {
            k_cache[block][head*s*64 + i] = cur_k[i];
            v_cache[block][head*s*64 + i] = cur_v[i];
        }
    }

}

std::vector<float> load1DWeights(const std::string& filePath) {
    std::vector<float> weights;
    std::ifstream inFile(filePath);

    if (inFile.is_open()) {
        float value;
        while (inFile >> value) { // Read each value into the vector
            weights.push_back(value);
        }
        inFile.close();
    } else {
        std::cerr << "Unable to open file " << filePath << std::endl;
    }

    return weights;
}

std::vector<std::vector<float>> loadWeightsWTEWPE(const std::string& filePath, int rows, int cols) {
    std::vector<float> flat_weights = load1DWeights(filePath);
    std::vector<std::vector<float>> reshaped_weights(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            reshaped_weights[i][j] = flat_weights[i * cols + j];
        }
    }

    return reshaped_weights;
}

void read_file(string path, vector<float>& Array) {
	ifstream infile;
	string full_path = "/home/ywtang23/Data/weight/";
	full_path = full_path + path;
	infile.open(full_path);
	if(!infile.is_open()) {
        cerr << "Error opening file." << endl;
        return;
    }
	std::string line;
	int element_count = 0;
    while (std::getline(infile, line)) {
        element_count++;
    }
	Array.resize(element_count);

    infile.clear();
    infile.seekg(0);
    element_count = 0;
    while (std::getline(infile, line)) {
        Array[element_count] = std::stof(line);
        element_count++;
    }
	infile.close();
}

void open_file() {

	ifstream infile;
	infile.open("/home/ywtang23/Data/activation/input_ids_before_embedding_iter_1.txt");
	if(!infile.is_open()) {
        cerr << "Error opening file input_ids_before_embedding_iter_1." << endl;
        return;
    }
	std::string line;
	int element_count = 0;
    while (std::getline(infile, line)) {
        element_count++;
    }
	Array_tokenization_out.resize(element_count);

    infile.clear();
    infile.seekg(0);
    element_count = 0;
    while (std::getline(infile, line)) {
        Array_tokenization_out[element_count] = std::stof(line);
        element_count++;
    }
	infile.close();

	for (int i = 0; i<num_block; i++) {
		read_file("transformer.h." + to_string(i) + ".ln_1.weight.txt", Array_ln_Data_g[i]);
		read_file("transformer.h." + to_string(i) + ".ln_1.bias.txt", Array_ln_Data_b[i]);
		read_file("transformer.h." + to_string(i) + ".attn.c_attn.weight_split_head.txt", Array_attn_Data_w_tiling[i]);
		read_file("transformer.h." + to_string(i) + ".attn.c_attn.bias_split_head.txt", Array_attn_Data_bias_tiling[i]);
		read_file("transformer.h." + to_string(i) + ".attn.c_proj.weight.txt", Array_attn_Data_c_proj_weight[i]);
		read_file("transformer.h." + to_string(i) + ".attn.c_proj.bias.txt", Array_attn_Data_c_proj_bias[i]);
		read_file("transformer.h." + to_string(i) + ".ln_2.weight.txt", Array_ln2_Data_g[i]);
		read_file("transformer.h." + to_string(i) + ".ln_2.bias.txt", Array_ln2_Data_b[i]);
		read_file("transformer.h." + to_string(i) + ".mlp.c_fc.weight.txt", Array_mlp_Data_w_1[i]);
		read_file("transformer.h." + to_string(i) + ".mlp.c_fc.bias.txt", Array_mlp_Data_b_1[i]);
		read_file("transformer.h." + to_string(i) + ".mlp.c_proj.weight.txt", Array_mlp_Data_w_2[i]);
		read_file("transformer.h." + to_string(i) + ".mlp.c_proj.bias.txt", Array_mlp_Data_b_2[i]);
	}
	read_file("transformer.ln_f.weight.txt", Array_lnf_Data_g);
	read_file("transformer.ln_f.bias.txt", Array_lnf_Data_b);
	read_file("linear_weight_new.txt", Array_last_linear_w);
    wte = loadWeightsWTEWPE("/home/ywtang23/Data/weight/transformer.wte.weight.txt", 50257, 768);
    wpe = loadWeightsWTEWPE("/home/ywtang23/Data/weight/transformer.wpe.weight.txt", 1024, 768);
}

void dataPrepare(float *ptr, vector<float>& Array,  int Nb_Of_Elements, enum DATA data, int host_wi, int s, int iter, int block) {

	switch (data) {
		case attn_Data_w_tiling_q: {
			for (int i = 0; i<768; i++) {
				for (int j = 0; j<64; j++) {
					ptr[i*64 + j] = Array[host_wi*192 + i*2304 + j];
				}
			}
			break;
		}
		case attn_Data_w_tiling_k: {
			for (int i = 0; i<768; i++) {
				for (int j = 0; j<64; j++) {
					ptr[i*64 + j] = Array[host_wi*192 + i*2304 + 64 + j];
				}
			}
			break;
		}
		case attn_Data_w_tiling_v: {
			for (int i = 0; i<768; i++) {
				for (int j = 0; j<64; j++) {
					ptr[i*64 + j] = Array[host_wi*192 + i*2304 + 128 + j];
				}
			}
			break;
		}
		case attn_Data_bias_tiling_q: {
			for (int i = 0; i<64; i++) {
				ptr[i] = Array[host_wi*192 + i];
			}
			break;
		}
		case attn_Data_bias_tiling_k: {
			for (int i = 0; i<64; i++) {
				ptr[i] = Array[host_wi*192 + 64 + i];
			}
			break;
		}
		case attn_Data_bias_tiling_v: {
			for (int i = 0; i<64; i++) {
				ptr[i] = Array[host_wi*192 + 128 + i];
			}
			break;
		}
		case attn_Data_k_tiling: {
			if(iter == 0) {
				for (int i = 0; i<s*64; i++) {
					ptr[i] = 0;
				}
			}
			else {
				for (int i = 0; i<s*64; i++) {
					ptr[i] = k_cache[block][host_wi*(s+1)*64 + i];
				}
			}
			break;
		}
		case attn_Data_v_tiling: {
			if(iter == 0) {
				for (int i = 0; i<s*64; i++) {
					ptr[i] = 0;
				}
			}
			else {
				for (int i = 0; i<s*64; i++) {
					ptr[i] = v_cache[block][host_wi*(s+1)*64 + i];
				}
			}
			break;
		}
		default: {
			std::memcpy(ptr, Array.data(), Nb_Of_Elements * sizeof(float));
			break;
		}
	}
}
void dataPrepare_tiling(float *Array_tiling, vector<float>& Array_origin, int Nb_Of_Elements, int flag, int cur_recur, int cur_depth) {
	if (flag == 1) {
		for (int i = 0; i < Nb_Of_Elements; i++) {
			Array_tiling[i] = Array_origin[cur_depth * K0 + i];
		}
	}
	else if (flag == 2) {
		for (int i = 0; i < K0; i++) {
			for (int j = 0; j < N0; j++) {
				Array_tiling[i * N0 + j] = Array_origin[i * DATA_SIZE_LAST_LINEAR_WEIGHT_WIDTH + cur_recur * N0 + j];
			}
		}
	}
}
void host_res_prepare(vector<float>& Array, int Nb_Of_Elements, int flag, int iter) {
    ifstream infile;
	string path;
	if(flag <= 12) path = "/home/ywtang23/Data/activation/layer_out_iter_" + to_string(iter+1) + "_block_" + to_string(flag) + ".txt";
	else if(flag == 13) path = "/home/ywtang23/Data/activation/lnf_out_iter_" + to_string(iter+1) + ".txt";
	else if(flag == 14) path = "/home/ywtang23/Data/activation/linear_output_iter_" + to_string(iter+1) + ".txt";
	else if(flag == 15) path = "/home/ywtang23/Data/activation/qkv_iter_" + to_string(iter+1) + "_block_1.txt";
	infile.open(path);

	std::string line;
	int element_count = 0;
	Array.resize(Nb_Of_Elements);
    infile.clear();
    infile.seekg(0);
    element_count = 0;
    while (std::getline(infile, line)) {
        Array[element_count] = std::stof(line);
        element_count++;
    }
	infile.close();

}
void run_custom_profiling (int Nb_Of_Kernels, int Nb_Of_Memory_Tranfers, cl_event* K_exe_event, cl_event* Mem_op_event,string* list_of_kernel_names) {
	typedef struct {
		string    action_type; // kernel, "memory (H->G)", "memory (G->H)"
		string    name;
		cl_event  event;
		cl_ulong  profiling_command_start;
		cl_ulong  profiling_command_end;
		double    duration;
	} profile_t;

	cl_int              errCode;

	// ---------------------------------
	// Profiling
	// ---------------------------------
	profile_t *PROFILE;

	PROFILE = new profile_t[Nb_Of_Kernels + Nb_Of_Memory_Tranfers];

	PROFILE[0].action_type="kernel";         PROFILE[0].name="krnl_vadd";  PROFILE[0].event = K_exe_event[0];

	for (int i=0; i<Nb_Of_Memory_Tranfers; i++) {
		PROFILE[Nb_Of_Kernels+i].action_type="mem (H<->G)";
		PROFILE[Nb_Of_Kernels+i].name="Transfer_" + to_string(i+1);
		PROFILE[Nb_Of_Kernels+i].event = Mem_op_event[i];
	}

	// -------------------------------------------------------------------------------------
	// Get events Start and End times and calculate duration for
	//   o) each Kernel and
	//   o) Memory (Global <-> Host) transfer.
	// Event Profile Info:
	//   o) CL_PROFILING_COMMAND_QUEUED
	//   o) CL_PROFILING_COMMAND_SUBMIT
	//   o) CL_PROFILING_COMMAND_START
	//   o) CL_PROFILING_COMMAND_END
	// -------------------------------------------------------------------------------------
	size_t nb_of_returned_bytes;

	for (int i=0; i<Nb_Of_Kernels + Nb_Of_Memory_Tranfers; i++) {
		errCode = clGetEventProfilingInfo(PROFILE[i].event, CL_PROFILING_COMMAND_START,
										  sizeof(cl_ulong), &PROFILE[i].profiling_command_start, &nb_of_returned_bytes);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get profiling info for " << PROFILE[i].name << " " << PROFILE[i].action_type << endl << endl;
			exit(0);
		}

		errCode = clGetEventProfilingInfo(PROFILE[i].event, CL_PROFILING_COMMAND_END,
										  sizeof(cl_ulong), &PROFILE[i].profiling_command_end, &nb_of_returned_bytes);
		if (errCode != CL_SUCCESS) {
			cout << endl << "HOST-Error: Failed to get profiling info for " << PROFILE[i].name << " " << PROFILE[i].action_type << endl << endl;
			exit(0);
		}

		PROFILE[i].duration = (double)(PROFILE[i].profiling_command_end - PROFILE[i].profiling_command_start) * 1.0e-6;
	}

	// -------------------------------------------------------------------------------------
	// Calculate Duration of
	//   o) All kernels execution time
	//   o) Application execution time (Kernels + Memory transfer)
	// -------------------------------------------------------------------------------------
	struct {
		int      Kernels_Start_Time_Index=0;
		int      Kernels_End_Time_Index=0;
		cl_ulong Kernels_Start_Time=0;
		cl_ulong Kernels_End_Time=0;

		int      Application_Start_Time_Index=0;
		int      Application_End_Time_Index=0;
		cl_ulong Application_Start_Time=0;
		cl_ulong Application_End_Time=0;
	} PROFILE_STAT;


	for (int i=0; i<Nb_Of_Kernels + Nb_Of_Memory_Tranfers; i++) {

		// Calculate Application statistics
		// .................................
		if ((PROFILE_STAT.Application_Start_Time == 0) || (PROFILE_STAT.Application_Start_Time > PROFILE[i].profiling_command_start)) {
			PROFILE_STAT.Application_Start_Time       = PROFILE[i].profiling_command_start;
			PROFILE_STAT.Application_Start_Time_Index = i;
		}

		if (PROFILE_STAT.Application_End_Time < PROFILE[i].profiling_command_end) {
			PROFILE_STAT.Application_End_Time       = PROFILE[i].profiling_command_end;
			PROFILE_STAT.Application_End_Time_Index = i;
		}

		// Calculate Kernel statistics
		// .................................
		if (PROFILE[i].action_type == "kernel") {
			if ((PROFILE_STAT.Kernels_Start_Time == 0) || (PROFILE_STAT.Kernels_Start_Time > PROFILE[i].profiling_command_start)) {
				PROFILE_STAT.Kernels_Start_Time       = PROFILE[i].profiling_command_start;
				PROFILE_STAT.Kernels_Start_Time_Index = i;
			}

			if (PROFILE_STAT.Kernels_End_Time < PROFILE[i].profiling_command_end) {
				PROFILE_STAT.Kernels_End_Time       = PROFILE[i].profiling_command_end;
				PROFILE_STAT.Kernels_End_Time_Index = i;
			}
		}
	}

	// ------------------------------
	// Print Profiling Data
	// ------------------------------
	int Column_Widths[5] = {15, 16, 15, 15, 15}, Separation_Line_Width = 0;

	// Print Table Header
	// ....................
	for (int i=0; i<5; i++)  Separation_Line_Width += Column_Widths[i];
	Separation_Line_Width += 3;
	cout << "HOST-Info: " << string(Separation_Line_Width, '-') << endl;

	// cout << "HOST-Info: "          << left << setw(Column_Widths[0]-1) << "Name"
	// 					  << " | " << left << setw(Column_Widths[1]-3) << "type"
	// 					  << " | " << left << setw(Column_Widths[2]-3) << "start"
	// 					  << " | " << left << setw(Column_Widths[2]-3) << "end"
	// 					  << " | " << left << setw(Column_Widths[2]-3) << "Duration(ms)"
	// 					  << endl;

	cout << "HOST-Info: " << string(Separation_Line_Width, '-') << endl;


	// Print Individual info for each Kernel and Memory transfer
	// ..........................................................
	// for (int i=0; i<Nb_Of_Kernels + Nb_Of_Memory_Tranfers; i++) {
	// 	cout << "HOST-Info: "          << left  << setw(Column_Widths[0]-1) << PROFILE[i].name
	// 						  << " | " << left  << setw(Column_Widths[1]-3) << PROFILE[i].action_type
	// 						  << " | " << right << setw(Column_Widths[2]-3) << PROFILE[i].profiling_command_start
	// 						  << " | " << right << setw(Column_Widths[2]-3) << PROFILE[i].profiling_command_end
	// 						  << " | " << right << setw(Column_Widths[2]-3) << PROFILE[i].duration
	// 						  << endl;
	// }
	cout << "HOST-Info: " << string(Separation_Line_Width, '-') << endl;

	// Print Duration of Kernels and Application execution
	// ..........................................................
	cout << "HOST-Info:     Kernels Execution Time (ms) :  "
			<< (double) (PROFILE_STAT.Kernels_End_Time - PROFILE_STAT.Kernels_Start_Time) * 0.000001 //1.0e-6
			<< "  (" << PROFILE[PROFILE_STAT.Kernels_End_Time_Index].name << "\'end - " << PROFILE[PROFILE_STAT.Kernels_Start_Time_Index].name << "\'begin)"
			<< endl;

	cout << "HOST-Info: Application Execution Time (ms) :  "
			<< (double) (PROFILE_STAT.Application_End_Time - PROFILE_STAT.Application_Start_Time) * 0.000001 //1.0e-6
			<< "  (" << PROFILE[PROFILE_STAT.Application_End_Time_Index].name << "\'end - " << PROFILE[PROFILE_STAT.Application_Start_Time_Index].name << "\'begin)"
			<< endl;

	cout << "HOST-Info: " << string(Separation_Line_Width, '-') << endl << endl;

}
