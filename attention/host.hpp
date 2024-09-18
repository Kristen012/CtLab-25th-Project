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


#include <iostream>
#include <CL/cl2.hpp>
using namespace std;

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

int s = 127; // TODO

float k_cache[12][768432]; // 12, 1024, 64 // 12個block各自attn的kv cache
float v_cache[12][768432]; // 12, 1024, 64
float k_tmp[786432];
float v_tmp[786432];

int idx_cache = 0;

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
float tmp[1769472 + 1];
void dataPrepare(float *Array, int Nb_Of_Elements, int flag, int host_wi, int iter) {
    ifstream infile;
	if (flag == 0) {
		if(iter == 0) infile.open("/home/ywtang23/attn_data/ln_out_iter_1_block_1.txt");
		else if(iter == 1) infile.open("/home/ywtang23/attn_data/ln_out_iter_2_block_1.txt");
		else infile.open("/home/ywtang23/attn_data/ln_out_iter_3_block_1.txt");
	} else if (flag == 1 || flag == 2 || flag == 3) {
        infile.open("/home/ywtang23/attn_data/transformer.h.0.attn.c_attn.weight_split_head.txt");
    } else if (flag == 4 || flag == 5 || flag == 6) {
        infile.open("/home/ywtang23/attn_data/transformer.h.0.attn.c_attn.bias_split_head.txt");
	} else if (flag == 7) {
//        infile.open("/home/ywtang23/attn_data/past_key_iter_2_block_1.txt");
		for (int i = 0; i<Nb_Of_Elements; i++) Array[i] = 0;
	} else if (flag == 8) {
//        infile.open("/home/ywtang23/attn_data/past_value_iter_2_block_1.txt");
		for (int i = 0; i<Nb_Of_Elements; i++) Array[i] = 0;
	} else if (flag == 9) {
        infile.open("/home/ywtang23/attn_data/transformer.h.0.attn.c_proj.weight.txt");
	} else if (flag == 10) {
		infile.open("/home/ywtang23/attn_data/transformer.h.0.attn.c_proj.bias.txt");
	}
	else if (flag == 11) {
		infile.open("/home/ywtang23/attn_data/qkv_res_iter_1_block_1.txt");
	}
	else if (flag == 12) {
		if(iter == 0) infile.open("/home/ywtang23/attn_data/cproj_out_iter_1_block_1.txt");
		else if(iter == 1) infile.open("/home/ywtang23/attn_data/cproj_out_iter_2_block_1.txt");
		else if(iter == 2)infile.open("/home/ywtang23/attn_data/cproj_out_iter_3_block_1.txt");
	} else if (flag == 13) {
		if(iter == 1) infile.open("/home/ywtang23/attn_data/key_after_kvcache_iter_2_block_1.txt");
		else infile.open("/home/ywtang23/attn_data/key_after_kvcache_iter_3_block_1.txt");
	} else if (flag == 14) {
		if(iter == 1) infile.open("/home/ywtang23/attn_data/value_after_kvcache_iter_2_block_1.txt");
		else infile.open("/home/ywtang23/attn_data/value_after_kvcache_iter_3_block_1.txt");
	}
	else {
        // cerr << "Invalid flag" << endl;
        // return;
    }

    if (!infile && (flag != 7 || flag != 8)) {
        cerr << "Error opening file" << flag << endl;
        return;
    }

    std::string line;

    // Read the data into the array
	if(host_wi == -1) {
    	int element_count = 0;
		while (std::getline(infile, line) && element_count < Nb_Of_Elements) {
			Array[element_count] = std::stof(line); // Convert the entire line to float
			element_count++;
		}
	}
	else {
		int cnt = 0;
		while (std::getline(infile, line)) {
			tmp[cnt++] = std::stof(line);
		}
		if(flag == 1) { // q
			for (int i = 0; i<768; i++) {
				for (int j = 0; j<64; j++) {
					Array[i*64 + j] = tmp[host_wi*192 + i*2304 + j];
				}
        	}
		}
		else if (flag == 2) { // k
			for (int i = 0; i<768; i++) {
				for (int j = 0; j<64; j++) {
					Array[i*64 + j] = tmp[host_wi*192 + i*2304 + 64 + j];
				}
			}
		}
		else if (flag == 3) { // v
			for (int i = 0; i<768; i++) {
				for (int j = 0; j<64; j++) {
					Array[i*64 + j] = tmp[host_wi*192 + i*2304 + 128 + j];
				}
			}
		}
		else if (flag == 4) {
			for (int i = 0; i<64; i++) {
            	Array[i] = tmp[host_wi*192 + i];
			}
		}
		else if (flag == 5) {
			for (int i = 0; i<64; i++) {
            	Array[i] = tmp[host_wi*192 + 64 + i];
			}
		}
		else if (flag == 6) {
			for (int i = 0; i<64; i++) {
            	Array[i] = tmp[host_wi*192 + 128 + i];
			}
		}
		else if (flag == 7) {
			if(iter == 0) {
				for (int i = 0; i<s*64; i++) {
					Array[i] = 0;
				}
			}
			else {
				for (int i = 0; i<s*64; i++) {
					Array[i] = k_cache[0][host_wi*(s+1)*64 + i];
				}
			}
		}
		else if (flag == 8) {
			if(iter == 0) {
				for (int i = 0; i<s*64; i++) {
					Array[i] = 0;
				}
			}
			else {
				for (int i = 0; i<s*64; i++) {
					Array[i] = v_cache[0][host_wi*(s+1)*64 + i];
				}
			}
		}
	}



    infile.close();

//    if (element_count < Nb_Of_Elements) {
//        std::cerr << "Warning: Only " << element_count << " elements were read, but " << Nb_Of_Elements << " were expected." << std::endl;
//    }
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
