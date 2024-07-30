/*******************************************************************************
Vendor: Xilinx
Associated Filename: vadd.h
Purpose: VITIS vector addition

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

/*******************************************************************************
Description:

    This example uses the load/compute/store coding style which is generally
    the most efficient for implementing kernels using HLS. The load and store
    functions are responsible for moving data in and out of the kernel as
    efficiently as possible. The core functionality is decomposed across one
    of more compute functions. Whenever possible, the compute function should
    pass data through HLS streams and should contain a single set of nested loops.

    HLS stream objects are used to pass data between producer and consumer
    functions. Stream read and write operations have a blocking behavior which
    allows consumers and producers to synchronize with each other automatically.

    The dataflow pragma instructs the compiler to enable task-level pipelining.
    This is required for to load/compute/store functions to execute in a parallel
    and pipelined manner. Here the kernel loads, computes and stores NUM_WORDS integer values per
    clock cycle and is implemented as below:
                                       _____________
                                      |             |<----- Input Vector 1 from Global Memory
                                      |  load_input |       __
                                      |_____________|----->|  |
                                       _____________       |  | in1_stream
Input Vector 2 from Global Memory --->|             |      |__|
                               __     |  load_input |        |
                              |  |<---|_____________|        |
                   in2_stream |  |     _____________         |
                              |__|--->|             |<--------
                                      | compute_add |      __
                                      |_____________|---->|  |
                                       ______________     |  | out_stream
                                      |              |<---|__|
                                      | store_result |
                                      |______________|-----> Output result to Global Memory

*******************************************************************************/

// Includes
#include <stdint.h>
#include <hls_stream.h>
#include <hls_math.h>

// TRIPCOUNT identifier
const int DEPTH = 1;
const int HEIGHT = 8;
const int NX = 6;
const int NF = 2;
const int DEPTH_HEIGHT = 8;

const int DATA_SIZE_X = 48; // TODO
const int DATA_SIZE_WEIGHT = 12; // TODO
const int DATA_SIZE_BIAS = 2; // TODO
const int DATA_SIZE_RES = 16; // TODO
// Coding Style: function宣告要為static，遇到for迴圈前可以取error_type的名稱(ex: mem_rd)
static void load_input(float* in, hls::stream<float>& inStream, int size) {
mem_rd:
    for (int i = 0; i < size; i++) {
        inStream << in[i];
    }
}
static void reshape_x(hls::stream<float>& in, float out[DEPTH_HEIGHT][NX]) {
change_dim:
   for(int i = 0; i<DEPTH; i++) {
       for(int j = 0; j<HEIGHT; j++) {
           for(int k = 0; k<NX; k++) {
#pragma HLS PIPELINE
               out[i*HEIGHT+j][k] = in.read();
           }
       }
   }
}

static void reshape_weight(hls::stream<float>& in, float out[NX][NF]) {
change_dim:
   for(int i = 0; i<DEPTH; i++) {
       for(int j = 0; j<NX; j++) {
           for(int k = 0; k<NF; k++) {
#pragma HLS PIPELINE
// #pragma HLS LOOP_TRIPCOUNT min = NOT_SURE max = NOT_SURE
               out[i*NX+j][k] = in.read();
           }
       }
   }
}
static void reshape_bias(hls::stream<float>& in, float out[NF]) {
change_dim:
        for(int i = 0; i<NF; i++) {
#pragma HLS PIPELINE
            out[i] = in.read();
        }
}
static void compute_matmul(float a[DEPTH_HEIGHT][NX],
                       float b[NX][NF],
                       float conv_tmp[DEPTH_HEIGHT][NF]) {
execute_matmul:
   for(int i = 0; i<DEPTH_HEIGHT; i++) {
       for(int j = 0; j<NF; j++) {
    	   conv_tmp[i][j] = 0;
           for(int k = 0; k<NX; k++) {
#pragma HLS PIPELINE
        	   conv_tmp[i][j] = conv_tmp[i][j] + a[i][k] * b[k][j];
           }
       }
   }
}
static void compute_add_bias(float bias[NF],
				   float conv_tmp[DEPTH_HEIGHT][NF],
                   float conv_result[DEPTH_HEIGHT][NF],
                   int NF) {
execute_bias:
   for(int i = 0; i<DEPTH_HEIGHT; i++) {
       for(int j = 0; j<NF; j++) {
#pragma HLS PIPELINE
           conv_result[i][j] = conv_tmp[i][j] + bias[j];
       }
   }
}
static void store_result(float* out, float conv_result[DEPTH_HEIGHT][NF]) {
mem_wr:
   for(int i = 0; i<DEPTH_HEIGHT; i++) {
       for(int j = 0; j<NF; j++) {
#pragma HLS PIPELINE
           out[i*NF+j] = conv_result[i][j];
       }
   }
}
extern "C" {

void krnl_conv1D(float* x, float* weight, float* bias, float* out) {
#pragma HLS INTERFACE m_axi port = x bundle = gmem0
#pragma HLS INTERFACE m_axi port = weight bundle = gmem1
#pragma HLS INTERFACE m_axi port = bias bundle = gmem2
#pragma HLS INTERFACE m_axi port = out bundle = gmem3

   static hls::stream<float> x_stream("input_stream_x");
   static hls::stream<float> weight_stream("input_stream_weight");
   static hls::stream<float> bias_stream("input_stream_bias");

   float processed_x[DEPTH_HEIGHT][NX]; // DEPTH, HEIGHT, WIDTH // WIDTH=NX
   float processed_weight[NX][NF]; // 1, NX, NF
   float processed_bias[NF]; // NF
   float conv_result[DEPTH_HEIGHT][NF]; // x*weight+bias
   float conv_tmp[DEPTH_HEIGHT][NF]; // x*weight

#pragma HLS dataflow
    // dataflow pragma instruct compiler to run following three APIs in parallel

   load_input(x, x_stream, DATA_SIZE_X);
   reshape_x(x_stream, processed_x);

   load_input(weight, weight_stream, DATA_SIZE_WEIGHT);
   reshape_weight(weight_stream, processed_weight);

   load_input(bias, bias_stream, NF);
   reshape_bias(bias_stream, processed_bias);
   compute_matmul(processed_x, processed_weight, conv_tmp);
   compute_add_bias(processed_bias, conv_tmp, conv_result, NX);
   store_result(out, conv_result);
}
}
