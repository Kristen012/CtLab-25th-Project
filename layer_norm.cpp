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
#include <ap_int.h>
#include <hls_math.h>
#include <stdio.h>


#define DATA_SIZE 4096

// TRIPCOUNT identifier
const int c_size = DATA_SIZE;
const int HH = 512;
const int WW = 768;

// Coding Style: function宣告要為static，遇到for迴圈前可以取error_type的名稱(ex: mem_rd)
static void load_input(float* in, hls::stream<float>& inStream, int size) {
mem_rd:
    for(int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        inStream << in[i];
    }
}

// static void compute_add(hls::stream<float>& in1_stream,
//                         hls::stream<float>& out_stream,
//                         int size) {
// // The kernel is operating with vector of NUM_WORDS integers. The + operator performs
// // an element-wise add, resulting in NUM_WORDS parallel additions.
// execute:
//     for (int i = 0; i < size; i++) {
//     	float a = in1_stream.read();
//         out_stream << (a * 0.5 * (1 + tanh(sqrt(2 / 3.141592653589) * (a + 0.044715 * pow(a, 3)))));
// //    	out_stream << (a * 0.5 * (1 + tanh(0.7978845608 * (a + 0.044715 * pow(a, 3)))));
// //    	out_stream << a * 0.5 + 0.08;
//     }
// }

static void compute_norm(hls::stream<float>& in1_stream,
                         hls::stream<float>& out_stream,
                         int BATCH_SIZE,
                         int HEIGHT,
                         int WIDTH) {


execute_norm:
    for(int i = 0; i < BATCH_SIZE; i++) {
        float mean = 0;
        float variance = 0;
        int size = HEIGHT * WIDTH;

        // Read input and compute mean
        float temp_input[512][768];
        compute_norm_mean:
        for(int j = 0; j < HEIGHT; j++) {
#pragma HLS PIPELINE II=1
            for(int k = 0; k < WIDTH; k++) {
                float val = in1_stream.read();
                temp_input[j][k] = val;
                mean = mean + val;
            }
        }
        mean = mean / size;

        // Compute variance
        compute_norm_variance:
        for(int j = 0; j < HEIGHT; j++) {
#pragma HLS PIPELINE II=1
            for(int k = 0; k < WIDTH; k++) {
                float diff = temp_input[j][k] - mean;
                variance = variance + diff * diff;
            }
        }
        variance = variance / size;

        // Normalize
        compute_norm_normalize:
        for(int j = 0; j < HEIGHT; j++) {
#pragma HLS PIPELINE II=1
            for(int k = 0; k < WIDTH; k++) {
                float norm = (temp_input[j][k] - mean) / sqrt(variance);
                out_stream << norm;
            }
        }
    }
}

static void store_result(float* out, hls::stream<float>& out_stream, int size) {
mem_wr:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = out_stream.read();
    }
}

extern "C" {

/*
    Vector Addition Kernel

    Arguments:
        in1  (input)  --> Input vector 1
        in2  (input)  --> Input vector 2
        out  (output) --> Output vector
        size (input)  --> Number of elements in vector
*/

/*
|  BATCH_SIZE = 1  |
|  HEIGHT = 512    |
|  WIDTH = 768     |
*/

void layer_norm(float* in1, float* out, int BATCH_SIZE, int HEIGHT, int WIDTH) {
#pragma HLS INTERFACE m_axi port=in1 offset=slave bundle=gmem0 depth=4096 max_read_burst_length=256 num_read_outstanding=16 num_write_outstanding=16 max_write_burst_length=256
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1 depth=4096 max_read_burst_length=256 num_read_outstanding=16 num_write_outstanding=16 max_write_burst_length=256
#pragma HLS INTERFACE s_axilite port=in1 bundle=control
#pragma HLS INTERFACE s_axilite port=out bundle=control
#pragma HLS INTERFACE s_axilite port=BATCH_SIZE bundle=control
#pragma HLS INTERFACE s_axilite port=HEIGHT bundle=control
#pragma HLS INTERFACE s_axilite port=WIDTH bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	static hls::stream<float> in1_stream("input_stream_1");
	static hls::stream<float> out_stream("output_stream");

//#pragma HLS STREAM variable=in1_stream depth=393216
//#pragma HLS STREAM variable=out_stream depth=393216

#pragma HLS STREAM variable=in1_stream depth=4096
#pragma HLS STREAM variable=out_stream depth=4096

    int size = BATCH_SIZE * HEIGHT * WIDTH;

#pragma HLS dataflow
    load_input(in1, in1_stream, size);
    compute_norm(in1_stream, out_stream, BATCH_SIZE, HEIGHT, WIDTH);
    store_result(out, out_stream, size);

}
}
