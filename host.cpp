/**********
Copyright (c) 2018, Xilinx, Inc.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
**********/

/*******************************************************************************
** HOST Code
*******************************************************************************/

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include <math.h>
#include <cmath>
#include "host.hpp"
#include "WPE_WTE_add.hpp"
#include "decode.hpp"
#include "sample.hpp"
#include "encode.hpp"

#define OCL_CHECK(error, call)                                                                   \
    do {                                                                                         \
        call;                                                                                    \
        if (error != CL_SUCCESS) {                                                               \
            printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
            exit(EXIT_FAILURE);                                                                  \
        }                                                                                        \
    } while (0)

using namespace std;

static const string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %f Device result = %f\n";

#define ALL_MESSAGES

// ********************************************************************************** //
// ---------------------------------------------------------------------------------- //
//                          M A I N    F U N C T I O N                                //
// ---------------------------------------------------------------------------------- //
// ********************************************************************************** //


int main(int argc, char* argv[]) {

    #ifdef ALL_MESSAGES
    cout << "HOST-Info: ============================================================= " << endl;
    cout << "HOST-Info: (Step 0) Set & Print Arguments                      " << endl;
    cout << "HOST-Info: ============================================================= " << endl;
    for (int i = 0; i < argc;i++){
        cout << "HOST-Info: CMD Aruguments " << i << " :" << argv[i] << endl;
    }
    #endif

    // TARGET_DEVICE macro needs to be passed from gcc command line
    // ============================================================================
	// Step 1: Check Command Line Arguments
	// ============================================================================
	//    o) argv[1] Platfrom Vendor
	//    o) argv[2] Device Name
	//    o) argv[3] XCLBIN file
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 1) Check Command Line Arguments                      " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

    if(argc != 4){
        cout << "HOST-Error: Incorrect command line syntax " << endl;
		cout << "HOST-Info:  Usage: " << argv[0] << " <Platform_Vendor> <Device_Name> <XCLBIN_File>  <Test Vectors Size>" << endl << endl;
        return EXIT_FAILURE;
    }

    string Target_Platform_Vendor   = argv[1];
	string Target_Device_Name       = argv[2];
	string xclbinFilename           = argv[3];

    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Platform_Vendor   : " << Target_Platform_Vendor << endl;
	cout << "HOST-Info: Device_Name       : " << Target_Device_Name << endl;
	cout << "HOST-Info: XCLBIN_file       : " << xclbinFilename << endl;
	#endif

    // ============================================================================
	// Step 2: Detect Target Platform and Target Device in a system.
	//         Create Context and Command Queue.
	// ============================================================================
	// Variables:
	//   o) Target_Platform_Vendor[] - defined as main() input argument
	//   o) Target_Device_Name[]     - defined as main() input argument
	//
	// After that
	//   o) Create a Context
	//   o) Create a Command Queue
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 2) Detect Target Platform and Target Device in a system " << endl;
	cout << "HOST-Info:          Create Context and Command Queue                     " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

    vector<cl::Platform> platforms;
    vector<cl::Device> devices;
    cl::Device device;
    cl::Context context;
    cl::CommandQueue q;
    cl::CommandQueue q2;
    cl_int err;
    bool found_target_device = false;

    // ------------------------------------------------------------------------------------
	// Step 2.1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
	// ------------------------------------------------------------------------------------
    // Get and store all PLATFORMS
	// ..................................................
    cl::Platform::get(&platforms);
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Number of detected platforms : " << platforms.size() << endl;
	#endif

    // Search for Platform (ex: Xilinx) using: CL_PLATFORM_VENDOR = Target_Platform_Vendor
    // Traversing all Platforms To find Xilinx Platform and targeted Device in Xilinx Platform
    // Check if the current platform matches Target_Platform_Vendor
	// .............................................................
    for (size_t i = 0; i < platforms.size(); i++) {
        cl::Platform platform = platforms[i];
        string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == Target_Platform_Vendor) {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            #ifdef ALL_MESSAGES
	        cout << "HOST-Info: Selected PlatformName[" << i << "] : " << platformName << endl;
	        #endif
            break;
        }
    }

    // ------------------------------------------------------------------------------------
	// Step 2.2:  Get All Devices for selected platform Target_Platform_ID
	//            then search for Xilinx platform (CL_DEVICE_NAME = Target_Device_Name)
	// ------------------------------------------------------------------------------------
	// Get the Number of Devices
	// ............................................................................
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Number of detected devices : " << devices.size() << endl;
	#endif

    // Search for CL_DEVICE_NAME = Target_Device_Name
    // Check if the current device matches Target_Device_Name
	// ............................................................................
    for (size_t i = 0; i < devices.size(); i++) {
        device = devices[i];
        string deviceName = device.getInfo<CL_DEVICE_NAME>();
        if (deviceName == Target_Device_Name) {
            found_target_device = true;
            #ifdef ALL_MESSAGES
	        cout << "HOST-Info: Selected Device[" << i << "] : " << deviceName << endl;
	        #endif
            break;
        }
    }

    if(found_target_device == false){
        cout << "HOST-Error: Unable to find Device : "<< Target_Device_Name << endl;
        return EXIT_FAILURE;
    }

    // ------------------------------------------------------------------------------------
	// Step 2.3: Create Context and Command Queue for selected Device
	// ------------------------------------------------------------------------------------
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Context ... " << endl;
	#endif
    OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Command Queue ... " << endl;
	#endif
    OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
    OCL_CHECK(err, q2 = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));


    // ============================================================================
	// Step 3: Create Program and Kernel
	// ============================================================================
	//   o) Create a Program from a Binary File and Build it
	//   o) Create a Kernel
	// ============================================================================
	#ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 3) Create Program and Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif

	// ------------------------------------------------------------------
	// Step 3.1: Load Binary File from a disk to Memory
	// ------------------------------------------------------------------
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
        printf("HOST-ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Loading " << xclbinFilename << " binary file to memory ..." << endl;
	#endif
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned bin_file_size = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char* buf = new char[bin_file_size];
    bin_file.read(buf, bin_file_size);

    // ------------------------------------------------------------
	// Step 3.2: Create a program using a Binary File
    //           Build (compiles and links) a program executable from binary
	// ------------------------------------------------------------
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Program with Binary ..." << endl;
	#endif
    //typedef vector<std::pair<const void*, size_type> > Binaries;
    cl::Program::Binaries bins = {{buf, bin_file_size}};
    cl::Program program(context, {device}, bins, nullptr, &err);
    if(err != CL_SUCCESS) {
        cout << "HOST-Error: Failed to create a Program from a Binary" << endl;
        return EXIT_FAILURE;
    }

    // -------------------------------------------------------------
	// Step 3.3: Create a Kernels
	// -------------------------------------------------------------
    cl::Kernel krnl_attn, krnl_c_proj, layer_norm, krnl_vadd, krnl_MLP, krnl_linear_head1, krnl_linear_head2;
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating Kernels ..." << endl;
	#endif
    OCL_CHECK(err, krnl_attn = cl::Kernel(program, "krnl_attn", &err));
    OCL_CHECK(err, krnl_c_proj = cl::Kernel(program, "krnl_c_proj", &err));
    OCL_CHECK(err, layer_norm = cl::Kernel(program, "layer_norm", &err));
    OCL_CHECK(err, krnl_vadd = cl::Kernel(program, "krnl_vadd", &err));
    OCL_CHECK(err, krnl_MLP = cl::Kernel(program, "krnl_MLP", &err));
    OCL_CHECK(err, krnl_linear_head1 = cl::Kernel(program, "krnl_linear_1", &err));
    OCL_CHECK(err, krnl_linear_head2 = cl::Kernel(program, "krnl_linear_2", &err));

    // ================================================================
	// Step 4: Prepare Data to Run Kernel
	// ================================================================
	//   o) Create Buffers in Memory & map our OpenCL buffers to get the pointers
    //   o) Use pointer to Generate data for DataIn_1 array
	//   o) Use pointer to Generate data for DataIn_2 array
	//   o) Allocate Memory to store the results: RES array
    //   o) Copy Input Data from Host to Global Memory
	// ================================================================
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: ============================================================= " << endl;
	cout << "HOST-Info: (Step 4) Prepare Data to Run Kernels                           " << endl;
	cout << "HOST-Info: ============================================================= " << endl;
	#endif
    // ------------------------------------------------------------------
	// Step 4.1: Create Buffers buffer_DataIn_1 in Memory & map to ptr_DataIn_1
	//           Generate data for DataIn_1 array by using ptr_DataIn_1
    //           Create Buffers buffer_DataIn_2 in Memory & map to ptr_DataIn_2
	//           Generate data for DataIn_2 array by using ptr_DataIn_2
	//           Allocate Memory to store the results: RES array
	// ------------------------------------------------------------------

    // These commands will allocate memory on the .Device
    // The cl::Buffer objects can be used to reference the memory locations on the device.


    /*-- ptr_ln_Data_in1 --*/
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Allocating Memory buffer... " << endl;
    #endif
    OCL_CHECK(err, buffer_Data_ln_Data_in1 = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_ln_inout, NULL, &err));
    OCL_CHECK(err, buffer_Data_ln_Data_g = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_ln_GandB, NULL, &err));
    OCL_CHECK(err, buffer_Data_ln_Data_b = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_ln_GandB, NULL, &err));
    OCL_CHECK(err, buffer_Data_ln_out = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_ln_inout, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_w_tiling_q = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_weight_tiled, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_w_tiling_k = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_weight_tiled, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_w_tiling_v = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_weight_tiled, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_bias_tiling_q = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_bias_tiled, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_bias_tiling_k = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_bias_tiled, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_bias_tiling_v = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_bias_tiled, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_k_tiling = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_attn_cache_tiled, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_v_tiling = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_attn_cache_tiled, NULL, &err));
    OCL_CHECK(err, buffer_attn_qkv_result = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_attn_out, NULL, &err));
    OCL_CHECK(err, buffer_attn_cur_k = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_attn_cur_k_v, NULL, &err));
    OCL_CHECK(err, buffer_attn_cur_v = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_attn_cur_k_v, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_c_proj_weight = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_proj_weight, NULL, &err));
    OCL_CHECK(err, buffer_Data_attn_c_proj_bias = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_c_proj_bias, NULL, &err));
    OCL_CHECK(err, buffer_attn_out = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_attn_out, NULL, &err));
    OCL_CHECK(err, buffer_vadd_out = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_vadd_out, NULL, &err));
    OCL_CHECK(err, buffer_mlp_w1 = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_mlp_weight, NULL, &err));
    OCL_CHECK(err, buffer_mlp_b1 = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_mlp_bias_1, NULL, &err));
    OCL_CHECK(err, buffer_mlp_w2 = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_mlp_weight, NULL, &err));
    OCL_CHECK(err, buffer_mlp_b2 = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_mlp_bias_2, NULL, &err));
    OCL_CHECK(err, buffer_mlp_out = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_mlp_out, NULL, &err));
    OCL_CHECK(err, buffer_layer_out = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_mlp_out, NULL, &err));
    OCL_CHECK(err, buffer_ll_head1_in = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_mlp_out, NULL, &err));
    OCL_CHECK(err, buffer_ll_head1_weight = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_ll_weight, NULL, &err));
    OCL_CHECK(err, buffer_ll_head1_res = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_ll_res, NULL, &err));
    OCL_CHECK(err, buffer_ll_head2_in = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_mlp_out, NULL, &err));
    OCL_CHECK(err, buffer_ll_head2_weight = cl::Buffer(context, CL_MEM_READ_ONLY, size_in_bytes_ll_weight, NULL, &err));
    OCL_CHECK(err, buffer_ll_head2_res = cl::Buffer(context, CL_MEM_READ_WRITE, size_in_bytes_ll_res, NULL, &err));


    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Mapping buffer to ptr... " << endl;
    #endif
    OCL_CHECK(err, ptr_ln_Data_in1 = (float*)q.enqueueMapBuffer(buffer_Data_ln_Data_in1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_ln_inout, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ln_Data_g = (float*)q.enqueueMapBuffer(buffer_Data_ln_Data_g, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_ln_GandB, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ln_Data_b = (float*)q.enqueueMapBuffer(buffer_Data_ln_Data_b, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_ln_GandB, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ln_out = (float*)q.enqueueMapBuffer(buffer_Data_ln_out, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_ln_inout, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_w_tiling_q = (float*)q.enqueueMapBuffer(buffer_Data_attn_w_tiling_q, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_weight_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_w_tiling_k = (float*)q.enqueueMapBuffer(buffer_Data_attn_w_tiling_k, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_weight_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_w_tiling_v = (float*)q.enqueueMapBuffer(buffer_Data_attn_w_tiling_v, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_weight_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_bias_tiling_q = (float*)q.enqueueMapBuffer(buffer_Data_attn_bias_tiling_q, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_bias_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_bias_tiling_k = (float*)q.enqueueMapBuffer(buffer_Data_attn_bias_tiling_k, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_bias_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_bias_tiling_v = (float*)q.enqueueMapBuffer(buffer_Data_attn_bias_tiling_v, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_bias_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_k_tiling = (float*)q.enqueueMapBuffer(buffer_Data_attn_k_tiling, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_attn_cache_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_v_tiling = (float*)q.enqueueMapBuffer(buffer_Data_attn_v_tiling, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_attn_cache_tiled, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_qkv_result = (float*)q.enqueueMapBuffer(buffer_attn_qkv_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_attn_out, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_cur_k = (float*)q.enqueueMapBuffer(buffer_attn_cur_k, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_attn_cur_k_v, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_cur_v = (float*)q.enqueueMapBuffer(buffer_attn_cur_v, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_attn_cur_k_v, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_c_proj_weight = (float*)q.enqueueMapBuffer(buffer_Data_attn_c_proj_weight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_proj_weight, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_Data_c_proj_bias = (float*)q.enqueueMapBuffer(buffer_Data_attn_c_proj_bias, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_proj_bias, NULL, NULL, &err));
    OCL_CHECK(err, ptr_attn_out = (float*)q.enqueueMapBuffer(buffer_attn_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_attn_out, NULL, NULL, &err));
    OCL_CHECK(err, ptr_vadd_out = (float*)q.enqueueMapBuffer(buffer_vadd_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_vadd_out, NULL, NULL, &err));
    OCL_CHECK(err, ptr_mlp_w1 = (float*)q.enqueueMapBuffer(buffer_mlp_w1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_mlp_weight, NULL, NULL, &err));
    OCL_CHECK(err, ptr_mlp_b1 = (float*)q.enqueueMapBuffer(buffer_mlp_b1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_mlp_bias_1, NULL, NULL, &err));
    OCL_CHECK(err, ptr_mlp_w2 = (float*)q.enqueueMapBuffer(buffer_mlp_w2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_mlp_weight, NULL, NULL, &err));
    OCL_CHECK(err, ptr_mlp_b2 = (float*)q.enqueueMapBuffer(buffer_mlp_b2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_mlp_bias_2, NULL, NULL, &err));
    OCL_CHECK(err, ptr_mlp_out = (float*)q.enqueueMapBuffer(buffer_mlp_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_mlp_out, NULL, NULL, &err));
    OCL_CHECK(err, ptr_layer_out = (float*)q.enqueueMapBuffer(buffer_layer_out, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_mlp_out, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ll_Data_head1_in = (float*)q.enqueueMapBuffer(buffer_ll_head1_in, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_mlp_out, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ll_Data_head1_weight = (float*)q.enqueueMapBuffer(buffer_ll_head1_weight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_ll_weight, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ll_head1_res = (float*)q.enqueueMapBuffer(buffer_ll_head1_res, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_ll_res, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ll_Data_head2_in = (float*)q2.enqueueMapBuffer(buffer_ll_head2_in, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_mlp_out, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ll_Data_head2_weight = (float*)q2.enqueueMapBuffer(buffer_ll_head2_weight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_ll_weight, NULL, NULL, &err));
    OCL_CHECK(err, ptr_ll_head2_res = (float*)q2.enqueueMapBuffer(buffer_ll_head2_res, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_ll_res, NULL, NULL, &err));

    // =====================
    //      dataprepare
    // =====================
    #ifdef ALL_MESSAGES
    cout << "HOST-Info: Opening File ... " << endl;
    #endif
    open_file();
    OCL_CHECK(err, err = layer_norm.setArg(1, buffer_Data_ln_out));
    OCL_CHECK(err, err = layer_norm.setArg(2, buffer_Data_ln_Data_g));
    OCL_CHECK(err, err = layer_norm.setArg(3, buffer_Data_ln_Data_b));

    OCL_CHECK(err, err = krnl_attn.setArg(0, buffer_Data_ln_out));
    OCL_CHECK(err, err = krnl_attn.setArg(1, buffer_Data_attn_w_tiling_q));
    OCL_CHECK(err, err = krnl_attn.setArg(2, buffer_Data_attn_w_tiling_k));
    OCL_CHECK(err, err = krnl_attn.setArg(3, buffer_Data_attn_w_tiling_v));
    OCL_CHECK(err, err = krnl_attn.setArg(4, buffer_Data_attn_bias_tiling_q));
    OCL_CHECK(err, err = krnl_attn.setArg(5, buffer_Data_attn_bias_tiling_k));
    OCL_CHECK(err, err = krnl_attn.setArg(6, buffer_Data_attn_bias_tiling_v));
    OCL_CHECK(err, err = krnl_attn.setArg(7, buffer_Data_attn_k_tiling));
    OCL_CHECK(err, err = krnl_attn.setArg(8, buffer_Data_attn_v_tiling));
    OCL_CHECK(err, err = krnl_attn.setArg(12, buffer_attn_qkv_result));
    OCL_CHECK(err, err = krnl_attn.setArg(13, buffer_attn_cur_k));
    OCL_CHECK(err, err = krnl_attn.setArg(14, buffer_attn_cur_v));

    OCL_CHECK(err, err = krnl_c_proj.setArg(0, buffer_attn_qkv_result));
    OCL_CHECK(err, err = krnl_c_proj.setArg(1, buffer_Data_attn_c_proj_weight));
    OCL_CHECK(err, err = krnl_c_proj.setArg(2, buffer_Data_attn_c_proj_bias));
    OCL_CHECK(err, err = krnl_c_proj.setArg(4, buffer_attn_out));

    OCL_CHECK(err, err = krnl_MLP.setArg(0, buffer_Data_ln_out));
    OCL_CHECK(err, err = krnl_MLP.setArg(1, buffer_mlp_w1));
    OCL_CHECK(err, err = krnl_MLP.setArg(2, buffer_mlp_b1));
    OCL_CHECK(err, err = krnl_MLP.setArg(3, buffer_mlp_w2));
    OCL_CHECK(err, err = krnl_MLP.setArg(4, buffer_mlp_b2));
    OCL_CHECK(err, err = krnl_MLP.setArg(6, buffer_mlp_out));

    OCL_CHECK(err, err = krnl_linear_head1.setArg(0, buffer_ll_head1_in));
    OCL_CHECK(err, err = krnl_linear_head1.setArg(1, buffer_ll_head1_weight));
    OCL_CHECK(err, err = krnl_linear_head1.setArg(2, buffer_ll_head1_res));

    OCL_CHECK(err, err = krnl_linear_head2.setArg(0, buffer_ll_head2_in));
    OCL_CHECK(err, err = krnl_linear_head2.setArg(1, buffer_ll_head2_weight));
    OCL_CHECK(err, err = krnl_linear_head2.setArg(2, buffer_ll_head2_res));

    cl::Event event1, event2;


    // vector<float> Array_ln_Data_in1;
    int next_token;
    vector<int> Array_next_token(1);
    // WPE_WTE_add(Array_ln_Data_in1);
    // if (!Array_ln_Data_in1.empty()) {
    //     std::cout << "Array_ln_Data_in1: " << Array_ln_Data_in1[0] << " " << Array_ln_Data_in1[1] << std::endl;
    // } else {
    //     std::cerr << "Array_ln_Data_in1 is empty after WPE_WTE_add" << std::endl;
    // }
    total_kv_time = total_decode_time = total_sample_time = total_wwa_time = 0;
    Array_tokenization_out = encode();
    s = Array_tokenization_out.size();
    cout << "===== input len: " << s << " =====" << endl;

    for (int iter = 0; iter < 100; iter++) {
        if(iter > 1) s += 1;
        int query_s = (iter != 0) ? 1 : s;
        cout << "===== ITER" << iter << "=====" << endl;
        Array_ln_Data_in1.clear();
        if(iter == 0) {
            total_wwa_time += WPE_WTE_add(Array_tokenization_out, Array_ln_Data_in1, 0);
            cout << "Array_ln_Data_in1: " << Array_ln_Data_in1[0] << " " << Array_ln_Data_in1[1] << endl;
        }
        else {
            cout << "get next token: " << next_token << endl;
            Array_next_token[0] = next_token;
            total_wwa_time += WPE_WTE_add(Array_next_token, Array_ln_Data_in1, s);
//            cout << "Array_ln_Data_in1: " << Array_ln_Data_in1[0] << " " << Array_ln_Data_in1[1] << endl;
        }
        /*-- ptr_ln_Data_in1 --*/
        cout << Array_ln_Data_in1.size() << endl;
        dataPrepare(ptr_ln_Data_in1, Array_ln_Data_in1, LN_DATA_WIDTH*query_s, ln_Data_in1, -1, s, iter, -1);
        cout << "ptr_ln_Data_in1: " << ptr_ln_Data_in1[0] << " " << ptr_ln_Data_in1[1] << endl;

        // int block = 0;
        for (int block = 0; block < 12; block++) {
            /*-- ptr_ln_Data_g --*/
            dataPrepare(ptr_ln_Data_g, Array_ln_Data_g[block], LN_DATA_WIDTH, ln_Data_g, -1, s, iter, block);
//            cout << "ptr_ln_Data_g: " << ptr_ln_Data_g[0] << " " << ptr_ln_Data_g[1] << endl;

            /*-- ptr_ln_Data_b --*/
            dataPrepare(ptr_ln_Data_b, Array_ln_Data_b[block], LN_DATA_WIDTH, ln_Data_b, -1, s, iter, block);
//            cout << "ptr_ln_Data_b: " << ptr_ln_Data_b[0] << " " << ptr_ln_Data_b[1] << endl;

            if(block == 0) OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_Data_ln_Data_in1, buffer_Data_ln_Data_g, buffer_Data_ln_Data_b}, 0 /* 0 means from host*/));
            else OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_layer_out, buffer_Data_ln_Data_g, buffer_Data_ln_Data_b}, 0 /* 0 means from host*/));

            // ----------------------------------------
            // Step 5.1: Set Kernel Arguments
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Setting Kernel arguments ..." << endl;
            #endif

            if(block == 0) OCL_CHECK(err, err = layer_norm.setArg(0, buffer_Data_ln_Data_in1));
            else OCL_CHECK(err, err = layer_norm.setArg(0, buffer_layer_out));
            OCL_CHECK(err, err = layer_norm.setArg(4, query_s));

            // ----------------------------------------
            // Step 5.2: Submit Kernels for Execution
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Submitting Kernel layer_norm ..." << endl;
            #endif
            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(layer_norm));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_Data_ln_out}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());
//            for (int i = 0; i<768; i++) cout << ptr_ln_out[i] << endl;
            for (int host_wi = 0; host_wi<12; host_wi++) {
                dataPrepare(ptr_attn_Data_w_tiling_q, Array_attn_Data_w_tiling[block], DATA_SIZE_C_ATTN_WEIGHT_TILED, attn_Data_w_tiling_q, host_wi, s, iter, block);
//                cout << "w_tiling_q: " << ptr_attn_Data_w_tiling_q[0] << " " << ptr_attn_Data_w_tiling_q[1] << endl;

                dataPrepare(ptr_attn_Data_w_tiling_k, Array_attn_Data_w_tiling[block], DATA_SIZE_C_ATTN_WEIGHT_TILED, attn_Data_w_tiling_k, host_wi, s, iter, block);
//                cout << "w_tiling_k: " << ptr_attn_Data_w_tiling_k[0] << " " << ptr_attn_Data_w_tiling_k[1] << endl;

                dataPrepare(ptr_attn_Data_w_tiling_v, Array_attn_Data_w_tiling[block], DATA_SIZE_C_ATTN_WEIGHT_TILED, attn_Data_w_tiling_v, host_wi, s, iter, block);
//                cout << "w_tiling_v: " << ptr_attn_Data_w_tiling_v[0] << " " << ptr_attn_Data_w_tiling_v[1] << endl;

                dataPrepare(ptr_attn_Data_bias_tiling_q, Array_attn_Data_bias_tiling[block], DATA_SIZE_C_ATTN_BIAS_TILED, attn_Data_bias_tiling_q, host_wi, s, iter, block);
//                cout << "bias_tiling_q: " << ptr_attn_Data_bias_tiling_q[0] << " " << ptr_attn_Data_bias_tiling_q[1] << endl;

                dataPrepare(ptr_attn_Data_bias_tiling_k, Array_attn_Data_bias_tiling[block], DATA_SIZE_C_ATTN_BIAS_TILED, attn_Data_bias_tiling_k, host_wi, s, iter, block);
//                cout << "bias_tiling_k: " << ptr_attn_Data_bias_tiling_k[0] << " " << ptr_attn_Data_bias_tiling_k[1] << endl;

                dataPrepare(ptr_attn_Data_bias_tiling_v, Array_attn_Data_bias_tiling[block], DATA_SIZE_C_ATTN_BIAS_TILED, attn_Data_bias_tiling_v, host_wi, s, iter, block);
//                cout << "bias_tiling_v: " << ptr_attn_Data_bias_tiling_v[0] << " " << ptr_attn_Data_bias_tiling_v[1] << endl;

                dataPrepare(ptr_attn_Data_k_tiling, k_cache[block], DATA_SIZE_ATTN_CACHE_TILED, attn_Data_k_tiling, host_wi, s, iter, block);
//                cout << "block" << block << ": k_tiling: " << ptr_attn_Data_k_tiling[0] << " " << ptr_attn_Data_k_tiling[1] << endl;

                dataPrepare(ptr_attn_Data_v_tiling, v_cache[block], DATA_SIZE_ATTN_CACHE_TILED, attn_Data_v_tiling, host_wi, s, iter, block);
//                cout << "block" << block << ": v_tiling: " << ptr_attn_Data_v_tiling[0] << " " << ptr_attn_Data_v_tiling[1] << endl;

                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_Data_ln_out, buffer_Data_attn_w_tiling_q, buffer_Data_attn_w_tiling_k, buffer_Data_attn_w_tiling_v, buffer_Data_attn_bias_tiling_q, buffer_Data_attn_bias_tiling_k, buffer_Data_attn_bias_tiling_v, buffer_Data_attn_k_tiling, buffer_Data_attn_v_tiling}, 0 /* 0 means from host*/));

                // ----------------------------------------
                // Step 5.1: Set Kernel Arguments
                // ----------------------------------------
                #ifdef ALL_MESSAGES
                cout << "HOST-Info: Setting Kernel attn arguments ..." << endl;
                #endif
                // set the kernel Arguments
                OCL_CHECK(err, err = krnl_attn.setArg(9, s)); // prefill len for iter0 and 1, then +1 for each iteration
                OCL_CHECK(err, err = krnl_attn.setArg(10, iter));
                OCL_CHECK(err, err = krnl_attn.setArg(11, host_wi));

                // ----------------------------------------
                // Step 5.2: Submit Kernels for Execution
                // ----------------------------------------
                #ifdef ALL_MESSAGES
                cout << "HOST-Info: Submitting Kernel krnl_attn ..." << endl;
                #endif
                // Launch the Kernel
                OCL_CHECK(err, err = q.enqueueTask(krnl_attn));
                OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_attn_qkv_result, buffer_attn_cur_k, buffer_attn_cur_v}, CL_MIGRATE_MEM_OBJECT_HOST));
                OCL_CHECK(err, q.finish());

                // cout << "ptr_attn_cur_k: " << ptr_attn_cur_k[0] << " " << ptr_attn_cur_k[1] << endl;
                total_kv_time += upd_kv_cache(ptr_attn_cur_k, ptr_attn_cur_v, block, host_wi, iter);
            }
            // if(iter == 0) {
            //     cout << "======= KV CACHE" << iter << "block: " << block << " ======="<< endl;
            //     for (int j = 0; j<s*64*12; j++) {
            //         cout << j << ": " << k_cache[block][j] << endl;
            //     }
            // }
            // if(iter == 1 && block == 0) {
            //     host_res_prepare(host_result, query_s*768, 15, iter);
            //     cout << "======= QKV RES =======" << endl;
            //     float max_err = abs(host_result[0] - ptr_attn_qkv_result[0]);
            //     float maxe_host = host_result[0];
            //     float maxe_ptr = ptr_attn_qkv_result[0];
            //     for (int i = 0; i < 768*query_s; i++) {
            //         if(abs(host_result[i] - ptr_attn_qkv_result[i]) > 0.001)
            //             cout << i << " host_result: " << host_result[i] << " ptr_result: " << ptr_attn_qkv_result[i] << endl;
            //         if(abs(host_result[i] - ptr_attn_qkv_result[i]) > max_err) {
            //             max_err = abs(host_result[i] - ptr_attn_qkv_result[i]);
            //             maxe_host = host_result[i];
            //             maxe_ptr = ptr_attn_qkv_result[i];
            //         }
            //     }
            //     float cnt = 0;
            //     int cnt_0 = 0;
            //     for (int i = 0; i<768*query_s; i++) {
            //         if(host_result[i] != 0) cnt += abs((abs(host_result[i] - ptr_attn_qkv_result[i])/host_result[i]));
            //         else cnt_0++;
            //     }
            //     cout << cnt << endl;
            //     cnt /= (768*query_s-cnt_0);
            //     cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;
            //     cout << "max_err_host: " << maxe_host << " max_err_ptr: " << maxe_ptr << " diff: "  << max_err << endl;
            // }
            dataPrepare(ptr_attn_Data_c_proj_weight, Array_attn_Data_c_proj_weight[block], DATA_SIZE_C_PROJ_WEIGHT, attn_Data_c_proj_weight, -1, s, iter, block);
//            cout << "c_proj_weight: " << ptr_attn_Data_c_proj_weight[0] << " " << ptr_attn_Data_c_proj_weight[1] << endl;

            dataPrepare(ptr_attn_Data_c_proj_bias, Array_attn_Data_c_proj_bias[block], DATA_SIZE_C_PROJ_BIAS, attn_Data_c_proj_bias, -1, s, iter, block);
//            cout << "c_proj_bias: " << ptr_attn_Data_c_proj_bias[0] << " " << ptr_attn_Data_c_proj_bias[1] << endl;

            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_attn_qkv_result, buffer_Data_attn_c_proj_weight, buffer_Data_attn_c_proj_bias, buffer_attn_out}, 0 /* 0 means from host*/));

            #ifdef ALL_MESSAGES
                cout << "HOST-Info: Setting Kernel cproj arguments ..." << endl;
            #endif
            OCL_CHECK(err, err = krnl_c_proj.setArg(3, query_s));

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Submitting Kernel krnl_c_proj ..." << endl;
            #endif
            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(krnl_c_proj));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_attn_out}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());
//            if(iter == 0 && block == 0) {
//		        host_res_prepare(host_result, query_s*768, 16, iter);
//		        cout << "======= CPROJ RES =======" << endl;
//		        float max_err = abs(host_result[0] - ptr_attn_out[0]);
//		        float maxe_host = host_result[0];
//		        float maxe_ptr = ptr_attn_out[0];
//		        for (int i = 0; i < 768*query_s; i++) {
//		            if(abs(host_result[i] - ptr_attn_out[i]) > 0.001)
//		                cout << i << " host_result: " << host_result[i] << " ptr_result: " << ptr_attn_out[i] << endl;
//		            if(abs(host_result[i] - ptr_attn_out[i]) > max_err) {
//		                max_err = abs(host_result[i] - ptr_attn_out[i]);
//		                maxe_host = host_result[i];
//		                maxe_ptr = ptr_attn_out[i];
//		            }
//		        }
//		        float cnt = 0;
//		        int cnt_0 = 0;
//		        for (int i = 0; i<768*query_s; i++) {
//		            if(host_result[i] != 0) cnt += abs((abs(host_result[i] - ptr_attn_out[i])/host_result[i]));
//		            else cnt_0++;
//		        }
//		        cout << cnt << endl;
//		        cnt /= (768*query_s-cnt_0);
//		        cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;
//		        cout << "max_err_host: " << maxe_host << " max_err_ptr: " << maxe_ptr << " diff: "  << max_err << endl;
//		    }

            #ifdef ALL_MESSAGES
                cout << "HOST-Info: Setting Kernel Vadd arguments ..." << endl;
            #endif

            OCL_CHECK(err, err = krnl_vadd.setArg(0, buffer_attn_out));

            if(block == 0) OCL_CHECK(err, err = krnl_vadd.setArg(1, buffer_Data_ln_Data_in1));
            else OCL_CHECK(err, err = krnl_vadd.setArg(1, buffer_layer_out));

            OCL_CHECK(err, err = krnl_vadd.setArg(2, buffer_vadd_out));
            OCL_CHECK(err, err = krnl_vadd.setArg(3, query_s*768));

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Submitting Kernel Vadd ..." << endl;
            #endif
            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(krnl_vadd));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_vadd_out}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());


            dataPrepare(ptr_ln_Data_g, Array_ln2_Data_g[block], LN_DATA_WIDTH, ln2_Data_g, -1, s, iter, block);
//            cout << "ptr_ln_Data_g: " << ptr_ln_Data_g[0] << " " << ptr_ln_Data_g[1] << endl;

            dataPrepare(ptr_ln_Data_b, Array_ln2_Data_b[block], LN_DATA_WIDTH, ln2_Data_b, -1, s, iter, block);
//            cout << "ptr_ln_Data_b: " << ptr_ln_Data_b[0] << " " << ptr_ln_Data_b[1] << endl;

            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_vadd_out, buffer_Data_ln_Data_g, buffer_Data_ln_Data_b}, 0 /* 0 means from host*/));

            // ----------------------------------------
            // Step 5.1: Set Kernel Arguments
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Setting Kernel layer_norm 2 arguments ..." << endl;
            #endif
            OCL_CHECK(err, err = layer_norm.setArg(0, buffer_vadd_out));
            OCL_CHECK(err, err = layer_norm.setArg(4, query_s));

            // ----------------------------------------
            // Step 5.2: Submit Kernels for Execution
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Submitting Kernel layer_norm 2 ..." << endl;
            #endif
            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(layer_norm));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_Data_ln_out}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());

            dataPrepare(ptr_mlp_w1, Array_mlp_Data_w_1[block], DATA_SIZE_MLP_WEIGHT, mlp_Data_w_1, -1, s, iter, block);
//            cout << "ptr_mlp_w1: " << ptr_mlp_w1[0] << " " << ptr_mlp_w1[1] << endl;

            dataPrepare(ptr_mlp_b1, Array_mlp_Data_b_1[block], DATA_SIZE_MLP_BIAS_1, mlp_Data_b_1, -1, s, iter, block);
//            cout << "ptr_mlp_b1: " << ptr_mlp_b1[0] << " " << ptr_mlp_b1[1] << endl;

            dataPrepare(ptr_mlp_w2, Array_mlp_Data_w_2[block], DATA_SIZE_MLP_WEIGHT, mlp_Data_w_2, -1, s, iter, block);
//            cout << "ptr_mlp_w2: " << ptr_mlp_w2[0] << " " << ptr_mlp_w2[1] << endl;

            dataPrepare(ptr_mlp_b2, Array_mlp_Data_b_2[block], DATA_SIZE_MLP_BIAS_2, mlp_Data_b_2, -1, s, iter, block);
//            cout << "ptr_mlp_b2: " << ptr_mlp_b2[0] << " " << ptr_mlp_b2[1] << endl;

            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_Data_ln_out, buffer_mlp_w1, buffer_mlp_b1, buffer_mlp_w2, buffer_mlp_b2}, 0 /* 0 means from host*/));

            // ----------------------------------------
            // Step 5.1: Set Kernel Arguments
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Setting Kernel krnl_MLP arguments ..." << endl;
            #endif
            OCL_CHECK(err, err = krnl_MLP.setArg(5, query_s));

            // ----------------------------------------
            // Step 5.2: Submit Kernels for Execution
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Submitting Kernel krnl_MLP ..." << endl;
            #endif
            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(krnl_MLP));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_mlp_out}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());

            #ifdef ALL_MESSAGES
                cout << "HOST-Info: Setting Kernel Vadd arguments ..." << endl;
            #endif

            OCL_CHECK(err, err = krnl_vadd.setArg(0, buffer_vadd_out));
            OCL_CHECK(err, err = krnl_vadd.setArg(1, buffer_mlp_out));
            OCL_CHECK(err, err = krnl_vadd.setArg(2, buffer_layer_out));
            OCL_CHECK(err, err = krnl_vadd.setArg(3, query_s*768));

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Submitting Kernel Vadd ..." << endl;
            #endif
            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(krnl_vadd));
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_layer_out}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, q.finish());

            // host_res_prepare(host_result, query_s*768, block+1, iter);
            // cout << "=======Layer_RES" << block << "=======" << endl;
            // float max_err = abs(host_result[0] - ptr_layer_out[0]);
            // float maxe_host = host_result[0];
            // float maxe_ptr = ptr_layer_out[0];
            // for (int i = 0; i < 768*query_s; i++) {
            //     // if(abs(host_result[i] - ptr_layer_out[i]) > 0.001)
            //     //     cout << i << " " << host_result[i] << " " << ptr_layer_out[i] << endl;
            //     if(abs(host_result[i] - ptr_layer_out[i]) > max_err) {
            //         max_err = abs(host_result[i] - ptr_layer_out[i]);
            //         maxe_host = host_result[i];
            //         maxe_ptr = ptr_layer_out[i];
            //     }
            // }
            // float cnt = 0;
            // int cnt_0 = 0;
            // for (int i = 0; i<768*query_s; i++) {
            //     if(host_result[i] != 0) cnt += abs((abs(host_result[i] - ptr_layer_out[i])/host_result[i]));
            //     else cnt_0++;
            // }
            // cout << cnt << endl;
            // cnt /= (768*query_s-cnt_0);
            // cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;
            // cout << "max_err_host: " << maxe_host << " max_err_ptr: " << maxe_ptr << " diff: "  << max_err << endl;
        }

        dataPrepare(ptr_ln_Data_g, Array_lnf_Data_g, LN_DATA_WIDTH, lnf_Data_g, -1, s, iter, -1);
//        cout << "ptr_ln_Data_g: " << ptr_ln_Data_g[0] << " " << ptr_ln_Data_g[1] << endl;

        dataPrepare(ptr_ln_Data_b, Array_lnf_Data_b, LN_DATA_WIDTH, lnf_Data_b, -1, s, iter, -1);
//        cout << "ptr_ln_Data_b: " << ptr_ln_Data_b[0] << " " << ptr_ln_Data_b[1] << endl;

        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_layer_out, buffer_Data_ln_Data_g, buffer_Data_ln_Data_b}, 0 /* 0 means from host*/));
        // ----------------------------------------
        // Step 5.1: Set Kernel Arguments
        // ----------------------------------------
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Setting Kernel layer_norm final arguments ..." << endl;
        #endif
        OCL_CHECK(err, err = layer_norm.setArg(0, buffer_layer_out));
        OCL_CHECK(err, err = layer_norm.setArg(4, query_s));

        // ----------------------------------------
        // Step 5.2: Submit Kernels for Execution
        // ----------------------------------------
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Submitting Kernel layer_norm final ..." << endl;
        #endif
        // Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(layer_norm));
        OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_Data_ln_out}, CL_MIGRATE_MEM_OBJECT_HOST));
        OCL_CHECK(err, q.finish());

        // host_res_prepare(host_result, query_s*768, 13, iter);
        // cout << "=======LN_RES=======" << endl;
        // float max_err = abs(host_result[0] - ptr_ln_out[0]);
        // float maxe_host = host_result[0];
        // float maxe_ptr = ptr_ln_out[0];
        // int maxe_idx = 0;
        // for (int i = 0; i < 768*query_s; i++) {
        //     // if(abs(host_result[i] - ptr_ln_out[i]) > 0.001)
        //     //     cout << i << " " << host_result[i] << " " << ptr_ln_out[i] << endl;
        //     if(abs(host_result[i] - ptr_ln_out[i]) > max_err) {
        //         max_err = abs(host_result[i] - ptr_ln_out[i]);
        //         maxe_host = host_result[i];
        //         maxe_ptr = ptr_ln_out[i];
        //         maxe_idx = i;
        //     }
        // }
        // float cnt = 0;
        // int cnt_0 = 0;
        // for (int i = 0; i<768*query_s; i++) {
        //     if(host_result[i] != 0) cnt += abs((abs(host_result[i] - ptr_ln_out[i])/host_result[i]));
        //     else cnt_0++;
        // }
        // cout << cnt << endl;
        // cnt /= (768*query_s-cnt_0);
        // cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;
        // cout << "i: " << maxe_idx << " max_err_host: " << maxe_host << " max_err_ptr: " << maxe_ptr << " diff: "  << max_err << endl;

        int ll_depth_size = 0;
        int ll_error_flag = 0;
        cl_ulong time_start1, time_end1;
        cl_ulong time_start2, time_end2;
        double total_execution_time_kernel1 = 0.0;
        double total_execution_time_kernel2 = 0.0;
        double total_overlap_time = 0.0;

        for (int i = 0; i<query_s; i += DATA_SIZE_LL_MAX_DEPTH) {
            if(query_s - i < DATA_SIZE_LL_MAX_DEPTH) {
                ll_depth_size = query_s - i;
            }
            else {
                ll_depth_size = DATA_SIZE_LL_MAX_DEPTH;
            }
            OCL_CHECK(err, err = krnl_linear_head1.setArg(3, ll_depth_size));
            OCL_CHECK(err, err = krnl_linear_head2.setArg(3, ll_depth_size));

            for (int j = 0; j<N1; j += 2) {

                vector<float> ptr_ln_out_vec(ptr_ln_out, ptr_ln_out + query_s*768);
                // for (int i = 0; i<query_s*768; i++) {
                //     if(ptr_ln_out[i] != ptr_ln_out_vec[i])
                //         cout << "ptr_ln_out: " << ptr_ln_out[i] << " ptr_ln_out_vec: " << ptr_ln_out_vec[i] << endl;
                // }
                dataPrepare_tiling(ptr_ll_Data_head1_in, ptr_ln_out_vec, ll_depth_size*K0, 1, j, i);

                dataPrepare_tiling(ptr_ll_Data_head1_weight, Array_last_linear_w, K0*N0, 2, j, i);

                dataPrepare_tiling(ptr_ll_Data_head2_in, ptr_ln_out_vec, ll_depth_size*K0, 1, j, i);

                dataPrepare_tiling(ptr_ll_Data_head2_weight, Array_last_linear_w, K0*N0, 2, j+1, i);

                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_ll_head1_in, buffer_ll_head1_weight}, 0));
                OCL_CHECK(err, err = q2.enqueueMigrateMemObjects({buffer_ll_head2_in, buffer_ll_head2_weight}, 0));

                OCL_CHECK(err, err = q.enqueueTask(krnl_linear_head1, nullptr, &event1));
                OCL_CHECK(err, err = q2.enqueueTask(krnl_linear_head2, nullptr, &event2));
                OCL_CHECK(err, err = q.finish());
                OCL_CHECK(err, err = q2.finish());

                // Time
                event1.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start1);
                event1.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end1);
                event2.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start2);
                event2.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end2);
                double execution_time_kernel1 = (time_end1 - time_start1) * 1.0e-6;
                double execution_time_kernel2 = (time_end2 - time_start2) * 1.0e-6;
                total_execution_time_kernel1 += execution_time_kernel1;
                total_execution_time_kernel2 += execution_time_kernel2;
                cl_ulong overlap_start = std::max(time_start1, time_start2);
                cl_ulong overlap_end = std::min(time_end1, time_end2);
                double overlap_time = 0.0;
                if (overlap_start < overlap_end) {
                    overlap_time = (overlap_end - overlap_start) * 1.0e-6;
                }
                total_overlap_time += overlap_time;

                OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_ll_head1_res}, CL_MIGRATE_MEM_OBJECT_HOST));
                OCL_CHECK(err, err = q2.enqueueMigrateMemObjects({buffer_ll_head2_res}, CL_MIGRATE_MEM_OBJECT_HOST));
                OCL_CHECK(err, err = q.finish());
                OCL_CHECK(err, err = q2.finish());


                for (int h = 0; h < 2; h++) {
                    for (int d = 0; d < ll_depth_size; d++) {
                        for (int k = 0; k < N0; k++) {
                            if (j+h == N1 - 1 && (N0 - k <= 11)) {
                                continue;
                            }
                            if (h == 0) {
                                LL_RES[i * 50257 + j * N0 + d * 50257 + k] = ptr_ll_head1_res[d * N0 + k];
//                                if (abs(ptr_ll_head1_res[d * N0 + k] - host_result_ll[i * 50257 + j * N0 + d * 50257 + k]) > 0.5) {
                                    // cout << "i: " << i << ", j: " << j << ", k: " << k << ", kernel: " <<  ptr_ll_head1_res[d * N0 + k] << ", host: " << host_result_ll[i * 50257 + j * N0 + d * 50257 + k] << ", err: " << abs(ptr_ll_head1_res[d * N0 + k] - host_result_ll[i * 50257 + j * N0 + d * 50257 + k]) << endl;
                                    // ll_error_flag = 1;
                                    // break;
//                                }
                            }
                            else if (h == 1) {
                                LL_RES[i * 50257 + (j + 1) * N0 + d * 50257 + k] = ptr_ll_head2_res[d * N0 + k];
//                                if (abs(ptr_ll_head2_res[d * N0 + k] - host_result_ll[i * 50257 + (j + 1) * N0 + d * 50257 + k]) > 0.5) {
                                    // cout << "i: " << i << ", j: " << (j + 1) << ", k: " << k << ", kernel: " <<  ptr_ll_head2_res[d * N0 + k] << ", host: " << host_result_ll[i * 50257 + (j + 1) * N0 + d * 50257 + k] << ", err: " << abs(ptr_ll_head2_res[d * N0 + k] - host_result_ll[i * 50257 + (j + 1) * N0 + d * 50257 + k]) << endl;
                                    // ll_error_flag = 1;
                                    // break;
//                                }
                            }
                        }
                        // if (ll_error_flag) break;
                    }
                    // if (ll_error_flag) break;
                }
            }

        }
        double parallel_execution_time = total_execution_time_kernel1 + total_execution_time_kernel2 - total_overlap_time;
//        cout << "Total Kernel 1 execution time: " << total_execution_time_kernel1 << " ms" << endl;
//        cout << "Total Kernel 2 execution time: " << total_execution_time_kernel2 << " ms" << endl;
//        cout << "Total Overlap time: " << total_overlap_time << " ms" << endl;
//        cout << "Parallel execution time: " << parallel_execution_time << " ms" << endl;

        // TODO: check result
        // cout << "=======LL_RES=======" << endl;
        // host_res_prepare(host_result_ll, 50257*query_s, 14, iter);
        // max_err = abs(host_result_ll[0] - LL_RES[0]);
        // maxe_host = host_result_ll[0];
        // maxe_ptr = LL_RES[0];
        // maxe_idx = 0;
        // for (int i = 0; i < 50257*query_s; i++) {
        //     // if(abs(host_result_ll[i] - LL_RES[i]) > 0.01)
        //     //     cout << i << " " << host_result_ll[i] << " " << LL_RES[i] << endl;
        //     if(abs(host_result_ll[i] - LL_RES[i]) > max_err) {
        //         max_err = abs(host_result_ll[i] - LL_RES[i]);
        //         maxe_host = host_result_ll[i];
        //         maxe_ptr = LL_RES[i];
        //         maxe_idx = i;
        //     }
        // }
        // cnt = 0;
        // cnt_0 = 0;
        // for (int i = 0; i<50257*query_s; i++) {
        //     if(host_result_ll[i] != 0) cnt += abs((abs(host_result_ll[i] - LL_RES[i])/host_result_ll[i]));
        //     else cnt_0++;
        // }
        // cout << cnt << endl;
        // cnt /= (50257*127-cnt_0);
        // cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;
        // cout << "i: " << maxe_idx << " max_err_host: " << maxe_host << " max_err_ptr: " << maxe_ptr << " diff: "  << max_err << endl;

        vector<float> sample_in(50257*query_s);
        for(size_t i = 0; i<sample_in.size(); i++) {
            sample_in[i] = LL_RES[i];
        }

        // cout << "Array_tokenization_out: " << Array_tokenization_out[0] << " " << Array_tokenization_out[1] << " " << Array_tokenization_out[2] << endl;
        next_token = sample(sample_in, Array_tokenization_out, iter);
//        total_sample_time += tmp_sample_time;
    }
    total_decode_time += Decode(Array_tokenization_out);
    cout << "===== SW Time =====" << endl;
    cout << "kv_cache_time: " << total_kv_time << " s" << endl;
    cout << "decode_time: " << total_decode_time << " s" << endl;
//    cout << "sample_time" << total_sample_time << " s" << endl;
    cout << "wwa_time" << total_wwa_time << " s" << endl;
    cout << "total sw time" << total_kv_time + total_decode_time + total_sample_time + total_wwa_time << " s" << endl;
    // ============================================================================
    // Step 8: Release Allocated Resources
    // ============================================================================
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_ln_Data_in1, ptr_ln_Data_in1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_ln_Data_g, ptr_ln_Data_g));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_ln_Data_b, ptr_ln_Data_b));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_ln_out, ptr_ln_out));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_w_tiling_q, ptr_attn_Data_w_tiling_q));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_w_tiling_k, ptr_attn_Data_w_tiling_k));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_w_tiling_v, ptr_attn_Data_w_tiling_v));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_bias_tiling_q, ptr_attn_Data_bias_tiling_q));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_bias_tiling_k, ptr_attn_Data_bias_tiling_k));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_bias_tiling_v, ptr_attn_Data_bias_tiling_v));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_k_tiling, ptr_attn_Data_k_tiling));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_v_tiling, ptr_attn_Data_v_tiling));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_attn_qkv_result, ptr_attn_qkv_result));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_attn_cur_k, ptr_attn_cur_k));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_attn_cur_v, ptr_attn_cur_v));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_c_proj_weight, ptr_attn_Data_c_proj_weight));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_attn_c_proj_bias, ptr_attn_Data_c_proj_bias));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_attn_out, ptr_attn_out));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_vadd_out, ptr_vadd_out));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_mlp_w1, ptr_mlp_w1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_mlp_b1, ptr_mlp_b1));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_mlp_w2, ptr_mlp_w2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_mlp_b2, ptr_mlp_b2));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_mlp_out, ptr_mlp_out));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_layer_out, ptr_layer_out));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_ll_head1_in, ptr_ll_Data_head1_in));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_ll_head1_weight, ptr_ll_Data_head1_weight));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_ll_head1_res, ptr_ll_head1_res));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_ll_head2_in, ptr_ll_Data_head2_in));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_ll_head2_weight, ptr_ll_Data_head2_weight));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_ll_head2_res, ptr_ll_head2_res));

    OCL_CHECK(err, err = q.finish());
    cout << "HOST-Info: DONE" << endl << endl;
    return 0;
}
