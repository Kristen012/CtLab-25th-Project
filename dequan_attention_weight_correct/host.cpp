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

#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS) {                                                                   \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

using namespace std;

// 可以在這裡塞一點需要用到的常數
static const int DATA_SIZE_X = 768*127; // TODO
static const int DATA_SIZE_C_ATTN_WEIGHT = 1769472;
static const int DATA_SIZE_C_ATTN_WEIGHT_TILED = 49152;
static const int DATA_SIZE_C_ATTN_BIAS = 2304;
static const int DATA_SIZE_C_ATTN_BIAS_TILED = 64;
static const int DATA_SIZE_C_PROJ_WEIGHT = 589824;
static const int DATA_SIZE_C_PROJ_BIAS = 768;
static const int DATA_SIZE_RES = 768*127; // TODO
static const int DATA_SIZE_CACHE_TILED = 65536; // TODO
static const int DATA_SIZE_CUR_K_V = 8192;

// Compute the size of array in bytes
size_t size_in_bytes_x = DATA_SIZE_X * sizeof(float);
size_t size_in_bytes_c_attn_weight_tiled = DATA_SIZE_C_ATTN_WEIGHT_TILED * sizeof(float);
size_t size_in_bytes_c_attn_scale_tiled_quant = DATA_SIZE_C_ATTN_WEIGHT_TILED/128 * sizeof(float);
size_t size_in_bytes_c_attn_qweight_tiled_quant = DATA_SIZE_C_ATTN_WEIGHT_TILED/128 * sizeof(ap_uint<512>);
size_t size_in_bytes_c_attn_zeros_tiled_quant = DATA_SIZE_C_ATTN_WEIGHT_TILED/128/128 * sizeof(ap_uint<512>);
size_t size_in_bytes_c_attn_bias_tiled = DATA_SIZE_C_ATTN_BIAS_TILED * sizeof(float);

size_t size_in_bytes_c_proj_weight = DATA_SIZE_C_PROJ_WEIGHT * sizeof(float);
size_t size_in_bytes_c_proj_qweight = DATA_SIZE_C_PROJ_WEIGHT/128 * sizeof(ap_uint<512>);
size_t size_in_bytes_c_proj_scale = DATA_SIZE_C_PROJ_WEIGHT/128 * sizeof(float);
size_t size_in_bytes_c_proj_zeros = DATA_SIZE_C_PROJ_WEIGHT/128/128 * sizeof(ap_uint<512>);
size_t size_in_bytes_c_proj_bias = DATA_SIZE_C_PROJ_BIAS * sizeof(float);

size_t size_in_bytes_cache_tiled = DATA_SIZE_CACHE_TILED * sizeof(float);

size_t size_in_bytes_res = DATA_SIZE_RES * sizeof(float);
size_t size_in_bytes_cur_k_v = DATA_SIZE_CUR_K_V * sizeof(float);

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
    cl::Kernel krnl_attn;
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: krnl_attn ..." << endl;
	#endif
    OCL_CHECK(err, krnl_attn = cl::Kernel(program, "krnl_attn", &err));

    cl::Kernel krnl_c_proj;
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: krnl_c_proj ..." << endl;
	#endif
    OCL_CHECK(err, krnl_c_proj = cl::Kernel(program, "krnl_c_proj", &err));

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
    /* 此處可以定義input和output的型別 */
    float* ptr_Data_x; // 0
    ap_uint<512>* ptr_Data_w_tiling_q_qweight; // 1
    ap_uint<512>* ptr_Data_w_tiling_q_zeros; // 2
    float* ptr_Data_w_tiling_q_scale; // 3
    ap_uint<512>* ptr_Data_w_tiling_k_qweight; // 4
    ap_uint<512>* ptr_Data_w_tiling_k_zeros; // 5
    float* ptr_Data_w_tiling_k_scale; // 6
    ap_uint<512>* ptr_Data_w_tiling_v_qweight; // 7
    ap_uint<512>* ptr_Data_w_tiling_v_zeros; // 8
    float* ptr_Data_w_tiling_v_scale; // 9
    float* ptr_Data_bias_tiling_q; // 10
    float* ptr_Data_bias_tiling_k; // 11
    float* ptr_Data_bias_tiling_v; // 12
    float* ptr_Data_k_tiling; // 13
    float* ptr_Data_v_tiling; // 14
    ap_uint<512>* ptr_Data_c_proj_qweight; // 15
    ap_uint<512>* ptr_Data_c_proj_zeros; // 16
    float* ptr_Data_c_proj_scale; // 17
    float* ptr_Data_c_proj_bias; // 18

    float* ptr_qkv_result; // 19
    float* ptr_deq_q_weight; // 19
    float* ptr_deq_k_weight; // 19
    float* ptr_deq_v_weight; // 19
    float* ptr_cur_k;
    float* ptr_cur_v;
    float* ptr_attn_result;
    float* ptr_proj_deqweight;


    // These commands will allocate memory on the .Device
    // The cl::Buffer objects can be used to reference the memory locations on the device.
    /* 以下區域為input data傳入host的code，只需要複製並且更改參數和型別即可(size_in_bytes(考慮input大小(byte)), ptr_DataIn_1, DATA_SIZE(考慮input大小))*/
        /*-- ptr_Data_x --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_x for Data_x ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_x(context, CL_MEM_READ_ONLY, size_in_bytes_x, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_x to ptr_Data_x ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_x = (float*)q.enqueueMapBuffer(buffer_Data_x, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_x, NULL, NULL, &err));

        /*-- ptr_Data_w_tiling_q_qweight --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_q_qweight for Data_w_tiling_q_qweight ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_q_qweight(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_qweight_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_q_qweight to ptr_Data_w_tiling_q_qweight ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_q_qweight = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_w_tiling_q_qweight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_qweight_tiled_quant, NULL, NULL, &err));

        /*-- ptr_Data_w_tiling_q_zeros --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_q_zeros for Data_w_tiling_q_zeros ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_q_zeros(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_zeros_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_q_zeros to ptr_Data_w_tiling_q_zeros ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_q_zeros = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_w_tiling_q_zeros, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_zeros_tiled_quant, NULL, NULL, &err));
        
        /*-- ptr_Data_w_tiling_q_scale --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_q_scale for Data_w_tiling_q_scale ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_q_scale(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_scale_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_q_scale to ptr_Data_w_tiling_q_scale ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_q_scale = (float*)q.enqueueMapBuffer(buffer_Data_w_tiling_q_scale, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_scale_tiled_quant, NULL, NULL, &err));

        /*-- ptr_Data_w_tiling_k_qweight --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_k_qweight for Data_w_tiling_k_qweight ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_k_qweight(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_qweight_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_k_qweight to ptr_Data_w_tiling_k_qweight ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_k_qweight = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_w_tiling_k_qweight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_qweight_tiled_quant, NULL, NULL, &err));

        /*-- ptr_Data_w_tiling_k_zeros --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_k_zeros for Data_w_tiling_k_zeros ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_k_zeros(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_zeros_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_k_zeros to ptr_Data_w_tiling_k_zeros ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_k_zeros = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_w_tiling_k_zeros, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_zeros_tiled_quant, NULL, NULL, &err));
        
        /*-- ptr_Data_w_tiling_k_scale --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_k_scale for Data_w_tiling_k_scale ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_k_scale(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_scale_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_k_scale to ptr_Data_w_tiling_k_scale ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_k_scale = (float*)q.enqueueMapBuffer(buffer_Data_w_tiling_k_scale, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_scale_tiled_quant, NULL, NULL, &err));

        /*-- ptr_Data_w_tiling_v_qweight --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_v_qweight for Data_w_tiling_v_qweight ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_v_qweight(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_qweight_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_v_qweight to ptr_Data_w_tiling_v_qweight ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_v_qweight = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_w_tiling_v_qweight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_qweight_tiled_quant, NULL, NULL, &err));

        /*-- ptr_Data_w_tiling_v_zeros --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_v_zeros for Data_w_tiling_v_zeros ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_v_zeros(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_zeros_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_v_zeros to ptr_Data_w_tiling_v_zeros ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_v_zeros = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_w_tiling_v_zeros, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_zeros_tiled_quant, NULL, NULL, &err));
        
        /*-- ptr_Data_w_tiling_v_scale --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_w_tiling_v_scale for Data_w_tiling_v_scale ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_w_tiling_v_scale(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_scale_tiled_quant, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_w_tiling_v_scale to ptr_Data_w_tiling_v_scale ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_w_tiling_v_scale = (float*)q.enqueueMapBuffer(buffer_Data_w_tiling_v_scale, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_scale_tiled_quant, NULL, NULL, &err));
        
        /*-- ptr_Data_bias_tiling_q --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_bias_tiling_q for Data_bias_tiling_q ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_bias_tiling_q(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_bias_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_bias_tiling_q to ptr_Data_bias_tiling_q ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_bias_tiling_q = (float*)q.enqueueMapBuffer(buffer_Data_bias_tiling_q, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_bias_tiled, NULL, NULL, &err));

        /*-- ptr_Data_bias_tiling_k --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_bias_tiling_k for Data_bias_tiling_k ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_bias_tiling_k(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_bias_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_bias_tiling_k to ptr_Data_bias_tiling_k ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_bias_tiling_k = (float*)q.enqueueMapBuffer(buffer_Data_bias_tiling_k, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_bias_tiled, NULL, NULL, &err));

        /*-- ptr_Data_bias_tiling_v --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_bias_tiling_v for Data_bias_tiling_v ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_bias_tiling_v(context, CL_MEM_READ_ONLY, size_in_bytes_c_attn_bias_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_bias_tiling_v to ptr_Data_bias_tiling_v ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_bias_tiling_v = (float*)q.enqueueMapBuffer(buffer_Data_bias_tiling_v, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_bias_tiled, NULL, NULL, &err));

        /*-- ptr_Data_k_tiling --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_k_tiling for Data_k_tiling ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_k_tiling(context, CL_MEM_READ_ONLY, size_in_bytes_cache_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_k_tiling to ptr_Data_k_tiling ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_k_tiling = (float*)q.enqueueMapBuffer(buffer_Data_k_tiling, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_cache_tiled, NULL, NULL, &err));

        /*-- ptr_Data_v_tiling --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_v_tiling for Data_v_tiling ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_v_tiling(context, CL_MEM_READ_ONLY, size_in_bytes_cache_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_v_tiling to ptr_Data_v_tiling ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_v_tiling = (float*)q.enqueueMapBuffer(buffer_Data_v_tiling, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_cache_tiled, NULL, NULL, &err));

        // Init output data buffer
        /*---qkv_res---*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_qkv_result for RES Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_qkv_result(context, CL_MEM_READ_WRITE, size_in_bytes_res, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_qkv_result to ptr_qkv_result ... " << endl;
        #endif
        OCL_CHECK(err, ptr_qkv_result = (float*)q.enqueueMapBuffer(buffer_qkv_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_res, NULL, NULL, &err));

        /*---cur_k---*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_cur_k for CUR_K Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_cur_k(context, CL_MEM_READ_WRITE, size_in_bytes_cur_k_v, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_cur_k to ptr_cur_k ... " << endl;
        #endif
        OCL_CHECK(err, ptr_cur_k = (float*)q.enqueueMapBuffer(buffer_cur_k, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_cur_k_v, NULL, NULL, &err));

        /*---cur_v---*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_cur_v for CUR_V Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_cur_v(context, CL_MEM_READ_WRITE, size_in_bytes_cur_k_v, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_cur_v to ptr_cur_v ... " << endl;
        #endif
        OCL_CHECK(err, ptr_cur_v = (float*)q.enqueueMapBuffer(buffer_cur_v, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_cur_k_v, NULL, NULL, &err));

/*---qkv_res---*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_deq_q_weight for RES Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_deq_q_weight(context, CL_MEM_READ_WRITE, size_in_bytes_c_attn_weight_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_deq_q_weight to ptr_deq_q_weight ... " << endl;
        #endif
        OCL_CHECK(err, ptr_deq_q_weight = (float*)q.enqueueMapBuffer(buffer_deq_q_weight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_attn_weight_tiled, NULL, NULL, &err));
/*---qkv_res---*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_deq_k_weight for RES Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_deq_k_weight(context, CL_MEM_READ_WRITE, size_in_bytes_c_attn_weight_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_deq_k_weight to ptr_deq_k_weight ... " << endl;
        #endif
        OCL_CHECK(err, ptr_deq_k_weight = (float*)q.enqueueMapBuffer(buffer_deq_k_weight, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_c_attn_weight_tiled, NULL, NULL, &err));
/*---qkv_res---*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_deq_v_weight for RES Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_deq_v_weight(context, CL_MEM_READ_WRITE, size_in_bytes_c_attn_weight_tiled, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_deq_v_weight to ptr_deq_v_weight ... " << endl;
        #endif
        OCL_CHECK(err, ptr_deq_v_weight = (float*)q.enqueueMapBuffer(buffer_deq_v_weight, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_c_attn_weight_tiled, NULL, NULL, &err));

/*-- ptr_Data_c_proj_qweight --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_c_proj_qweight for Data_c_proj_qweight ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_c_proj_qweight(context, CL_MEM_READ_ONLY, size_in_bytes_c_proj_qweight, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_c_proj_qweight to ptr_Data_c_proj_qweight ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_c_proj_qweight = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_c_proj_qweight, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_proj_qweight, NULL, NULL, &err));

        /*-- ptr_Data_c_proj_zeros --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_c_proj_zeros for Data_c_proj_zeros ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_c_proj_zeros(context, CL_MEM_READ_ONLY, size_in_bytes_c_proj_zeros, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_c_proj_zeros to ptr_Data_c_proj_zeros ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_c_proj_zeros = (ap_uint<512>*)q.enqueueMapBuffer(buffer_Data_c_proj_zeros, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_proj_zeros, NULL, NULL, &err));
        
        /*-- ptr_Data_c_proj_scale --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_c_proj_scale for Data_c_proj_scale ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_c_proj_scale(context, CL_MEM_READ_ONLY, size_in_bytes_c_proj_scale, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_c_proj_scale to ptr_Data_c_proj_scale ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_c_proj_scale = (float*)q.enqueueMapBuffer(buffer_Data_c_proj_scale, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_proj_scale, NULL, NULL, &err));

        /*-- ptr_Data_c_proj_bias --*/
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_Data_c_proj_bias for Data_c_proj_bias ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_Data_c_proj_bias(context, CL_MEM_READ_ONLY, size_in_bytes_c_proj_bias, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_Data_c_proj_bias to ptr_Data_c_proj_bias ... " << endl;
        #endif
        OCL_CHECK(err,
                ptr_Data_c_proj_bias = (float*)q.enqueueMapBuffer(buffer_Data_c_proj_bias, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_c_proj_bias, NULL, NULL, &err));
// proj dequant weight
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_proj_deqweight for RES Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_proj_deqweight(context, CL_MEM_READ_WRITE, size_in_bytes_c_proj_weight, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_proj_deqweight to ptr_proj_deqweight ... " << endl;
        #endif
        OCL_CHECK(err, ptr_proj_deqweight = (float*)q.enqueueMapBuffer(buffer_proj_deqweight, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_c_proj_weight, NULL, NULL, &err));

        // Init output data buffer
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Allocating Memory buffer_attn_result for RES Array ... " << endl;
        #endif
        OCL_CHECK(err, cl::Buffer buffer_attn_result(context, CL_MEM_WRITE_ONLY, size_in_bytes_res, NULL, &err));
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Mapping buffer_attn_result to ptr_attn_result ... " << endl;
        #endif
        OCL_CHECK(err, ptr_attn_result = (float*)q.enqueueMapBuffer(buffer_attn_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_res, NULL, NULL, &err));

    for (int iter = 0; iter < 1; iter++) {
        if(iter > 1) s += 1;
        int query_s = (iter != 0) ? 1 : s;
        for (int host_wi = 0; host_wi<12; host_wi++) {
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_x ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_x, query_s*768, 0, -1, iter);
            cout << "x: " << ptr_Data_x[0] << " " << ptr_Data_x[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_X << " values" << endl;
            #endif
            
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_q_qweight ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare_int512(ptr_Data_w_tiling_q_qweight, DATA_SIZE_C_ATTN_WEIGHT_TILED, 0, host_wi, iter);
            cout << "w_tiling_q: " << ptr_Data_w_tiling_q_qweight[0].range(3,0) << " " << ptr_Data_w_tiling_q_qweight[0].range(259,256) << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_q_zeros ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare_int512(ptr_Data_w_tiling_q_zeros, DATA_SIZE_C_ATTN_WEIGHT_TILED, 3, host_wi, iter);
            cout << "w_tiling_q: " << ptr_Data_w_tiling_q_zeros[0].range(3,0) << " " << ptr_Data_w_tiling_q_zeros[0].range(259,256) << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_q_scale ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_w_tiling_q_scale, DATA_SIZE_C_ATTN_WEIGHT_TILED, 1, host_wi, iter);
            cout << "w_tiling_q: " << ptr_Data_w_tiling_q_scale[0] << " " << ptr_Data_w_tiling_q_scale[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_k_qweight ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare_int512(ptr_Data_w_tiling_k_qweight, DATA_SIZE_C_ATTN_WEIGHT_TILED, 1, host_wi, iter);
            cout << "w_tiling_k: " << ptr_Data_w_tiling_k_qweight[0].range(3,0) << " " << ptr_Data_w_tiling_k_qweight[1].range(259,256) << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_k_zeros ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare_int512(ptr_Data_w_tiling_k_zeros, DATA_SIZE_C_ATTN_WEIGHT_TILED, 4, host_wi, iter);
            cout << "w_tiling_k: " << ptr_Data_w_tiling_k_zeros[0].range(3,0) << " " << ptr_Data_w_tiling_k_zeros[0].range(259,256) << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_k_scale ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_w_tiling_k_scale, DATA_SIZE_C_ATTN_WEIGHT_TILED, 2, host_wi, iter);
            cout << "w_tiling_k: " << ptr_Data_w_tiling_k_scale[0] << " " << ptr_Data_w_tiling_k_scale[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_v_qweight ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare_int512(ptr_Data_w_tiling_v_qweight, DATA_SIZE_C_ATTN_WEIGHT_TILED, 2, host_wi, iter);
            cout << "w_tiling_v: " << ptr_Data_w_tiling_v_qweight[0].range(3,0) << " " << ptr_Data_w_tiling_v_qweight[1].range(259,256) << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

             #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_v_zeros ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare_int512(ptr_Data_w_tiling_v_zeros, DATA_SIZE_C_ATTN_WEIGHT_TILED, 5, host_wi, iter);
            cout << "w_tiling_v: " << ptr_Data_w_tiling_v_zeros[0].range(3,0) << " " << ptr_Data_w_tiling_v_zeros[1].range(259,256) << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

             #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_w_tiling_v_scale ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_w_tiling_v_scale, DATA_SIZE_C_ATTN_WEIGHT_TILED, 3, host_wi, iter);
            cout << "w_tiling_v: " << ptr_Data_w_tiling_v_scale[0] << " " << ptr_Data_w_tiling_v_scale[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_WEIGHT_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_bias_tiling_q ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_bias_tiling_q, DATA_SIZE_C_ATTN_BIAS_TILED, 4, host_wi, iter);
            cout << "bias_tiling_q: " << ptr_Data_bias_tiling_q[0] << " " << ptr_Data_bias_tiling_q[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_BIAS_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_bias_tiling_k ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_bias_tiling_k, DATA_SIZE_C_ATTN_BIAS_TILED, 5, host_wi, iter);
            cout << "bias_tiling_k: " << ptr_Data_bias_tiling_k[0] << " " << ptr_Data_bias_tiling_k[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_BIAS_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_bias_tiling_v ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_bias_tiling_v, DATA_SIZE_C_ATTN_BIAS_TILED, 6, host_wi, iter);
            cout << "bias_tiling_v: " << ptr_Data_bias_tiling_v[0] << " " << ptr_Data_bias_tiling_v[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_C_ATTN_BIAS_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_k_tiling ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_k_tiling, DATA_SIZE_CACHE_TILED, 7, host_wi, iter);
            cout << "k_tiling: " << ptr_Data_k_tiling[0] << " " << ptr_Data_k_tiling[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_CACHE_TILED << " values" << endl;
            #endif

            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Generating buffer_Data_v_tiling ..." << endl;
            #endif
            // Call dataPrepare to init data
            dataPrepare(ptr_Data_v_tiling, DATA_SIZE_CACHE_TILED, 8, host_wi, iter);
            cout << "v_tiling: " << ptr_Data_v_tiling[0] << " " << ptr_Data_v_tiling[1] << endl;
            #ifdef ALL_MESSAGES
            cout << "           Generated " << DATA_SIZE_CACHE_TILED << " values" << endl;
            #endif
            // Data will be migrated to kernel space
            // 考慮input量，若有多份則要改成{buffer_DataIn1, buffer_DataIn2...}
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_Data_x, buffer_Data_w_tiling_q_qweight, buffer_Data_w_tiling_q_zeros, buffer_Data_w_tiling_q_scale, buffer_Data_w_tiling_k_qweight, buffer_Data_w_tiling_k_zeros, buffer_Data_w_tiling_k_scale, buffer_Data_w_tiling_v_qweight, buffer_Data_w_tiling_v_zeros, buffer_Data_w_tiling_v_scale, buffer_Data_bias_tiling_q, buffer_Data_bias_tiling_k, buffer_Data_bias_tiling_v, buffer_Data_k_tiling, buffer_Data_v_tiling, buffer_deq_q_weight, buffer_deq_k_weight, buffer_deq_v_weight}, 0 /* 0 means from host*/));

            // ============================================================================
            // Step 5: Set Kernel Arguments and Run the Application
            //         o) Set Kernel Arguments
            // 				----------------------------------------------------
            // 				 Kernel	  		Argument Nb		Description
            // 				----------------------------------------------------
            //  			 krnl_attn	    0				GlobMem_BUF_DataIn_1
            //  			 krnl_attn	    1				GlobMem_BUF_DataIn_2
            //  			 krnl_attn	    2				GlobMem_BUF_RES
            //  			 krnl_attn	    3				CONST_arg
            // 				----------------------------------------------------
            //
            //         o) Submit Kernels for Execution
            //         o) Copy Results from Global Memory to Host
            // ============================================================================
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: ============================================================= " << endl;
            cout << "HOST-Info: (Step 5) Run Application                                      " << endl;
            cout << "HOST-Info: ============================================================= " << endl;
            #endif
            // ----------------------------------------
            // Step 5.1: Set Kernel Arguments
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Setting Kernel arguments ..." << endl;
            #endif
            // set the kernel Arguments
            // 需要注意順序數量
            int narg = 0;
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_x));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_q_qweight));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_q_zeros));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_q_scale));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_k_qweight));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_k_zeros));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_k_scale));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_v_qweight));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_v_zeros));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_w_tiling_v_scale));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_bias_tiling_q));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_bias_tiling_k));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_bias_tiling_v));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_k_tiling));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_Data_v_tiling));

            OCL_CHECK(err, err = krnl_attn.setArg(narg++, s)); // prefill len for iter0 and 1, then +1 for each iteration
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, iter));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, host_wi));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_qkv_result));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_cur_k));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_cur_v));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_deq_q_weight));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_deq_k_weight));
            OCL_CHECK(err, err = krnl_attn.setArg(narg++, buffer_deq_v_weight));

            // ----------------------------------------
            // Step 5.2: Submit Kernels for Execution
            // ----------------------------------------
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: Submitting Kernel krnl_attn ..." << endl;
            #endif
            // Launch the Kernel
            OCL_CHECK(err, err = q.enqueueTask(krnl_attn));
            cout << "Point 1" << endl;
            // The result of the previous kernel execution will need to be retrieved in
            // order to view the results. This call will transfer the data from FPGA to
            // source_results vector
            OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_qkv_result, buffer_cur_k, buffer_cur_v, buffer_deq_q_weight, buffer_deq_k_weight, buffer_deq_v_weight}, CL_MIGRATE_MEM_OBJECT_HOST));
            cout << "Point 2" << endl;
            OCL_CHECK(err, q.finish());
            cout << "Point 3" << endl;



            // ============================================================================
            // Step 6: Processing Output Results
            //         o) Check correctness of the output results
            // ============================================================================
            #ifdef ALL_MESSAGES
            cout << "HOST-Info: ============================================================= " << endl;
            cout << "HOST-Info: (Step 6) Check the Output Results                             " << endl;
            cout << "HOST-Info: ============================================================= " << endl;
            #endif

            // ------------------------------------------------------
            // Step 6.1: Check correctness of the output results
            // ------------------------------------------------------
            /* 此處為error detection，可以選擇直接印出來*/
            // bool error_detected = false;

            // cout << "=======K_TILING=======" << endl;
            // for (int i = 0; i<768*query_s; i++) {
            //     cout << ptr_cur_k[i] << endl;
            // }
            // cout << "=======V_TILING=======" << endl;
            // for (int i = 0; i<768*query_s; i++) {
            //     cout << ptr_cur_v[i] << endl;
            // }
            upd_kv_cache(ptr_cur_k, ptr_cur_v, 0, host_wi, iter);
            // ============================================================================
            // Step 7: Custom Profiling
            // ============================================================================
            // cout << "HOST-Info: ============================================================= " << endl;
            // cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
            // cout << "HOST-Info: ============================================================= " << endl;

            // int Nb_Of_Kernels = 1;
            // int Nb_Of_Memory_Tranfers = Nb_Of_Mem_Events;

            // string list_of_kernel_names[Nb_Of_Kernels];
            // list_of_kernel_names[0]="krnl_attn";
            // run_custom_profiling (Nb_Of_Kernels,Nb_Of_Memory_Tranfers,K_exe_event,Mem_op_event,list_of_kernel_names);

            if(iter == 0) {
                float host_result[127*768];
                dataPrepare(host_result, 127*768, 11, -1, iter);
                    cout << "=======" << iter << "=======" << endl;
                    cout << "=======" << host_wi << "=======" << endl;
                    cout << "=======QKV_RES=======" << endl;
                    for (int i = 0; i < 127*64; i++) {
                        // if(abs(host_result[host_wi*127*64 + i] - ptr_qkv_result[i]) > 0.001)
                            cout << i << " " << host_result[host_wi*127*64 + i] << " " << ptr_qkv_result[i] << endl;
                        //   cout << ptr_qkv_result[i] << " ";
                        //   if(i%10 == 0) cout << endl;
                    //        cout << host_result << " " << ptr_qkv_result[i] << endl;
                    //    if (abs(ptr_qkv_result[i] - host_result[i]) > 0.005) {
                    //         cout << "host_result" << host_result[i] << " krnl_result" << " " << ptr_qkv_result[i] << endl;
                    //         // printf(error_message.c_str(), i, host_result, ptr_qkv_result[i]);
                    //         error_detected = true;
                    //         break;
                    //    }
                    }
                
                float deq_result_q[768*64];
                float deq_result_k[768*64];
                float deq_result_v[768*64];
                dataPrepare(deq_result_q, 768*64, 15, host_wi, iter);
                dataPrepare(deq_result_k, 768*64, 16, host_wi, iter);
                dataPrepare(deq_result_v, 768*64, 17, host_wi, iter);
                for (int i = 0; i < 768*64; i++){
                    if(abs(deq_result_q[i] - ptr_deq_q_weight[i]) > 1e-6){
                        cout << "answer deqweight q " << i <<" : " << deq_result_q[i] << "\n";
                        cout << "deqweight q " << i <<" : " << ptr_deq_q_weight[i] << "\n";
                    }
                    if(abs(deq_result_k[i] - ptr_deq_k_weight[i]) > 1e-6){
                        cout << "answer deqweight k " << i <<" : " << deq_result_k[i] << "\n";
                        cout << "deqweight k " << i <<" : " << ptr_deq_k_weight[i] << "\n";
                    }
                    if(abs(deq_result_v[i] - ptr_deq_v_weight[i]) > 1e-6){
                        cout << "answer deqweight v " << i <<" : " << deq_result_v[i] << "\n";
                        cout << "deqweight v " << i <<" : " << ptr_deq_v_weight[i] << "\n";
                    }
                }
            }

            // cout << "HOST-Info: TEST " << (error_detected ? "FAILED" : "PASSED") << endl;
        }
        
        // // ============================================================================
        // // Step 8: Release Allocated Resources
        // // ============================================================================
        // // 同樣需要注意data數量及argument名稱
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_x, ptr_Data_x));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_q, ptr_Data_w_tiling_q));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_k, ptr_Data_w_tiling_k));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_v, ptr_Data_w_tiling_v));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_bias_tiling_q, ptr_Data_bias_tiling_q));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_bias_tiling_k, ptr_Data_bias_tiling_k));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_bias_tiling_v, ptr_Data_bias_tiling_v));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_k_tiling, ptr_Data_k_tiling));
        // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_v_tiling, ptr_Data_v_tiling));

        // // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_qkv_result, ptr_qkv_result));
        // OCL_CHECK(err, err = q.finish());


        // Call dataPrepare to init data
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Generating buffer_Data_c_proj_qweight ..." << endl;
        #endif
        dataPrepare_int512(ptr_Data_c_proj_qweight, DATA_SIZE_C_PROJ_WEIGHT, 6, -1, iter);
        cout << "c_proj_qweight: " << ptr_Data_c_proj_qweight[0] << " " << ptr_Data_c_proj_qweight[1] << endl;
        #ifdef ALL_MESSAGES
        cout << "           Generated " << DATA_SIZE_C_PROJ_WEIGHT << " values" << endl;
        #endif

        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Generating buffer_Data_c_proj_zeros ..." << endl;
        #endif
        dataPrepare_int512(ptr_Data_c_proj_zeros, DATA_SIZE_C_PROJ_WEIGHT/128, 7, -1, iter);
        cout << "c_proj_zeros: " << ptr_Data_c_proj_zeros[0] << " " << ptr_Data_c_proj_zeros[1] << endl;
        #ifdef ALL_MESSAGES
        cout << "           Generated " << DATA_SIZE_C_PROJ_WEIGHT << " values" << endl;
        #endif

        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Generating buffer_Data_c_proj_scale ..." << endl;
        #endif
        dataPrepare(ptr_Data_c_proj_scale, DATA_SIZE_C_PROJ_WEIGHT/128, 9, -1, iter);
        cout << "c_proj_scale: " << ptr_Data_c_proj_scale[0] << " " << ptr_Data_c_proj_scale[1] << endl;
        #ifdef ALL_MESSAGES
        cout << "           Generated " << DATA_SIZE_C_PROJ_WEIGHT << " values" << endl;
        #endif

        // Call dataPrepare to init data
        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Generating buffer_Data_c_proj_weight ..." << endl;
        #endif
        dataPrepare(ptr_Data_c_proj_bias, DATA_SIZE_C_PROJ_BIAS, 10, -1, iter);
        cout << "c_proj_bias: " << ptr_Data_c_proj_bias[0] << " " << ptr_Data_c_proj_bias[1] << endl;
        #ifdef ALL_MESSAGES
        cout << "           Generated " << DATA_SIZE_C_PROJ_BIAS << " values" << endl;
        #endif

        // Data will be migrated to kernel space
        // 考慮input量，若有多份則要改成{buffer_DataIn1, buffer_DataIn2...}
        OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_qkv_result, buffer_Data_c_proj_qweight, buffer_Data_c_proj_zeros, buffer_Data_c_proj_scale, buffer_Data_c_proj_bias, buffer_attn_result, buffer_proj_deqweight}, 0 /* 0 means from host*/));

        int cproj_narg = 0;
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, buffer_qkv_result));
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, buffer_Data_c_proj_qweight));
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, buffer_Data_c_proj_zeros));
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, buffer_Data_c_proj_scale));
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, buffer_Data_c_proj_bias));
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, query_s));
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, buffer_attn_result));
        OCL_CHECK(err, err = krnl_c_proj.setArg(cproj_narg++, buffer_proj_deqweight));


        #ifdef ALL_MESSAGES
        cout << "HOST-Info: Submitting Kernel krnl_c_proj ..." << endl;
        #endif
        // Launch the Kernel
        OCL_CHECK(err, err = q.enqueueTask(krnl_c_proj));
        cout << "Point 1" << endl;
        // The result of the previous kernel execution will need to be retrieved in
        // order to view the results. This call will transfer the data from FPGA to
        // source_results vector
        OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_attn_result, buffer_proj_deqweight}, CL_MIGRATE_MEM_OBJECT_HOST));
        cout << "Point 2" << endl;
        OCL_CHECK(err, q.finish());
        cout << "Point 3" << endl;
        // bool error_detected = false;
        float host_result[768*query_s];
        dataPrepare(host_result, 768*query_s, 12, -1, iter);
        cout << "=======ITER " << iter << " =======" << endl;
        cout << "=======ATTN_RES=======" << endl;
        // for (int i = 0; i < 768*query_s; i++) {
        //     if(abs(host_result[i] - ptr_attn_result[i]) > 0.001)
        //         cout << i << " " << host_result[i] << " " << ptr_attn_result[i] << endl;
        // }
        float deq_result_proj[768*768];
        dataPrepare(deq_result_proj, 768*768, 18, -1, iter);
        for (int i = 0; i < DATA_SIZE_C_PROJ_WEIGHT; i++){
            if(abs(deq_result_proj[i] - ptr_proj_deqweight[i]) > 1e-6){
                cout << "answer deqweight proj " << i <<" : " << deq_result_proj[i] << "\n";
                cout << "deqweight proj " << i <<" : " << ptr_proj_deqweight[i] << "\n";
            }
        }
        cout << "answer deqweight proj  " << 0 <<" : " << deq_result_proj[0] << "\n";
        cout << "deqweight proj " << 0 <<" : " << ptr_proj_deqweight[0] << "\n";
        if(iter == 0) {
            cout << "======= CPROJ RES =======" << endl;
            float max_err = abs(host_result[0] - ptr_attn_result[0]);
            float maxe_host = host_result[0];
            float maxe_ptr = ptr_attn_result[0];
            for (int i = 0; i < 768*query_s; i++) {
                if(abs(host_result[i] - ptr_attn_result[i]) > 0.001)
                    cout << i << " host_result: " << host_result[i] << " ptr_result: " << ptr_attn_result[i] << endl;
                if(abs(host_result[i] - ptr_attn_result[i]) > max_err) {
                    max_err = abs(host_result[i] - ptr_attn_result[i]);
                    maxe_host = host_result[i];
                    maxe_ptr = ptr_attn_result[i];
                }
            }
            float cnt = 0;
            int cnt_0 = 0;
            for (int i = 0; i<768*query_s; i++) {
                if(host_result[i] != 0) cnt += abs((abs(host_result[i] - ptr_attn_result[i])/host_result[i]));
                else cnt_0++;
            }
            cout << cnt << endl;
            cnt /= (768*query_s-cnt_0);
            cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;
            cout << "max_err_host: " << maxe_host << " max_err_ptr: " << maxe_ptr << " diff: "  << max_err << endl;
        }
        cout << "=======\n";
        float cnt = 0;
        int cnt_0 = 0;
        for (int i = 0; i < 768 * query_s; i++) {
            if (isnan(host_result[i]) || isnan(ptr_attn_result[i])) {
                cout << "NaN detected at index " << i << " " << host_result[i] << " " << ptr_attn_result[i] << endl;
            }
            if (host_result[i] != 0) {
                cnt += abs((abs(host_result[i] - ptr_attn_result[i]) / host_result[i]));
            } else {
                cnt_0++;
            }
        }
        cout << cnt << endl;
        cnt /= (768*query_s-cnt_0);
        cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;

        if(iter != 0) {
            float host_kv[768432];
            
            cout << "===s: " << s << "===" << endl;
            dataPrepare(host_kv, 768*(s+1), 13, -1, iter);
            cout << "=======K_CACHE=======" << endl;
            for (int i = 0; i < 768*(s+1); i++) {
                if(abs(host_kv[i] - k_cache[0][i]) > 0.001)
                    cout << i << " " << k_cache[0][i] << " " << host_kv[i] << endl;
            }
            cnt = 0;
            cnt_0 = 0;
            for (int i = 0; i<768*(s+1); i++) {
                if(host_kv[i] != 0) cnt += abs((abs(host_kv[i] - k_cache[0][i])/host_kv[i]));
                else cnt_0++;
            }
            cnt /= (768*(s+1)-cnt_0);
            cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;

            dataPrepare(host_kv, 768*(s+1), 14, -1, iter);
            cout << "=======V_CACHE=======" << endl;
            for (int i = 0; i < 768*(s+1); i++) {
                if(abs(host_kv[i] - v_cache[0][i]) > 0.001)
                    cout << i << " " << v_cache[0][i] << " " << host_kv[i] << endl;
            }
            cnt = 0;
            cnt_0 = 0;
            for (int i = 0; i<768*(s+1); i++) {
                if(host_kv[i] != 0) cnt += abs((abs(host_kv[i] - v_cache[0][i])/host_kv[i]));
                else cnt_0++;
            }
            cnt /= (768*(s+1)-cnt_0);
            cout << "cnt_0: " << cnt_0 << " err: " << cnt << endl;
        }
    }
    
    
    // ============================================================================
    // Step 8: Release Allocated Resources
    // ============================================================================
    // 同樣需要注意data數量及argument名稱
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_x, ptr_Data_x));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_q_qweight, ptr_Data_w_tiling_q_qweight));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_q_zeros, ptr_Data_w_tiling_q_zeros));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_q_scale, ptr_Data_w_tiling_q_scale));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_k_qweight, ptr_Data_w_tiling_k_qweight));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_k_zeros, ptr_Data_w_tiling_k_zeros));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_k_scale, ptr_Data_w_tiling_k_scale));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_v_qweight, ptr_Data_w_tiling_v_qweight));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_v_zeros, ptr_Data_w_tiling_v_zeros));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_w_tiling_v_scale, ptr_Data_w_tiling_v_scale));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_bias_tiling_q, ptr_Data_bias_tiling_q));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_bias_tiling_k, ptr_Data_bias_tiling_k));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_bias_tiling_v, ptr_Data_bias_tiling_v));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_k_tiling, ptr_Data_k_tiling));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_v_tiling, ptr_Data_v_tiling));

    // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_qkv_result, ptr_qkv_result));
    // OCL_CHECK(err, err = q.finish());

    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_qkv_result, ptr_qkv_result));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_c_proj_qweight, ptr_Data_c_proj_qweight));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_c_proj_zeros, ptr_Data_c_proj_zeros));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_c_proj_scale, ptr_Data_c_proj_scale));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_Data_c_proj_bias, ptr_Data_c_proj_bias));
    OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_attn_result, ptr_attn_result));

    OCL_CHECK(err, err = q.finish());
    cout << "HOST-Info: DONE" << endl << endl;

//    return (error_detected ? EXIT_FAILURE : EXIT_SUCCESS);
    return 0;
}
