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
static const int DATA_SIZE_X_fc = 128*768; // TODO
static const int DATA_SIZE_WEIGHT_fc = 768*(258); // TODO
static const int DATA_SIZE_BIAS_fc = 258; // TODO
static const int DATA_SIZE_RES_fc = 128*(258); // TODO
static const int DATA_SIZE_MAX_DEPTH_fc = 32;

static const int DATA_SIZE_X_proj = 128*3072; // TODO
static const int DATA_SIZE_WEIGHT_proj = 3072*(258); // TODO
static const int DATA_SIZE_BIAS_proj = 258; // TODO
static const int DATA_SIZE_RES_proj = 128*(258); // TODO
static const int DATA_SIZE_MAX_DEPTH_proj = 32;

#define fc_K0 768
#define fc_N0 258
#define fc_K1 1
#define fc_N1 12

#define proj_K0 3072
#define proj_N0 258
#define proj_K1 1
#define proj_N1 3

// Compute the size of array in bytes
size_t size_in_bytes_x_fc = DATA_SIZE_X_fc * sizeof(float);
size_t size_in_bytes_weight_fc = DATA_SIZE_WEIGHT_fc * sizeof(float);
size_t size_in_bytes_bias_fc = DATA_SIZE_BIAS_fc * sizeof(float);
size_t size_in_bytes_res_fc = DATA_SIZE_RES_fc * sizeof(float);

size_t size_in_bytes_x_proj = DATA_SIZE_X_proj * sizeof(float);
size_t size_in_bytes_weight_proj = DATA_SIZE_WEIGHT_proj * sizeof(float);
size_t size_in_bytes_bias_proj = DATA_SIZE_BIAS_proj * sizeof(float);
size_t size_in_bytes_res_proj = DATA_SIZE_RES_proj * sizeof(float);

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
    cl::Kernel krnl_MLP_fc_linear;
    cl::Kernel krnl_MLP_proj_linear;
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: krnl_MLP_fc_linear ..." << endl;
	cout << "HOST-Info: Creating a Kernel: krnl_MLP_proj_linear ..." << endl;
	#endif
    OCL_CHECK(err, krnl_MLP_fc_linear = cl::Kernel(program, "krnl_MLP_fc_linear", &err));
    OCL_CHECK(err, krnl_MLP_proj_linear = cl::Kernel(program, "krnl_MLP_proj_linear", &err));

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
    float* ptr_DataIn_1; // x
    float* ptr_DataIn_2; // weight
    float* ptr_DataIn_3; // bias
    float* ptr_result;   // result
    int DEPTH = 127;
    float* X = new float[DEPTH * fc_K0 + 5];
    float* W = new float[fc_K0 * fc_WEIGHT_WIDTH + 5];
    float* B = new float[fc_WEIGHT_WIDTH + 5];
    float* RES = new float [DEPTH * fc_WEIGHT_WIDTH + 5];

    dataPrepare(X, DEPTH*fc_K0, 1);
    dataPrepare(W, fc_K0*fc_WEIGHT_WIDTH, 2);
    dataPrepare(B, fc_WEIGHT_WIDTH, 3);

    OCL_CHECK(err, cl::Buffer buffer_DataIn_1(context, CL_MEM_READ_ONLY, size_in_bytes_x_fc, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_DataIn_2(context, CL_MEM_READ_ONLY, size_in_bytes_weight_fc, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_DataIn_3(context, CL_MEM_READ_ONLY, size_in_bytes_bias_fc, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result(context, CL_MEM_READ_WRITE, size_in_bytes_res_fc, NULL, &err));

    OCL_CHECK(err, err = krnl_MLP_fc_linear.setArg(0, buffer_DataIn_1));
    OCL_CHECK(err, err = krnl_MLP_fc_linear.setArg(1, buffer_DataIn_2));
    OCL_CHECK(err, err = krnl_MLP_fc_linear.setArg(2, buffer_DataIn_3));
    OCL_CHECK(err, err = krnl_MLP_fc_linear.setArg(3, buffer_result));


    float* host_result = new float [DEPTH*fc_WEIGHT_WIDTH + 5];
    dataPrepare(host_result, DEPTH*3072, 4);
    int error_flag = 0;
    int depth_size = 0;
    for (int i = 0; i < DEPTH; i += DATA_SIZE_MAX_DEPTH_fc) {
        if (DEPTH - i < DATA_SIZE_MAX_DEPTH_fc) {
            depth_size = DEPTH - i;
        }
        else {
            depth_size = DATA_SIZE_MAX_DEPTH_fc;
        }
        OCL_CHECK(err, err = krnl_MLP_fc_linear.setArg(4, depth_size));
        for (int j = 0; j < fc_N1; j++) {
        	cout << "i: " << i << ", j: " << j << endl;

            OCL_CHECK(err, ptr_DataIn_1 = (float*)q.enqueueMapBuffer(buffer_DataIn_1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_x_fc, NULL, NULL, &err));
            dataPrepare_tiling_fc(ptr_DataIn_1, X, depth_size * fc_K0, 1, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_1, ptr_DataIn_1));

            OCL_CHECK(err, ptr_DataIn_2 = (float*)q.enqueueMapBuffer(buffer_DataIn_2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_weight_fc, NULL, NULL, &err));
            dataPrepare_tiling_fc(ptr_DataIn_2, W, fc_K0*fc_N0, 2, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_2, ptr_DataIn_2));
            
            OCL_CHECK(err, ptr_DataIn_3 = (float*)q.enqueueMapBuffer(buffer_DataIn_3, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_bias_fc, NULL, NULL, &err));
            dataPrepare_tiling_fc(ptr_DataIn_3, B, fc_N0, 3, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_3, ptr_DataIn_3));

            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_DataIn_1, buffer_DataIn_2, buffer_DataIn_3}, 0)); // Migrate new data to the device
        
            OCL_CHECK(err, err = q.enqueueTask(krnl_MLP_fc_linear));
            OCL_CHECK(err, err = q.finish());

            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, err = q.finish());

            OCL_CHECK(err, ptr_result = (float*)q.enqueueMapBuffer(buffer_result, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_res_fc, NULL, NULL, &err));
            for (int d = 0; d < depth_size; d++) {
                for (int k = 0; k < fc_N0; k++) {
                	if (fc_N0 - k == 1 || fc_N0 - k == 2) {
                		continue;
                	}
                    RES[i * 3072 + j * 256 + d * 3072 + k] = ptr_result[d * fc_N0 + k];
                    if (abs(ptr_result[d * fc_N0 + k] - host_result[i * 3072 + j * 256 + d * 3072 + k]) > 0.01) {
                        cout << "i: " << i << ", j: " << j << ", k: " << k << ", kernel: " <<  ptr_result[d * fc_N0 + k] << ", host: " << host_result[i * 3072 + j * 256 + d * 3072 + k] << endl;
                        error_flag = 1;
                        break;
                    }
                }
                if (error_flag) break;
            }
            if (error_flag) break;
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_result, ptr_result));
        }
        if (error_flag) break;
    }

    float* ptr_DataIn_4; // RES
    float* ptr_DataIn_5; // weight_2
    float* ptr_DataIn_6; // bias_2
    float* ptr_result_final;   // result_2
    float* X_proj = new float[DEPTH * proj_K0 + 5];
    float* W_proj = new float[proj_K0 * proj_WEIGHT_WIDTH + 5];
    float* B_proj = new float[proj_WEIGHT_WIDTH + 5];
    float* RES_final = new float [DEPTH * proj_WEIGHT_WIDTH + 5];

     memcpy(X_proj, RES, DEPTH * proj_K0 * sizeof(float));
//    dataPrepare(X_proj, proj_K0*DEPTH, 8);
    dataPrepare(W_proj, proj_K0*proj_WEIGHT_WIDTH, 5);
    dataPrepare(B_proj, proj_WEIGHT_WIDTH, 6);

    // for (int i = 0; i < DEPTH * proj_K0; i++) {
    //     if (X_proj[i] != RES[i]) {
    //         cout << "ERROR_memcpy!!" << endl;
    //         break;
    //     }
    // }

    OCL_CHECK(err, cl::Buffer buffer_DataIn_4(context, CL_MEM_READ_ONLY, size_in_bytes_x_proj, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_DataIn_5(context, CL_MEM_READ_ONLY, size_in_bytes_weight_proj, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_DataIn_6(context, CL_MEM_READ_ONLY, size_in_bytes_bias_proj, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result_final(context, CL_MEM_WRITE_ONLY, size_in_bytes_res_proj, NULL, &err));

    OCL_CHECK(err, err = krnl_MLP_proj_linear.setArg(0, buffer_DataIn_4));
    OCL_CHECK(err, err = krnl_MLP_proj_linear.setArg(1, buffer_DataIn_5));
    OCL_CHECK(err, err = krnl_MLP_proj_linear.setArg(2, buffer_DataIn_6));
    OCL_CHECK(err, err = krnl_MLP_proj_linear.setArg(3, buffer_result_final));
    
    dataPrepare(host_result, DEPTH*768, 7);

    for (int i = 0; i < DEPTH; i += DATA_SIZE_MAX_DEPTH_proj) {
        if (DEPTH - i < DATA_SIZE_MAX_DEPTH_proj) {
            depth_size = DEPTH - i;
        }
        else {
            depth_size = DATA_SIZE_MAX_DEPTH_proj;
        }
        OCL_CHECK(err, err = krnl_MLP_proj_linear.setArg(4, depth_size));
        for (int j = 0; j < proj_N1; j++) {
        	cout << "i: " << i << ", j: " << j << endl;

            OCL_CHECK(err, ptr_DataIn_4 = (float*)q.enqueueMapBuffer(buffer_DataIn_4, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_x_proj, NULL, NULL, &err));
            dataPrepare_tiling_proj(ptr_DataIn_4, X_proj, depth_size * proj_K0, 1, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_4, ptr_DataIn_4));

            OCL_CHECK(err, ptr_DataIn_5 = (float*)q.enqueueMapBuffer(buffer_DataIn_5, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_weight_proj, NULL, NULL, &err));
            dataPrepare_tiling_proj(ptr_DataIn_5, W_proj, proj_K0*proj_N0, 2, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_5, ptr_DataIn_5));
            
            OCL_CHECK(err, ptr_DataIn_6 = (float*)q.enqueueMapBuffer(buffer_DataIn_6, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_bias_proj, NULL, NULL, &err));
            dataPrepare_tiling_proj(ptr_DataIn_6, B_proj, proj_N0, 3, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_6, ptr_DataIn_6));

            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_DataIn_4, buffer_DataIn_5, buffer_DataIn_6}, 0)); // Migrate new data to the device
        
            OCL_CHECK(err, err = q.enqueueTask(krnl_MLP_proj_linear));
            OCL_CHECK(err, err = q.finish());

            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_result_final}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, err = q.finish());

            OCL_CHECK(err, ptr_result_final = (float*)q.enqueueMapBuffer(buffer_result_final, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_res_proj, NULL, NULL, &err));
            for (int d = 0; d < depth_size; d++) {
                for (int k = 0; k < proj_N0; k++) {
                	if (proj_N0 - k == 1 || proj_N0 - k == 2) {
                		continue;
                	}
                    RES_final[i * 768 + j * 256 + d * 768 + k] = ptr_result_final[d * proj_N0 + k];
                    if (abs(ptr_result_final[d * proj_N0 + k] - host_result[i * 768 + j * 256 + d * 768 + k]) > 0.1) {
                        cout << "i: " << i << ", j: " << j << ", k: " << k << ", kernel: " <<  ptr_result_final[d * proj_N0 + k] << ", host: " << host_result[i * 768 + j * 256 + d * 768 + k] << endl;
                        error_flag = 1;
                        break;
                    }
                }
                if (error_flag) break;
            }
            if (error_flag) break;
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_result_final, ptr_result_final));
        }
        if (error_flag) break;
    }

    delete [] X;
    delete [] W;
    delete [] B;
    delete [] X_proj;
    delete [] W_proj;
    delete [] B_proj;


    // ============================================================================
	// Step 5: Set Kernel Arguments and Run the Application
	//         o) Set Kernel Arguments
	// 				----------------------------------------------------
	// 				 Kernel	  		Argument Nb		Description
	// 				----------------------------------------------------
	//  			 krnl_conv1D	    0				GlobMem_BUF_DataIn_1
	//  			 krnl_conv1D	    1				GlobMem_BUF_DataIn_2
	//  			 krnl_conv1D	    2				GlobMem_BUF_RES
	//  			 krnl_conv1D	    3				CONST_arg
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
    bool error_detected = false;
    float error_rate = 0;
    float max_error = 0;  // 記錄最大誤差
    int max_error_index = -1; // 記錄最大誤差的索引
    float max_host_result = 0;  // 記錄最大誤差時的 host_result 值
    float max_res_final = 0;    // 記錄最大誤差時的 RES_final 值

    for (int i = 0; i < DEPTH*768; i++) {
        float error = 0;

        if (abs(host_result[i]) > 1e-7) {
            error = abs(RES_final[i] - host_result[i]) / abs(host_result[i]);
        } else {
            error = abs(RES_final[i] - host_result[i]);
        }

        error_rate += error;

        // 更新最大誤差
        if (error > max_error) {
            max_error = error;
            max_error_index = i;  // 記錄最大誤差所在的索引
            max_host_result = host_result[i];  // 記錄此時的 host_result 值
            max_res_final = RES_final[i];      // 記錄此時的 RES_final 值
        }
    }

    // 印出最大誤差及其對應的索引和結果值
    cout << "Max error: " << max_error << " at index: " << max_error_index << endl;
    cout << "Host result at max error: " << max_host_result << endl;
    cout << "RES_final at max error: " << max_res_final << endl;
    cout << "Overall error rate: " << error_rate << endl;

    cout << "error rate: " << error_rate << endl;
    delete [] host_result;
    delete [] RES;
    delete [] RES_final;
    // ============================================================================
	// Step 7: Custom Profiling
	// ============================================================================
	// cout << "HOST-Info: ============================================================= " << endl;
	// cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
	// cout << "HOST-Info: ============================================================= " << endl;

	// int Nb_Of_Kernels = 1;
	// int Nb_Of_Memory_Tranfers = Nb_Of_Mem_Events;

	// string list_of_kernel_names[Nb_Of_Kernels];
	// list_of_kernel_names[0]="krnl_MLP_fc_linear";
	// run_custom_profiling (Nb_Of_Kernels,Nb_Of_Memory_Tranfers,K_exe_event,Mem_op_event,list_of_kernel_names);
    
    // ============================================================================
	// Step 8: Release Allocated Resources
	// ============================================================================
    // 同樣需要注意data數量及argument名稱
    // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_DataIn_1, ptr_DataIn_1));
    // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_DataIn_2, ptr_DataIn_2));
    // OCL_CHECK(err, err = q.enqueueUnmapMemObject(buffer_result, ptr_result));
    // OCL_CHECK(err, err = q.finish());
    
    cout << "HOST-Info: TEST " << (error_detected ? "FAILED" : "PASSED") << endl;
    cout << "HOST-Info: DONE" << endl << endl;

    return (error_detected ? EXIT_FAILURE : EXIT_SUCCESS);
}
