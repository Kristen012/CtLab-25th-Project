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
static const int DATA_SIZE_X = 128*768; // TODO
static const int DATA_SIZE_WEIGHT = 768*(2*3*71); // TODO
static const int DATA_SIZE_RES = 128*(2*3*71); // TODO
static const int DATA_SIZE_MAX_DEPTH = 32;

#define K0 768
#define N0 426
#define K1 1
#define N1 118


// Compute the size of array in bytes
size_t size_in_bytes_x = DATA_SIZE_X * sizeof(float);
size_t size_in_bytes_weight = DATA_SIZE_WEIGHT * sizeof(float);
size_t size_in_bytes_res = DATA_SIZE_RES * sizeof(float);


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
    cl::Kernel krnl_linear_head1;
    cl::Kernel krnl_linear_head2;
    #ifdef ALL_MESSAGES
	cout << "HOST-Info: Creating a Kernel: krnl_linear ..." << endl;
	#endif
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
    /* 此處可以定義input和output的型別 */

    float* ptr_DataIn_head1_1; // x
    float* ptr_DataIn_head1_2; // weight
    float* ptr_result_head1;   // result

    float* ptr_DataIn_head2_1; // x
    float* ptr_DataIn_head2_2; // weight
    float* ptr_result_head2;   // result
    int DEPTH = 127;

    float* X = new float[DEPTH * K0 + 5];
    float* W = new float[K0 * WEIGHT_WIDTH + 5];
    float* RES = new float [DEPTH * WEIGHT_WIDTH + 5];

    cout << "checkpoint1" << endl;

    dataPrepare(X, DEPTH*K0, 1);
    dataPrepare(W, K0*WEIGHT_WIDTH, 2);

    cout << "checkpoint2" << endl;

    // set up buffer
    OCL_CHECK(err, cl::Buffer buffer_DataIn_head1_1(context, CL_MEM_READ_ONLY, size_in_bytes_x, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_DataIn_head1_2(context, CL_MEM_READ_ONLY, size_in_bytes_weight, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result_head1(context, CL_MEM_WRITE_ONLY, size_in_bytes_res, NULL, &err));

    OCL_CHECK(err, cl::Buffer buffer_DataIn_head2_1(context, CL_MEM_READ_ONLY, size_in_bytes_x, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_DataIn_head2_2(context, CL_MEM_READ_ONLY, size_in_bytes_weight, NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_result_head2(context, CL_MEM_WRITE_ONLY, size_in_bytes_res, NULL, &err));

    // set up kernel arguments
    OCL_CHECK(err, err = krnl_linear_head1.setArg(0, buffer_DataIn_head1_1));
    OCL_CHECK(err, err = krnl_linear_head1.setArg(1, buffer_DataIn_head1_2));
    OCL_CHECK(err, err = krnl_linear_head1.setArg(2, buffer_result_head1));

    OCL_CHECK(err, err = krnl_linear_head2.setArg(0, buffer_DataIn_head2_1));
    OCL_CHECK(err, err = krnl_linear_head2.setArg(1, buffer_DataIn_head2_2));
    OCL_CHECK(err, err = krnl_linear_head2.setArg(2, buffer_result_head2));

    cout << "checkpoint3" << endl;

    cl::Event event1, event2;

    // check the answer
    float* host_result = new float [DEPTH*WEIGHT_WIDTH + 5];
    dataPrepare(host_result, DEPTH*50257, 3);
    int error_flag = 0;
    int depth_size = 0;

    cl_ulong time_start1, time_end1;
    cl_ulong time_start2, time_end2;
    double total_execution_time_kernel1 = 0.0;
    double total_execution_time_kernel2 = 0.0;
    double total_overlap_time = 0.0;

    for (int i = 0; i < DEPTH; i += DATA_SIZE_MAX_DEPTH) {
        // cheak depth size
        if (DEPTH - i < DATA_SIZE_MAX_DEPTH) {
            depth_size = DEPTH - i;
        }
        else {
            depth_size = DATA_SIZE_MAX_DEPTH;
        }
        OCL_CHECK(err, err = krnl_linear_head1.setArg(3, depth_size));
        OCL_CHECK(err, err = krnl_linear_head2.setArg(3, depth_size));
        for (int j = 0; j < N1; j += 2) {
//        	cout << "i: " << i << ", j: " << j << endl;

            // head 1 enqueue map
            OCL_CHECK(err, ptr_DataIn_head1_1 = (float*)q.enqueueMapBuffer(buffer_DataIn_head1_1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_x, NULL, NULL, &err));
            dataPrepare_tiling(ptr_DataIn_head1_1, X, depth_size * K0, 1, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_head1_1, ptr_DataIn_head1_1));

            OCL_CHECK(err, ptr_DataIn_head1_2 = (float*)q.enqueueMapBuffer(buffer_DataIn_head1_2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_weight, NULL, NULL, &err));
            dataPrepare_tiling(ptr_DataIn_head1_2, W, K0*N0, 2, j, i);
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_DataIn_head1_2, ptr_DataIn_head1_2));

            // head 2 enqueue map
            OCL_CHECK(err, ptr_DataIn_head2_1 = (float*)q2.enqueueMapBuffer(buffer_DataIn_head2_1, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_x, NULL, NULL, &err));
            dataPrepare_tiling(ptr_DataIn_head2_1, X, depth_size * K0, 1, j, i);
            OCL_CHECK(err, q2.enqueueUnmapMemObject(buffer_DataIn_head2_1, ptr_DataIn_head2_1));

            OCL_CHECK(err, ptr_DataIn_head2_2 = (float*)q2.enqueueMapBuffer(buffer_DataIn_head2_2, CL_TRUE, CL_MAP_WRITE, 0, size_in_bytes_weight, NULL, NULL, &err));
            dataPrepare_tiling(ptr_DataIn_head2_2, W, K0*N0, 2, j+1, i);
            OCL_CHECK(err, q2.enqueueUnmapMemObject(buffer_DataIn_head2_2, ptr_DataIn_head2_2));

            // Migrate the data from host to kernel
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_DataIn_head1_1, buffer_DataIn_head1_2}, 0));
            OCL_CHECK(err, err = q2.enqueueMigrateMemObjects({buffer_DataIn_head2_1, buffer_DataIn_head2_2}, 0));

            // Execute both head
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

            // Get result
            OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_result_head1}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, err = q2.enqueueMigrateMemObjects({buffer_result_head2}, CL_MIGRATE_MEM_OBJECT_HOST));
            OCL_CHECK(err, err = q.finish());
            OCL_CHECK(err, err = q2.finish());

            OCL_CHECK(err, ptr_result_head1 = (float*)q.enqueueMapBuffer(buffer_result_head1, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_res, NULL, NULL, &err));
            OCL_CHECK(err, ptr_result_head2 = (float*)q2.enqueueMapBuffer(buffer_result_head2, CL_TRUE, CL_MAP_READ, 0, size_in_bytes_res, NULL, NULL, &err));

            for (int h = 0; h < 2; h++) {
                for (int d = 0; d < depth_size; d++) {
                    for (int k = 0; k < N0; k++) {
                        if (j+h == N1 - 1 && (N0 - k <= 11)) {
                            continue;
                        }
                        if (h == 0) {
                            RES[i * 50257 + j * N0 + d * 50257 + k] = ptr_result_head1[d * N0 + k];
                            if (abs(ptr_result_head1[d * N0 + k] - host_result[i * 50257 + j * N0 + d * 50257 + k]) > 0.5) {
                                cout << "i: " << i << ", j: " << j << ", k: " << k << ", kernel: " <<  ptr_result_head1[d * N0 + k] << ", host: " << host_result[i * 50257 + j * N0 + d * 50257 + k] << ", err: " << abs(ptr_result_head1[d * N0 + k] - host_result[i * 50257 + j * N0 + d * 50257 + k]) << endl;
                                error_flag = 1;
                                break;
                            }
                        }
                        else if (h == 1) {
                            RES[i * 50257 + (j + 1) * N0 + d * 50257 + k] = ptr_result_head2[d * N0 + k];
                            if (abs(ptr_result_head2[d * N0 + k] - host_result[i * 50257 + (j + 1) * N0 + d * 50257 + k]) > 0.5) {
                                cout << "i: " << i << ", j: " << (j + 1) << ", k: " << k << ", kernel: " <<  ptr_result_head2[d * N0 + k] << ", host: " << host_result[i * 50257 + (j + 1) * N0 + d * 50257 + k] << ", err: " << abs(ptr_result_head2[d * N0 + k] - host_result[i * 50257 + (j + 1) * N0 + d * 50257 + k]) << endl;
                                error_flag = 1;
                                break;
                            }
                        }

                    }
                    if (error_flag) break;
                }
                if (error_flag) break;
            }
            if (error_flag) break;
            OCL_CHECK(err, q.enqueueUnmapMemObject(buffer_result_head1, ptr_result_head1));
            OCL_CHECK(err, q2.enqueueUnmapMemObject(buffer_result_head2, ptr_result_head2));
        }
        if (error_flag) break;
    }


    double parallel_execution_time = total_execution_time_kernel1 + total_execution_time_kernel2 - total_overlap_time;
    cout << "Total Kernel 1 execution time: " << total_execution_time_kernel1 << " ms" << endl;
    cout << "Total Kernel 2 execution time: " << total_execution_time_kernel2 << " ms" << endl;
    cout << "Total Overlap time: " << total_overlap_time << " ms" << endl;
    cout << "Parallel execution time: " << parallel_execution_time << " ms" << endl;

    // event1.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start1);
    // event1.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end1);

    // event2.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start2);
    // event2.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end2);

    // double execution_time_kernel1 = (time_end1 - time_start1) * 1.0e-6; // Convert to ms
    // double execution_time_kernel2 = (time_end2 - time_start2) * 1.0e-6; // Convert to ms

    // cout << "Kernel 1 execution time: " << execution_time_kernel1 << " ms" << endl;
    // cout << "Kernel 2 execution time: " << execution_time_kernel2 << " ms" << endl;
    // cout << "Kernel 1 begin time: " << time_start1 * 1.0e-6 << ",  Kernel 1 end time: " << time_end1 * 1.0e-6 << endl;
    // cout << "Kernel 2 begin time: " << time_start2 * 1.0e-6 << ",  Kernel 2 end time: " << time_end2 * 1.0e-6 << endl;

    
    delete [] X;
    delete [] W;


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
//    float* host_result = new float [DEPTH*WEIGHT_WIDTH + 5];
    float error_rate = 0;
//    dataPrepare(host_result, DEPTH*WEIGHT_WIDTH, 3);
    for (int i = 0; i < DEPTH*50257; i++) {
//    	cout << "index: " << i << ", host: " << host_result[i] << ", kernel: " << RES[i] << endl;
    	error_rate += abs(RES[i] - host_result[i]) / abs(host_result[i]);
//    	if (abs(RES[i] - host_result[i]) > 10) {
//            cout << host_result[i] << " " << RES[i] << endl;
//            printf(error_message.c_str(), i, host_result[i], RES[i]);
//            error_detected = true;
//            break;
//    	}
    }
    cout << "error rate: " << error_rate << endl;
    delete [] host_result;
    delete [] RES;
    // ============================================================================
	// Step 7: Custom Profiling
	// ============================================================================
	// cout << "HOST-Info: ============================================================= " << endl;
	// cout << "HOST-Info: (Step 7) Custom Profiling                                     " << endl;
	// cout << "HOST-Info: ============================================================= " << endl;

	// int Nb_Of_Kernels = 1;
	// int Nb_Of_Memory_Tranfers = Nb_Of_Mem_Events;

	// string list_of_kernel_names[Nb_Of_Kernels];
	// list_of_kernel_names[0]="krnl_linear";
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
