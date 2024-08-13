// Includes
#include <stdint.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_fixed.h>

// TRIPCOUNT identifier
//const int DEPTH = 1;
//const int HEIGHT = 256;
//const int NX = 384;
//const int NF = 384;
//const int DEPTH_HEIGHT = 256;
//
//const int DATA_SIZE_X = 98304;
//const int DATA_SIZE_WEIGHT = 147456;
//const int DATA_SIZE_BIAS = 384;
//const int DATA_SIZE_RES = 98304;
#define DEPTH 1
#define HEIGHT 512
#define NX 768
#define NF 768
#define DEPTH_HEIGHT 512

#define DATA_SIZE_X 393216
#define DATA_SIZE_WEIGHT 589824
#define DATA_SIZE_BIAS 768
#define DATA_SIZE_RES 393216
#define a_max DEPTH_HEIGHT*NX
#define b_max NX*NF
#define o_max DEPTH_HEIGHT*NF
//const int B1 = 128;
//const int B2 = 192;

//typedef ap_fixed<18, 9> half_t;

static void reshape(float* in, hls::stream<float>& outStream, int size) {
change_dim:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        outStream.write(in[i]);
    }
}

// static void reshape_weight(float* in, hls::stream<float>& outStream, int size) {
// change_dim:
//     for (int i = 0; i < size; i++) {
// #pragma HLS PIPELINE II=1
//         outStream.write(in[i]);
//     }
// }

static void compute_matmul(hls::stream<float>& xStream,
						   hls::stream<float>& weightStream,
						   hls::stream<float>& biasStream,
						   hls::stream<float>& outStream) {

//     int a_max = DEPTH_HEIGHT*NX;
//     int b_max = NX*NF;
//     int o_max = DEPTH_HEIGHT*NF;

     float a[a_max];
     float b[b_max];
     float conv_result[o_max];
     float bias[NF];
    //  float conv_tmp[NF];
    //  float b_tmp[NF];
#pragma HLS bind_storage variable=conv_result type=RAM_T2P impl=bram
#pragma HLS bind_storage variable=a type=RAM_T2P impl=uram
#pragma HLS bind_storage variable=b type=RAM_T2P impl=uram

//#pragma HLS ARRAY_RESHAPE variable=a type=cyclic factor=384
//#pragma HLS ARRAY_RESHAPE variable=b type=cyclic factor=384

     for (int i = 0; i < a_max; i++) {
 #pragma HLS PIPELINE II=1
 		a[i] = xStream.read();
 	}

     for (int i = 0; i < b_max; i++) {
 #pragma HLS PIPELINE II=1
 		b[i] = weightStream.read();
     }

     for (int i = 0; i < NF; i++) {
 #pragma HLS PIPELINE II=1
         bias[i] = biasStream.read();
     }


    for(int i = 0; i < DEPTH_HEIGHT; i++) {
        for(int j = 0; j < NF; j++) {
#pragma HLS PIPELINE II=1
        	conv_result[i * NF + j] = bias[j];
        }
    }

//     for(int i = 0; i < DEPTH_HEIGHT; i++) {
//         int iNF = i * NF;
//         for(int j = 0; j < NF; j++) {
// #pragma HLS PIPELINE II=1
// #pragma HLS UNROLL skip_exit_check factor=16
//             conv_tmp[j] = bias[j];
//         }
//         for(int k = 0; k < NX; k++) {
//             int kNF = k * NF;
//             float aik = a[i * NX + k];
//             for(int j = 0; j < NF; j++) {
// #pragma HLS PIPELINE II=1
// #pragma HLS UNROLL skip_exit_check factor=16
//                 b_tmp[j] = b[kNF + j];
//             }
//             for(int j = 0; j < NF; j++) {
// #pragma HLS PIPELINE II=1
// #pragma HLS UNROLL skip_exit_check factor=16
//             	conv_tmp[j] += aik * b_tmp[j];
//             }
//             for(int j = 0; j<NF; j++) {
// #pragma HLS PIPELINE II=1
// #pragma HLS UNROLL skip_exit_check factor=16
//                 conv_result[iNF + j] = conv_tmp[j];
//             }
//         }
//     }
        for(int i = 0; i < DEPTH_HEIGHT; i++) {
        int iNF = i * NF;
        for(int k = 0; k < NX; k++) {
            int kNF = k * NF;
            float aik = a[i * NX + k];
            for(int j = 0; j < NF; j++) {
#pragma HLS PIPELINE II=1
#pragma HLS UNROLL skip_exit_check factor=384
            	conv_result[iNF + j] += aik * b[kNF + j];
            }
        }
    }

     for (int i = 0; i < o_max; i++) {
 #pragma HLS PIPELINE II=1
 	   outStream.write(conv_result[i]);
     }
}

static void store_result(float* out, hls::stream<float>& inStream, int size) {
mem_wr:
    for (int i = 0; i < size; i++) {
#pragma HLS PIPELINE II=1
        out[i] = inStream.read();
    }
}

extern "C" {

void krnl_conv1D(float* x, float* weight, float* bias, float* out) {
#pragma HLS INTERFACE m_axi port = x offset = slave bundle = gmem0 depth = DATA_SIZE_X max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = weight offset = slave bundle = gmem1 depth = DATA_SIZE_WEIGHT max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = bias offset = slave bundle = gmem2 depth = DATA_SIZE_BIAS max_read_burst_length = 256
#pragma HLS INTERFACE m_axi port = out offset = slave bundle = gmem3 depth = DATA_SIZE_RES max_write_burst_length = 256
#pragma HLS INTERFACE s_axilite port = x
#pragma HLS INTERFACE s_axilite port = weight
#pragma HLS INTERFACE s_axilite port = bias
#pragma HLS INTERFACE s_axilite port = out
#pragma HLS INTERFACE s_axilite port = return

    hls::stream<float> xStream("xStream");
    hls::stream<float> weightStream("weightStream");
    hls::stream<float> biasStream("biasStream");
    hls::stream<float> outStream("outStream");

#pragma HLS DATAFLOW

     reshape(x, xStream, DATA_SIZE_X);
     reshape(weight, weightStream, DATA_SIZE_WEIGHT);

     for (int i = 0; i < DATA_SIZE_BIAS; i++) {
         biasStream.write(bias[i]);
     }

     compute_matmul(xStream, weightStream, biasStream, outStream);
     store_result(out, outStream, DATA_SIZE_RES);
}
}
