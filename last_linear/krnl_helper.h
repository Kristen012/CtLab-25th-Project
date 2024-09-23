#ifndef KRNL_HELPER_H
#define KRNL_HELPER_H

#include <stdint.h>
#include <hls_math.h>
#include <ap_int.h>
#include <ap_fixed.h>

#define K0 768
#define N0 426
#define MAX_SIZE (K0 * N0)
#define MAX_DEPTH 32

// 定义 fp24 格式，10 位整数，14 位小数
typedef ap_fixed<24, 10> fp24_t;

// 打包三组 fp24 数据
typedef ap_uint<72> packed_fp24_t;
typedef half half_t;

// Pack three fp24 values into one 72-bit integer
inline packed_fp24_t pack_fp24(fp24_t a, fp24_t b, fp24_t c) {
    #pragma HLS INLINE off
    packed_fp24_t packed;
    ap_uint<24> a_bits = *(ap_uint<24>*)&a;
    ap_uint<24> b_bits = *(ap_uint<24>*)&b;
    ap_uint<24> c_bits = *(ap_uint<24>*)&c;
    packed.range(23, 0) = a_bits;
    packed.range(47, 24) = b_bits;
    packed.range(71, 48) = c_bits;
    return packed;
}

// Unpack three fp24 values from one 72-bit integer
inline void unpack_fp24(packed_fp24_t packed, fp24_t& a, fp24_t& b, fp24_t& c) {
    #pragma HLS INLINE off
    ap_uint<24> a_bits = packed.range(23, 0);
    ap_uint<24> b_bits = packed.range(47, 24);
    ap_uint<24> c_bits = packed.range(71, 48);
    a = *(fp24_t*)&a_bits;
    b = *(fp24_t*)&b_bits;
    c = *(fp24_t*)&c_bits;
}

#endif // KRNL_HELPER_H
