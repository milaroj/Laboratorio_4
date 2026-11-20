#ifndef SOFTMAX_HLS_HPP
#define SOFTMAX_HLS_HPP

#include <hls_half.h>

using data_t = half; // FP16

extern "C" void softmax_hls(
    const data_t x[1000],
    data_t y[1000],
    int n
);

#endif