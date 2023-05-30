// Includes
#include <stdint.h>
#include <iostream>
#include <hls_stream.h>

static void load_input(uint32_t* in1,
                        hls::stream<uint32_t>& in_stream,
                        size_t size) {
mem_rd:
#pragma HLS loop pipeline
    for (size_t i = 0; i < size; i += 3) {
        in_stream << in1[i];
        in_stream << in1[i+1];
        in_stream << in1[i+2];
        //std::cout << in1[i] << " " << in1[i+1] << " " << in1[i+2] << std::endl;
    }
}

static void compute_add(hls::stream<uint32_t>& in_stream,
                        hls::stream<uint32_t>& out_stream,
                        size_t size) {
// The kernel is operating with vector of NUM_WORDS integers. The + operator performs
// an element-wise add, resulting in NUM_WORDS parallel additions.
execute:
#pragma HLS loop pipeline
    for (size_t i = 0; i < size; i += 3) {
        unsigned char avg = (in_stream.read() * 0.0722 + in_stream.read() * 0.7152 + in_stream.read() * 0.2126);
        out_stream << avg;
        out_stream << avg;
        out_stream << avg;
        //std::cout << in_stream.read() << " " << in_stream.read() << " " << in_stream.read() << std::endl;
    }
}

static void store_result(uint32_t* out, 
                        hls::stream<uint32_t>& out_stream, 
                        size_t size) {
mem_wr:
#pragma HLS loop pipeline
    for (size_t i = 0; i < size; i += 3) {
        out[i] = out_stream.read();
        out[i+1] = out_stream.read();
        out[i+2] = out_stream.read();
    }
}

extern "C" {

void vadd(uint32_t* in1, uint32_t* out, size_t size) {
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = out bundle = gmem0

    static hls::stream<uint32_t> in_stream("Input_stream");
    static hls::stream<uint32_t> out_stream("Output_stream");

    // dataflow pragma instruct compiler to run following three APIs in parallel
#pragma HLS dataflow
    load_input(in1, in_stream, size);
    compute_add(in_stream, out_stream, size);
    store_result(out, out_stream, size);
}
}