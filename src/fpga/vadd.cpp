// Includes
#include <stdint.h>
#include <iostream>
#include <hls_stream.h>

static void load_input(unsigned char* in1,
                       hls::stream<unsigned char>& in_stream,
                       size_t size) {
    std::cout << "Started load_input" << std::endl;
    mem_rd:
    for (size_t i = 0; i < size; i += 3) {
        #pragma HLS pipeline II=1
        in_stream << in1[i];    //b
        in_stream << in1[i+1];  //g
        in_stream << in1[i+2];  //r
    }
    std::cout << "Completed load_input" << std::endl;
}

static void compute_add(hls::stream<unsigned char>& in_stream,
                        hls::stream<unsigned char>& out_stream,
                        size_t size) {
    std::cout << "Started compute_add" << std::endl;
    execute:
    for (size_t i = 0; i < size; i += 3) {
        #pragma HLS pipeline II=1
        unsigned char val1 = in_stream.read();
        unsigned char val2 = in_stream.read();
        unsigned char val3 = in_stream.read();
        unsigned char avg = (unsigned char)(val1 * 0.0722 + val2 * 0.7152 + val3 * 0.2126);
        out_stream << avg;
        out_stream << avg;
        out_stream << avg;
    }
    std::cout << "Completed compute_add" << std::endl;
}

static void store_result(unsigned char* out,
                         hls::stream<unsigned char>& out_stream,
                         size_t size) {
    std::cout << "Started store_result" << std::endl;
    mem_wr:
    for (size_t i = 0; i < size; i += 3) {
        #pragma HLS pipeline II=1
        out[i] = out_stream.read();
        out[i+1] = out_stream.read();
        out[i+2] = out_stream.read();
    }

    // Consume any remaining data in the out_stream
    while (!out_stream.empty()) {
        out_stream.read();
    }

    std::cout << "Completed store_result" << std::endl;
}

extern "C" {

void vadd(unsigned char* in1, unsigned char* out, size_t size) {
#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = out bundle = gmem0

    static hls::stream<unsigned char> in_stream("Input_stream");
    static hls::stream<unsigned char> out_stream("Output_stream");

#pragma HLS stream variable=in_stream depth=4096
#pragma HLS stream variable=out_stream depth=4096
    // dataflow pragma instruct compiler to run following three APIs in parallel
#pragma HLS dataflow
    load_input(in1, in_stream, size);
    compute_add(in_stream, out_stream, size);
    store_result(out, out_stream, size);
}
}