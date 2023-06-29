#define OCL_CHECK(error, call)                                                                   \
    call;                                                                                        \
    if (error != CL_SUCCESS)                                                                     \
    {                                                                                            \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
        exit(EXIT_FAILURE);                                                                      \
    }

static const int DATA_SIZE = 4096;

#include <vadd.h>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <typeinfo>
#include "PPM.h"

static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

int main(int argc, char *argv[])
{
    // TARGET_DEVICE macro needs to be passed from gcc command line
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " <xclbin>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string xclbinFilename = argv[1];
    std::cout << "Usage: " << argv[1] << " <xclbin>" << std::endl;
    std::string videofilename = "/home/vishnu/Documents/Xilinx_Test2/Image/sample2.mp4";
    std::string filepath = "/home/vishnu/Documents/Xilinx_Test2/Image/output.mp4";

    VideoDetails videoDetails;
    std::vector<cv::Mat> NewData;

    NewData = extractFrameData(videofilename, videoDetails);

    size_t totalSize = 0;
    size_t frameElementSize = NewData[0].elemSize(); // Get the element size of the frames

    for (const auto &frame : NewData)
    {
        totalSize += frame.total() * frameElementSize;
    }

    unsigned char *dataArray = new unsigned char[totalSize];
    unsigned char *output_arr = new unsigned char[totalSize];

    // Copy the data from each frame into the result array
    int currentIndex = 0;
    for (const auto &frame : NewData)
    {
        const unsigned char *frameData = frame.ptr();
        int frameSize = frame.total() * frameElementSize;
        int frameStride = frame.step; // Assuming each frame is stored as a single continuous row in memory

        for (int row = 0; row < frame.rows; row++)
        {
            const unsigned char *rowData = frameData + row * frameStride;
            std::memcpy(dataArray + currentIndex, rowData, frameSize / frame.rows);
            currentIndex += frameSize / frame.rows;
        }
    }

    std::vector<cv::Mat> HostGreyscale;
    HostGreyscale = ProcessPixelData(NewData, videoDetails);

    // Creates a vector of DATA_SIZE elements with an initial value of 10 and 32
    // using customized allocator for getting buffer alignment to 4k boundary

    std::vector<cl::Device> devices;
    cl_int err;
    cl::Context context;
    cl::CommandQueue q;
    cl::Kernel krnl_argument;
    cl::Program program;
    std::vector<cl::Platform> platforms;
    bool found_device = false;

    // traversing all Platforms To find Xilinx Platform and targeted
    // Device in Xilinx Platform
    cl::Platform::get(&platforms);
    for (size_t i = 0; (i < platforms.size()) & (found_device == false); i++)
    {
        cl::Platform platform = platforms[i];
        std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
        if (platformName == "Xilinx")
        {
            devices.clear();
            platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
            if (devices.size())
            {
                found_device = true;
                break;
            }
        }
    }
    if (found_device == false)
    {
        std::cout << "Error: Unable to find Target Device " << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "INFO: Reading " << xclbinFilename << std::endl;
    FILE *fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr)
    {
        printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
        exit(EXIT_FAILURE);
    }
    // Load xclbin
    std::cout << "Loading: '" << xclbinFilename << "'\n";
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg(0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg(0, bin_file.beg);
    char *buf = new char[nb];
    bin_file.read(buf, nb);

    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf, nb});
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++)
    {
        auto device = devices[i];
        // Creating Context and Command Queue for selected Device
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS)
        {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        }
        else
        {
            std::cout << "Device[" << i << "]: program successful!\n";
            OCL_CHECK(err, krnl_argument = cl::Kernel(program, "vadd", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device)
    {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }

    // These commands will allocate memory on the Device. The cl::Buffer objects can
    // be used to reference the memory locations on the device.
    OCL_CHECK(err, cl::Buffer buffer_a(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, totalSize,
                                       dataArray, &err));
    OCL_CHECK(err, cl::Buffer buffer_result(context, CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY, totalSize,
                                            output_arr, &err));

    // set the kernel Arguments
    int narg = 0;
    OCL_CHECK(err, err = krnl_argument.setArg(narg++, buffer_a));
    OCL_CHECK(err, err = krnl_argument.setArg(narg++, buffer_result));
    OCL_CHECK(err, err = krnl_argument.setArg(narg++, totalSize));

    // Data will be migrated to kernel space
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_a}, 0 /* 0 means from host*/));

    // Launch the Kernel
    OCL_CHECK(err, err = q.enqueueTask(krnl_argument));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, q.enqueueMigrateMemObjects({buffer_result}, CL_MIGRATE_MEM_OBJECT_HOST));

    OCL_CHECK(err, q.finish());

    size_t frameWidth = NewData[0].cols;
    size_t frameHeight = NewData[0].rows;
    int frameType = NewData[0].type();

    std::vector<cv::Mat> newFrames;
    int index = 0;
    for (const auto &frame : NewData)
    {
        cv::Mat newFrame(frameHeight, frameWidth, frameType);
        unsigned char *newFrameData = newFrame.ptr();
        int dataSize = frame.total() * frame.elemSize();
        std::memcpy(newFrameData, output_arr + index, dataSize);
        newFrames.push_back(newFrame);
        index += dataSize;
    }

    // Verify the result
    int match = 0;
    if (!compareVectorOfMats(HostGreyscale, newFrames))
    {
        printf(error_message.c_str());
        match = 1;
    }

    if (match == 0)
    {
        std::cout << "****Out of Kernel****" << std::endl;
        writeIntoVideo(newFrames, videoDetails, filepath);
        std::cout << "********************" << std::endl;
    }

    delete[] dataArray;
    delete[] output_arr;
    return 0;
}