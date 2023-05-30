#pragma once
#include <fstream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class VideoDetails
{
    public:
            int fourcc, fps, rows, cols;
};

class Pixel
{
    public:
            std::vector<cv::Vec3b> Pixels;
};

void extractFrameData(std::string filepath, VideoDetails& videoDetails,std::vector<Pixel>& vecOfAllFramesPixels);
std::vector<cv::Mat> writePixelIntoFrame(std::vector<Pixel>& AllFrames, VideoDetails videoDetails);
void ProcessPixelData(std::vector<Pixel>& AllFrames);
void writeIntoVideo(std::vector<cv::Mat> NewFrames, VideoDetails mp4, std::string filepath);
std::vector<uint32_t> GetAllPixels(std::vector<Pixel>& AllFrames);
Pixel WtiteKernelData(std::vector<uint32_t> output_result, int size);