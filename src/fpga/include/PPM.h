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

std::vector<cv::Mat> extractFrameData(std::string filepath, VideoDetails& videoDetails);
std::vector<cv::Mat> ProcessPixelData(std::vector<cv::Mat> Frames, VideoDetails videoDetails);
void writeIntoVideo(std::vector<cv::Mat> NewFrames, VideoDetails mp4, std::string filepath);
bool compareVectorOfMats(const std::vector<cv::Mat> &vec1, const std::vector<cv::Mat> &vec2);