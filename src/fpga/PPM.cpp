#include <iostream>
#include <vector>
#include <stdlib.h>
#include <typeinfo>
#include <cmath>
#include "opencv2/core/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "PPM.h"

// Extract frames from video and store them in vector
std::vector<cv::Mat> extractFrameData(std::string filepath, VideoDetails &videoDetails)
{
	try
	{
		std::cout << "Started extractFrameData" << std::endl;
		std::vector<cv::Mat> AllFrames;
		Pixel VecOfPixel;

		cv::VideoCapture cap(filepath); // open the video file
		if (!cap.isOpened())			// check if we succeeded
			std::cout << "Can not open Video file" << std::endl;

		videoDetails.fourcc = cap.get(cv::CAP_PROP_FOURCC);
		videoDetails.fps = cap.get(cv::CAP_PROP_FPS);			// FPS of video
		videoDetails.rows = cap.get(cv::CAP_PROP_FRAME_WIDTH);	// Width of video
		videoDetails.cols = cap.get(cv::CAP_PROP_FRAME_HEIGHT); // Height of video

		std::cout << "fps"
				  << " " << videoDetails.fps << " rows: " << videoDetails.rows << " cols: " << videoDetails.cols << std::endl;

		for (int frameNum = 0; frameNum < cap.get(cv::CAP_PROP_FRAME_COUNT); frameNum++)
		{
			cv::Mat frame;
			Pixel VecOfPixel;
			cap >> frame; // get the next frame from vide
			AllFrames.push_back(frame);
		}
		std::cout << "Completed extractFrameData" << std::endl;
		return AllFrames;
	}
	catch (cv::Exception &e)
	{
		std::cout << "Failed : " << e.msg << std::endl;
		exit(1);
	}
}

// Converting each frame pixels into greyscale pixel
std::vector<cv::Mat> ProcessPixelData(std::vector<cv::Mat> Frames, VideoDetails videoDetails)
{
	try
	{
		std::cout << "Started ProcessPixelData" << std::endl;
		std::vector<cv::Mat> AllFrames;
		for (int j = 0; j < Frames.size(); j++)
		{
			cv::Mat newFrame(videoDetails.cols, videoDetails.rows, CV_8UC3);
			for (int i = 0; i < 3 * videoDetails.cols * videoDetails.rows; i += 3)
			{
				unsigned char grayscaleValue = Frames[j].data[i] * 0.0722 + Frames[j].data[i + 1] * 0.7152 + Frames[j].data[i + 2] * 0.2126; // getting greyscale pixel
				newFrame.data[i] = grayscaleValue;
				newFrame.data[i + 1] = grayscaleValue;
				newFrame.data[i + 2] = grayscaleValue;
			}
			AllFrames.push_back(newFrame);
		}
		std::cout << "Completed ProcessPixelData" << std::endl;
		return AllFrames;
	}
	catch (const std::exception &e)
	{
		std::cout << e.what() << std::endl;
	}
}

// Creating video file using frames
void writeIntoVideo(std::vector<cv::Mat> NewFrames, VideoDetails mp4, std::string filepath)
{
	try
	{
		std::cout << "New Frames size " << NewFrames.size() << std::endl;
		std::cout << "Started writeIntoVideo" << std::endl;
		cv::VideoWriter output_cap(filepath, mp4.fourcc, mp4.fps, cv::Size(mp4.rows, mp4.cols), true);
		for (int i = 0; i < NewFrames.size(); i++)
		{
			output_cap.write(NewFrames[i]);
		}
		std::cout << "Completed writeIntoVideo" << std::endl;
	}
	catch (const std::exception &e)
	{
		std::cout << e.what() << std::endl;
	}
}

bool compareVectorOfMats(const std::vector<cv::Mat> &vec1, const std::vector<cv::Mat> &vec2)
{
	if (vec1.size() != vec2.size())
	{
		return false; // Vectors have different sizes, not equal
	}

	for (size_t i = 0; i < vec1.size(); ++i)
	{
		if (vec1[i].size() != vec2[i].size() ||
			vec1[i].type() != vec2[i].type() ||
			cv::countNonZero(vec1[i] != vec2[i]) > 0)
		{
			return false; // Matrices at index i are different, not equal
		}
	}

	return true; // All elements are equal
}
