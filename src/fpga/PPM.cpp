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


void extractFrameData(std::string filepath, VideoDetails& videoDetails, std::vector<Pixel>& AllFrames)
{
	try
	{
		Pixel VecOfPixel;

		cv::VideoCapture cap(filepath); // open the video file
		if(!cap.isOpened())  // check if we succeeded
			std::cout << "Can not open Video file" << std::endl;

		videoDetails.fourcc = cap.get(cv::CAP_PROP_FOURCC);
		videoDetails.fps = cap.get(cv::CAP_PROP_FPS);   // FPS of video
		videoDetails.rows = cap.get(cv::CAP_PROP_FRAME_WIDTH); // Width of video 
		videoDetails.cols = cap.get(cv::CAP_PROP_FRAME_HEIGHT); // Height of video

		std::cout << videoDetails.fourcc << " " << videoDetails.fps << " " << videoDetails.rows << " " << videoDetails.cols << std::endl;

		for(int frameNum = 0; frameNum < cap.get(cv::CAP_PROP_FRAME_COUNT); frameNum++)
		{
			cv::Mat frame;
			Pixel VecOfPixel;
			cap >> frame; // get the next frame from vide
			for(int i = 0; i < frame.rows; i++)
			{
				for(int j = 0; j < frame.cols; j++)
				{
					cv::Vec3b pixel;
					pixel[0] = frame.at<cv::Vec3b>(i, j)[0]; //B
					pixel[1] = frame.at<cv::Vec3b>(i, j)[1]; //G
					pixel[2] = frame.at<cv::Vec3b>(i, j)[2]; //R
					VecOfPixel.Pixels.push_back(pixel);
				}
			}
			AllFrames.push_back(VecOfPixel);
		}
		std::cout << "Completed extractFrameData" << std::endl;
	}
	catch( cv::Exception& e )
	{
		std::cout << "Failed : " << e.msg << std::endl;
		exit(1);
	}
}

void ProcessPixelData(std::vector<Pixel>& AllFrames)
{
	try
	{
		for (int i = 0; i < AllFrames.size(); i++)
		{
			for (int j = 0; j < AllFrames[i].Pixels.size(); j++)
			{
				unsigned char grayscaleValue = static_cast<unsigned char>(AllFrames[i].Pixels[j][0] * 0.0722 
												+ AllFrames[i].Pixels[j][1] * 0.7152 + AllFrames[i].Pixels[j][2] * 0.2126);
				AllFrames[i].Pixels[j][0] = grayscaleValue;
				AllFrames[i].Pixels[j][1] = grayscaleValue;
				AllFrames[i].Pixels[j][2] = grayscaleValue;
			}
		}
		std::cout << "Completed ProcessPixelData" << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

std::vector<cv::Mat> writePixelIntoFrame(std::vector<Pixel>& AllFrames, VideoDetails videoDetails)
{
	try
	{
		std::vector<cv::Mat> vecOfGreyFrames;
		int b;
		for(int k = 0; k < AllFrames.size() ; k++)
		{
			int pixelCount = 0;
			cv::Mat newFrame(videoDetails.cols, videoDetails.rows, CV_8UC3);
			for (int i = 0; i < newFrame.rows; i++) 
			{
				for (int j = 0; j < newFrame.cols; j++) 
				{
					newFrame.at<cv::Vec3b>(i, j)[0] = static_cast<unsigned char>(AllFrames[k].Pixels[pixelCount][0]);
					//std::cout << static_cast<unsigned char>(AllFrames[k].Pixels[pixelCount][0]) << std::endl;
					newFrame.at<cv::Vec3b>(i, j)[1] = static_cast<unsigned char>(AllFrames[k].Pixels[pixelCount][1]);
					//std::cout << static_cast<unsigned char>(AllFrames[k].Pixels[pixelCount][1]) << std::endl;
					newFrame.at<cv::Vec3b>(i, j)[2] = static_cast<unsigned char>(AllFrames[k].Pixels[pixelCount][2]);  //Greyscale value
					//std::cout << static_cast<unsigned char>(AllFrames[k].Pixels[pixelCount][2]) << std::endl;
					pixelCount++;
				}
			}
			namedWindow("result", cv::WINDOW_AUTOSIZE);
			cv::imshow("result", newFrame);
			cv::waitKey(0); // key press to close window
			cv::destroyWindow("result");
			vecOfGreyFrames.push_back(newFrame);
		}
		std::cout << "Completed writePixelIntoFrame" << std::endl;
		return vecOfGreyFrames;
	}
	catch(const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

void writeIntoVideo(std::vector<cv::Mat> NewFrames, VideoDetails mp4, std::string filepath)
{
	try
	{
		cv::VideoWriter output_cap(filepath, mp4.fourcc, 60, cv::Size(mp4.rows, mp4.cols), true);
		for (int i = 0; i < NewFrames.size(); i++)
		{
			output_cap.write(NewFrames[i]);
			output_cap.write(NewFrames[i]);
		}
		std::cout << "Completed writeIntoVideo" << std::endl;
	}
	catch(const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

std::vector<uint32_t> GetAllPixels(std::vector<Pixel>& AllFrames)
{
	try
	{
		std::vector<uint32_t> AllPixels;
		for (int j = 0; j < AllFrames[1].Pixels.size(); j++)
		{
			AllPixels.push_back((uint32_t)(AllFrames[0].Pixels[j][0]));
			AllPixels.push_back((uint32_t)(AllFrames[0].Pixels[j][1]));
			AllPixels.push_back((uint32_t)(AllFrames[0].Pixels[j][2]));
		}
		return AllPixels;
	}
	catch(const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}


Pixel WtiteKernelData(std::vector<uint32_t> output_result, int size)
{
	try
	{
		Pixel VecOfPixel;
		for(size_t i = 0; i < size; i += 3)
		{
			cv::Vec3b pixel;
			pixel[0] = static_cast<unsigned char>(output_result[0]);
			pixel[1] = static_cast<unsigned char>(output_result[1]);
			pixel[2] = static_cast<unsigned char>(output_result[2]);
			//std::cout << output_result[0] << " " << output_result[1] << " " << output_result[2] << std::endl;
			VecOfPixel.Pixels.push_back(pixel);
		}
		return VecOfPixel;
	}
	catch(const std::exception& e)
	{
		std::cout << e.what() << std::endl;
	}
}

