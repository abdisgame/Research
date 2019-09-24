// FeatureExtraction.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <opencv2\opencv.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>       // std::vector
#include <opencv2/core.hpp>
#include<deque>
#include <string>
#include <sstream>
#include <iostream>
#include <chrono>
#include <thread> 
#include <conio.h>
#include <vector>
#include <fstream>
#include "Dehaze.h"
#include <cmath>
#define PI 3.14159265

using namespace std;
using namespace cv;



/*
int main()
{
	
	//VideoCapture cap(0);



	Mat image = imread("D:/k.jpg");


	
	if (image.empty())
	{
		std::cout << "!!! Failed imread(): image not found" << std::endl;
		// don't let the execution continue, else imshow() will crash.
	}
	Mat dilate_im = image.clone();
	

	
	Mat baru = image;
	Rect roi;
	roi.x = (image.size().width*0.2);
	roi.y = 0;
	roi.width = image.size().width - (image.size().width*0.2);
	roi.height = image.size().height;

	baru = baru(roi);


	//cvtColor(dilate_im, dilate_im, CV_BGR2GRAY);
	
	cvtColor(baru, baru, CV_BGR2GRAY);
	
	
	//dilate(baru, baru, Mat());
	

	threshold(baru, baru, Thresh(baru), 255, THRESH_BINARY);
	dilate(baru, baru, Mat());
	//threshold(dilate_im, dilate_im, Thresh(dilate_im), 255, THRESH_BINARY);
	//dilate(dilate_im, dilate_im, Mat());

	int *PixelRow = new int[baru.rows];
	int *PixelCol = new int[baru.cols];
	for (int i = 0; i < baru.rows; i++)
	{
		PixelRow[i] = baru.cols - countNonZero(baru.row(i));
		cout << PixelRow[i] << endl;
	}
	
	while (true) {
		
		//cap >> edges;
		namedWindow("Original", 1);
		imshow("Original", baru);

		


		namedWindow("Dilate Image", 1);
		imshow("Dilate Image", image);


		waitKey(3);
	}
}
*/
void insertionSort(int window[])
{
	int temp, i, j;
	for (i = 0; i < 9; i++) {
		temp = window[i];
		for (j = i - 1; j >= 0 && temp < window[j]; j--) {
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}
Mat claheGO(Mat src, int _step = 8)
{
	Mat CLAHE_GO = src.clone();
	int block = _step;//pblock
	int width = src.cols;
	int height = src.rows;
	int width_block = width / block; //?????????
	int height_block = height / block;
	//???????  
	int tmp2[8 * 8][256] = { 0 };
	float C2[8 * 8][256] = { 0.0 };
	//??
	int total = width_block * height_block;
	for (int i = 0; i<block; i++)
	{
		for (int j = 0; j<block; j++)
		{
			int start_x = i * width_block;
			int end_x = start_x + width_block;
			int start_y = j * height_block;
			int end_y = start_y + height_block;
			int num = i + block * j;
			//????,?????
			for (int ii = start_x; ii < end_x; ii++)
			{
				for (int jj = start_y; jj < end_y; jj++)
				{
					int index = src.at<uchar>(jj, ii);
					tmp2[num][index]++;
				}
			}
			//???????,???clahe??cl??
			//????? ??«Gem»?? fCliplimit  = 4  , uiNrBins  = 255
			int average = width_block * height_block / 255;
			//????????,????????????????
			//???????,?????cl???,?????? 
			int LIMIT = 40 * average;
			int steal = 0;
			for (int k = 0; k < 256; k++)
			{
				if (tmp2[num][k] > LIMIT) {
					steal += tmp2[num][k] - LIMIT;
					tmp2[num][k] = LIMIT;
				}
			}
			int bonus = steal / 256;
			//hand out the steals averagely  
			for (int k = 0; k < 256; k++)
			{
				tmp2[num][k] += bonus;
			}
			//?????????  
			for (int k = 0; k < 256; k++)
			{
				if (k == 0)
					C2[num][k] = 1.0f * tmp2[num][k] / total;
				else
					C2[num][k] = C2[num][k - 1] + 1.0f * tmp2[num][k] / total;
			}
		}
	}
	//?????????  
	//????????,?????????  
	for (int i = 0; i < width; i++)
	{
		for (int j = 0; j < height; j++)
		{
			//four coners  
			if (i <= width_block / 2 && j <= height_block / 2)
			{
				int num = 0;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i <= width_block / 2 && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * (block - 1);
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j <= height_block / 2) {
				int num = block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2) && j >= ((block - 1)*height_block + height_block / 2)) {
				int num = block * block - 1;
				CLAHE_GO.at<uchar>(j, i) = (int)(C2[num][CLAHE_GO.at<uchar>(j, i)] * 255);
			}
			//four edges except coners  
			else if (i <= width_block / 2)
			{
				//????  
				int num_i = 0;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (i >= ((block - 1)*width_block + width_block / 2)) {
				//????  
				int num_i = block - 1;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + block;
				float p = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j <= height_block / 2) {
				//????  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = 0;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			else if (j >= ((block - 1)*height_block + height_block / 2)) {
				//????  
				int num_i = (i - width_block / 2) / width_block;
				int num_j = block - 1;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				float p = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float q = 1 - p;
				CLAHE_GO.at<uchar>(j, i) = (int)((q*C2[num1][CLAHE_GO.at<uchar>(j, i)] + p * C2[num2][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//?????
			else {
				int num_i = (i - width_block / 2) / width_block;
				int num_j = (j - height_block / 2) / height_block;
				int num1 = num_j * block + num_i;
				int num2 = num1 + 1;
				int num3 = num1 + block;
				int num4 = num2 + block;
				float u = (i - (num_i*width_block + width_block / 2)) / (1.0f*width_block);
				float v = (j - (num_j*height_block + height_block / 2)) / (1.0f*height_block);
				CLAHE_GO.at<uchar>(j, i) = (int)((u*v*C2[num4][CLAHE_GO.at<uchar>(j, i)] +
					(1 - v)*(1 - u)*C2[num1][CLAHE_GO.at<uchar>(j, i)] +
					u * (1 - v)*C2[num2][CLAHE_GO.at<uchar>(j, i)] +
					v * (1 - u)*C2[num3][CLAHE_GO.at<uchar>(j, i)]) * 255);
			}
			//smooth
			CLAHE_GO.at<uchar>(j, i) = CLAHE_GO.at<uchar>(j, i) + (CLAHE_GO.at<uchar>(j, i) << 8) + (CLAHE_GO.at<uchar>(j, i) << 16);
		}
	}
	return CLAHE_GO;
}

void BrightnessAndContrastAuto(const cv::Mat &src, cv::Mat &dst, float clipHistPercent = 0)
{

	CV_Assert(clipHistPercent >= 0);
	CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

	int histSize = 256;
	float alpha, beta;
	double minGray = 0, maxGray = 0;

	//to calculate grayscale histogram
	cv::Mat gray;
	if (src.type() == CV_8UC1) gray = src;
	else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
	else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
	if (clipHistPercent == 0)
	{
		// keep full available range
		cv::minMaxLoc(gray, &minGray, &maxGray);
	}
	else
	{
		cv::Mat hist; //the grayscale histogram

		float range[] = { 0, 256 };
		const float* histRange = { range };
		bool uniform = true;
		bool accumulate = false;
		calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

		// calculate cumulative distribution from the histogram
		std::vector<float> accumulator(histSize);
		accumulator[0] = hist.at<float>(0);
		for (int i = 1; i < histSize; i++)
		{
			accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
		}

		// locate points that cuts at required value
		float max = accumulator.back();
		clipHistPercent *= (max / 100.0); //make percent as absolute
		clipHistPercent /= 2.0; // left and right wings
								// locate left cut
		minGray = 0;
		while (accumulator[minGray] < clipHistPercent)
			minGray++;

		// locate right cut
		maxGray = histSize - 1;
		while (accumulator[maxGray] >= (max - clipHistPercent))
			maxGray--;
	}

	// current range
	float inputRange = maxGray - minGray;

	alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
	beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

										 // Apply brightness and contrast normalization
										 // convertTo operates with saurate_cast
	src.convertTo(dst, -1, alpha, beta);

	// restore alpha channel from source 
	if (dst.type() == CV_8UC4)
	{
		int from_to[] = { 3, 3 };
		cv::mixChannels(&src, 4, &dst, 1, from_to, 1);
	}
	return;
}


//own Func
//__________________//

struct Position {
	int pointX;
	int pointY;
};
struct MinMax {
	int min;
	int max;
};

struct HueMomentData {
	double data[7];
};

struct MomentInvariant {
	double Mean;
	double Median;
	double StDev;
	double Skewness;
	double Kurtosis;
	double Variance;
};

struct LineSlope {
	float LeftToMid;
	float MidToRight;
	float LeftToLeftMid;
	float LeftMidToMid;
	float MidToRightMid;
	float RightMidToRight;
	float LeftTopToMid;
	float MidTopToRight;
};


struct DynLine {
	int leftX;
	int leftY;
	int rightX;
	int rightY;
	int topX;
	int topY;
	int botX;
	int botY;
};

DynLine dynLine;

//for adaptive threshold
int Thresh(Mat image) {
	int total = 0;
	int count = 0;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++)
		{
			count++;
			total += (int)image.at<uchar>(i, j);
		}
	}
	double average = (double)total / (double)count;

	return (average - (0.2*average));
}

//filter percent / histogram
int* NoiseFilter(Mat img,bool rowMode) {
	if (rowMode) {

		int *PixelRow = new int[img.rows];
		int maxVal = 0;
		//get max peak
		for (int i = 0; i < img.rows; i++)
		{
			PixelRow[i] = img.cols - countNonZero(img.row(i));
			if (PixelRow[i] > maxVal)
				maxVal = PixelRow[i];
		}
		//cout << maxVal << endl;
		//remove any data under threshold 10% from maxVal
		int threshOutlier = (int)(0.1*maxVal);

		for (int i = 0; i < img.rows; i++)
		{
			
			if (PixelRow[i] < threshOutlier)
				PixelRow[i] = 0;
		}
		return PixelRow;
	}
	else {
		int *PixelCol = new int[img.cols];
		int maxVal = 0;
		//get max peak
		for (int i = 0; i < img.cols; i++)
		{
			PixelCol[i] = img.rows - countNonZero(img.col(i));
			if (PixelCol[i] > maxVal)
				maxVal = PixelCol[i];
		}
		//remove any data under threshold 10% from maxVal
		int threshOutlier = (int)(0.1*maxVal);

		for (int i = 0; i < img.cols; i++)
		{

			if (PixelCol[i] < threshOutlier)
				PixelCol[i] = 0;
		}
		return PixelCol;
	}
}

//filter percent / histogram
int* WithoutNoiseFilter(Mat img, bool rowMode) {
	if (rowMode) {

		int *PixelRow = new int[img.rows];
		int maxVal = 0;
		//get max peak
		for (int i = 0; i < img.rows; i++)
		{
			PixelRow[i] = img.cols - countNonZero(img.row(i));
			if (PixelRow[i] > maxVal)
				maxVal = PixelRow[i];
		}
		//cout << maxVal << endl;
		//remove any data under threshold 10% from maxVal
		int threshOutlier = (int)(0.1*maxVal);

		for (int i = 0; i < img.rows; i++)
		{

			//if (PixelRow[i] < threshOutlier)
				//PixelRow[i] = 0;
		}
		return PixelRow;
	}
	else {
		int *PixelCol = new int[img.cols];
		int maxVal = 0;
		//get max peak
		for (int i = 0; i < img.cols; i++)
		{
			PixelCol[i] = img.rows - countNonZero(img.col(i));
			if (PixelCol[i] > maxVal)
				maxVal = PixelCol[i];
		}
		//remove any data under threshold 10% from maxVal
		int threshOutlier = (int)(0.1*maxVal);

		for (int i = 0; i < img.cols; i++)
		{

			//if (PixelCol[i] < threshOutlier)
			//	PixelCol[i] = 0;
		}
		return PixelCol;
	}
}


//get top of eyelid position, Position is struct data contain x,y position
Position getPosEyelidTop(int *pixelRow, Mat img) {
	Position pos;
	pos.pointX = (int)(img.cols / 2);
	pos.pointY = 0;

	int addMargin = (int)(0.05*img.cols);

	int xStart = pos.pointX - addMargin;
	int xStop = pos.pointX + addMargin;
	int widtheye = xStop - xStart;

	Rect myROI(xStart, 0, widtheye, img.rows);

	Mat nwImg = img(myROI);

//	namedWindow("topeye");
	//imshow("topeye", nwImg);

	int *PixelRownwImg = new int[nwImg.rows];
	PixelRownwImg = NoiseFilter(nwImg, true);


	for (int i = 0; i < nwImg.rows; i++)
	{
		pos.pointY = i;
		if (PixelRownwImg[i] != 0)
			break;
	}

	//namedWindow("topeye");
//	imshow("topeye", nwImg);
	/*
	int *PixelCol = new int[nwImg.cols];

	for (int i = 0; i < nwImg.cols; i++)
	{
		PixelCol[i] = nwImg.rows - countNonZero(nwImg.col(i));
	}
	int jyPixelCol = 0;
	int jColPixelblack = 0;
	for (int j = 0; j < nwImg.cols; j++)
	{
		jyPixelCol += (j + 1)*PixelCol[j];
		jColPixelblack += PixelCol[j];
	}
	pos.pointX = jyPixelCol / jColPixelblack;
	*/
	/*

	cout << "posisi" << endl;
	int densityPixelCol=0;
	for (int i = 0; i < img.cols; i++)
	{
		if ((int)img.at<uchar>(pos.pointY, i) == 0) {
			densityPixelCol += (int)(i + 1) * 1;
		}
		cout << (int)img.at<uchar>(pos.pointY, i) << endl;
	}

	pos.pointX = (int)((densityPixelCol / pixelRow[pos.pointY])-1);
	*/
//	cout << "posisi" << endl;
	//cout << pos.pointX << endl;
	//cout << pos.pointY << endl;



	return pos;
}

Position getPosEyelidBot(int *pixelRow, int *pixelCol, Mat img) {
	Position pos;
	pos.pointX = (int)(img.cols/2);
	pos.pointY = 0;

	int addMargin = (int)(0.05*img.cols);

	int xStart = pos.pointX-addMargin;
	int xStop = pos.pointX + addMargin;
	int widtheye = xStop - xStart;

	Rect myROI(xStart, 0, widtheye, img.rows);

	Mat nwImg = img(myROI);

	//namedWindow("boteye");
	//imshow("boteye", nwImg);

	int *PixelRownwImg = new int[nwImg.rows];
	PixelRownwImg = NoiseFilter(nwImg, true);


	for (int i = nwImg.rows-1; i >0; i--)
	{
		pos.pointY = i;
		if (PixelRownwImg[i] != 0)
			break;
	}

	return pos;
}


Position getPosEyesideRight(int *pixelRow, int *pixelCol, Mat img, int startPosY) {
	Position pos;
	pos.pointX = 0;
	pos.pointY = 0;

	int xStart = (int)(0.9*img.cols);
	int xStop = img.cols;
	int widtheye = img.cols - xStart;
	
	int yStart = startPosY;
	int yStop = img.rows;
	int heighteye = img.rows - yStart;

	Rect myROI(xStart, yStart, widtheye, heighteye);

	Mat nwImg = img(myROI);

	//namedWindow("righteye");
	//imshow("righteye", nwImg);

	int *PixelColnwImg = new int[nwImg.cols];
	PixelColnwImg = NoiseFilter(nwImg, false);


	for (int i = nwImg.cols - 1; i > 0; i--)
	{
		pos.pointX = i;
		if (PixelColnwImg[i] != 0)
			break;
	}

	pos.pointX += xStart;


	int *PixelRownwImg = new int[nwImg.rows];
	PixelRownwImg = NoiseFilter(nwImg, true);


	for (int i = nwImg.rows-1; i > 0; i--)
	{
		pos.pointY = i;
		if (PixelRownwImg[i] != 0)
			break;
	}

	pos.pointY += yStart;
	

	/*
	for (int i = img.rows - 1; i >0; i--)
	{
		pos.pointY = i;
		if (pixelRow[i] != 0)
			break;
	}


	
	int pixelValue = 0;
	
	cout << "val bw" << endl;

	int xStart = 0;
	int xStop = img.cols;
	int widtheye = img.rows-pos.pointY;

	Rect myROI(0, pos.pointY, img.cols-1, widtheye);
	Mat nwImg = img(myROI);

	int *PixelCol = new int[nwImg.cols];

	//get max peak
	for (int i = nwImg.cols-1; i > 0; i--)
	{
		pos.pointX = i;
		PixelCol[i] = nwImg.rows - countNonZero(nwImg.col(i));
		if (PixelCol[i] != 0)
			break;
	}

	*/
//	int startcrop = xStart;


//	Rect myROI(startcrop, 0, widtheye, img.rows);
//	Mat nwImg = img(myROI);
	//cout << (int)img.at<uchar>(pos.pointX, pos.pointY) << endl;

	/*
	int xStart = pos.pointX;
	int xStop = img.cols;
	int widtheye = img.cols-xStart;

	int startcrop = xStart;
	

	Rect myROI(startcrop, 0, widtheye, img.rows);
	Mat nwImg = img(myROI);

	int *PixelRownwImg = new int[nwImg.rows];
	PixelRownwImg = NoiseFilter(nwImg, true);


	for (int i = startPosY; i < nwImg.rows; i++)
	{
		pos.pointY = i;
		if (PixelRownwImg[i] != 0)
			break;
	}
	
	*/
	return pos;
}


void Hist_compare() {
	Mat src_base, hsv_base;
	Mat src_test1, hsv_test1;
	Mat src_test2, hsv_test2;
	Mat hsv_half_down;

	src_base = imread("D:/d2.jpg");
	src_test1 = imread("D:/d1.jpg");
	src_test2 = imread("D:/d6.jpg");

	/// Convert to HSV
	cvtColor(src_base, hsv_base, COLOR_BGR2HSV);
	cvtColor(src_test1, hsv_test1, COLOR_BGR2HSV);
	cvtColor(src_test2, hsv_test2, COLOR_BGR2HSV);
	hsv_half_down = hsv_base(Range(hsv_base.rows / 2, hsv_base.rows - 1), Range(0, hsv_base.cols - 1));

	/// Using 50 bins for hue and 60 for saturation
	int h_bins = 50; int s_bins = 60;
	int histSize[] = { h_bins, s_bins };

	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };

	const float* ranges[] = { h_ranges, s_ranges };

	// Use the o-th and 1-st channels
	int channels[] = { 0, 1 };


	/// Histograms
	MatND hist_base;
	MatND hist_half_down;
	MatND hist_test1;
	MatND hist_test2;

	/// Calculate the histograms for the HSV images
	calcHist(&hsv_base, 1, channels, Mat(), hist_base, 2, histSize, ranges, true, false);
	normalize(hist_base, hist_base, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false);
	normalize(hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());

	calcHist(&hsv_test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false);
	normalize(hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat());

	/// Apply the histogram comparison methods
	for (int i = 0; i < 4; i++)
	{
		int compare_method = i;
		double base_base = compareHist(hist_base, hist_base, compare_method);
		double base_half = compareHist(hist_base, hist_half_down, compare_method);
		double base_test1 = compareHist(hist_base, hist_test1, compare_method);
		double base_test2 = compareHist(hist_base, hist_test2, compare_method);

		printf(" Method [%d] Perfect, Base-Half, Base-Test(1), Base-Test(2) : %f, %f, %f, %f \n", i, base_base, base_half, base_test1, base_test2);
	}


}


HueMomentData HueMomentCal(int* PixelRowCropped, int* PixelColCropped, Mat img) {
	//___________________________Recrop ___________________________
	int xStart = 0;
	int xStop = img.cols;
	int yStart = 0;
	int yStop = img.rows;
	int width = img.cols;
	int height = img.rows;

	for (int i = 0; i < img.cols; i++)
	{
		if (PixelColCropped[i] != 0) {
			xStart = i;
			break;
		}
	}
	for (int i = img.cols - 1; i > 0; i--)
	{
		if (PixelColCropped[i] != 0) {
			xStop = i;
			break;
		}
	}

	for (int i = 0; i < img.rows; i++)
	{
		if (PixelRowCropped[i] != 0) {
			yStart = i;
			break;
		}
	}
	for (int i = img.rows - 1; i > 0; i--)
	{
		if (PixelRowCropped[i] != 0) {
			yStop = i;
			break;
		}
	}

	width = xStop - xStart;
	height = yStop - yStart;

	Rect myROI(xStart, yStart, width, height);

	Mat nwImg = img(myROI);
	//___________________________End Recrop ___________________________

	HueMomentData hueMomentData;

	Moments mymoments = moments(nwImg, true);

	// Calculate Hu Moments
	double huMoments[7];
	HuMoments(mymoments, huMoments);

	// Log scale hu moments
	for (int i = 0; i < 7; i++)
	{
		huMoments[i] = -1 * copysign(1.0, huMoments[i]) * log10(abs(huMoments[i]));
		hueMomentData.data[i] = huMoments[i];
	}



	cout << "_________Match____________" << endl;
	cout << hueMomentData.data[0] << endl;
	cout << hueMomentData.data[1] << endl;
	cout << hueMomentData.data[2] << endl;
	cout << hueMomentData.data[3] << endl;
	cout << hueMomentData.data[4] << endl;
	cout << hueMomentData.data[5] << endl;
	cout << hueMomentData.data[6] << endl;
	cout << hueMomentData.data[7] << endl;
	cout << "__________________________" << endl;


	return hueMomentData;
}


LineSlope LineDetection(int* PixelRowCropped, int* PixelColCropped, Mat lineDetect, Mat newout) {
/*	int xstart = 0;
	int xstop = lineDetect.cols - 1;
	int ystart = 0;
	int ystop = lineDetect.rows-1;

	for (int i = 0; i < lineDetect.rows; i++)
	{
		ystart = i;
		if (PixelRowCropped[i] != 0)
			break;
	}
	
	for (int i = lineDetect.rows - 1; i >0; i--)
	{
		ystop = i;
		if (PixelRowCropped[i] != 0)
			break;
	}

	for (int i = 0; i < lineDetect.cols; i++)
	{
		xstart = i;
		if (PixelColCropped[i] != 0)
			break;
	}

	for (int i = lineDetect.cols - 1; i >0; i--)
	{
		xstop = i;
		if (PixelColCropped[i] != 0)
			break;
	}
	*/
	/*
	cout << "Line Start End of xY" << endl;
	cout << xstart << endl;
	cout << xstop << endl;
	cout << ystart << endl;
	cout << ystop << endl;
	*/
	/*
	int * myLine = new int[lineDetect.cols];
	for (int i = 0; i<lineDetect.cols; i++)
		myLine[i] =0;

	cout << "My line" << endl;
	for (int i = 0; i < lineDetect.cols ; i++)
	{
		for (int j = lineDetect.rows-1; j > 0; j--)
		{
			int k = lineDetect.at<uchar>(i, j);
			//cout << k << endl;
			//cout << "_____" << endl;
			myLine[i] = j;
			if (k == 0) {
				circle(newout, Point(i, j), 3, Scalar(255, 128, 1288), 1, 8);
				j = lineDetect.rows - 1;
				i++;
				//i = xstart;
			//	break;
			}
			//{
			//	
			//}
		}
	}
	*/


//	namedWindow("newout");
	//imshow("newout", newout);
	/*
	cout << lineDetect.rows << endl;
	cout << "My line" << endl;
	
	for(int i=0;i<lineDetect.cols;i++)
		cout << myLine[i] << endl;

	cout << "My line" << endl;
	//cout << lineDetect.rows << endl;
	*/

	LineSlope lineSlope;
	lineSlope.LeftToMid = 0;
	lineSlope.MidToRight = 0;
	lineSlope.LeftToLeftMid = 0;
	lineSlope.LeftMidToMid = 0;
	lineSlope.MidToRightMid = 0;
	lineSlope.RightMidToRight = 0;

	lineSlope.LeftTopToMid = 0;
	lineSlope.MidTopToRight = 0;



	int xStart = 0;
	int xStop = lineDetect.cols;
	int yStart = 0;
	int yStop = lineDetect.rows;
	int width = lineDetect.cols;
	int height = lineDetect.rows;

	for (int i = 0; i < lineDetect.cols; i++)
	{
		if (PixelColCropped[i] != 0) {
			xStart = i;
			break;
		}
	}
	for (int i = lineDetect.cols-1; i > 0 ; i--)
	{
		if (PixelColCropped[i] != 0) {
			xStop = i;
			break;
		}
	}

	for (int i = 0; i < lineDetect.rows; i++)
	{
		if (PixelRowCropped[i] != 0) {
			yStart = i;
			break;
		}
	}
	for (int i = lineDetect.rows - 1; i > 0; i--)
	{
		if (PixelRowCropped[i] != 0) {
			yStop = i;
			break;
		}
	}

	width = xStop - xStart;
	height = yStop - yStart;

	Rect myROI(xStart, yStart, width, height);

	Mat nwImg = lineDetect(myROI);

	newout = newout(myROI);

	//namedWindow("nwImg");
	//imshow("nwImg", nwImg);

	//-------------End of recrop image---------------//

//
//	Size size(100, 100);//the dst image size,e.g.100x100
	
//	Mat smallimg;//src image
//	resize(nwImg, smallimg, size);//resize image

	//-------------Begin of Top Eyelid---------------//

	int midOfTop = (int)(nwImg.cols / 2);
	
	int topWidth = 10;
	int topHeight = nwImg.rows;

	Rect nwROI((midOfTop-5), 0, topWidth, topHeight);

	Mat topImg2 = nwImg(nwROI);

	int topYLine = 0;

//	namedWindow("topImg2");
//	imshow("topImg2", topImg2);

	int *TopPixelRow = new int[topImg2.rows];

	for (int i = 0; i < topImg2.rows; i++)
	{
		TopPixelRow[i] = topImg2.cols - countNonZero(topImg2.row(i));
		topYLine = i;

		if (TopPixelRow[i] != 0) {
			break;
		}

	}

	//-------------End of Top Eyelid---------------//





//	cout << "new img cols" << endl;
//	cout << nwImg.cols << endl;
//	cout << nwImg.cols%7 << endl;

	int mod = nwImg.cols%5;
	int newSizeCol = nwImg.cols - mod;
	int totalLoop = newSizeCol / 5;

	int currentCheck = 0;
	int borderCheck = 5;

	int * myXLine = new int[totalLoop];
	int * myYLine = new int[totalLoop];

	//cout << "Start Line" << endl;

	for (int i = 0; i < totalLoop; i++) {
		int nwWidth = borderCheck - currentCheck;
		int nwHeight = nwImg.rows;

		Rect nwROI(currentCheck, 0, nwWidth, nwHeight);

		Mat nwImg2 = nwImg(nwROI);

		int xLine = currentCheck + 3;
		int yLine = nwHeight - 1;

		//loop, get rows from y line, if pixelrows not 0, pick y
		int *PixelRow = new int[nwImg2.rows];

		for (int i = nwImg2.rows - 1; i > 0; i--)
		{
			PixelRow[i] = nwImg2.cols - countNonZero(nwImg2.row(i));
			yLine = i;

			if (PixelRow[i] != 0) {
				break;
			}

		}

		currentCheck = borderCheck;
		borderCheck += 5;

		myXLine[i] = xLine;
		myYLine[i] = yLine;
		//cout << nwImg2.cols << endl;
		//cout << "#####" << endl;
	//	cout << xLine << endl;
	//	cout << yLine << endl;
		//cout << "______" << endl;
		//circle(newout, Point(xLine, yLine), 3, Scalar(255, 0, 255), 1, 8);

	}

	int total = 0;
	int average = 0;

	for (int i = 0; i < totalLoop; i++) {
		total += myYLine[i];
	}

	average = (int)(total / totalLoop);
	average -= (0.2*average);

	int midValX = myXLine[(int)(totalLoop / 2)];
	int midValY = myYLine[(int)(totalLoop / 2)];

	

	for (int i = 0; i < totalLoop; i++) {
		if (myYLine[i] < average) {
			
			int slope = (midValY - myYLine[i]) / (midValX-myXLine[i]);
			
			myYLine[i] = average;


		}
	}

	for (int i = 0; i < totalLoop; i++) {
		circle(newout, Point(myXLine[i], myYLine[i]), 3, Scalar(255, 0, 255), 1, 8);
	}

//	cout << "End Line" << endl;
	
	int leftPos = (int)(totalLoop*0.03);
	int rightPos = totalLoop - 1;
	int midPos = (int)(totalLoop/2);
	int midLeftPos = midPos - 3;
	int midRightPos = midPos + 3;

	//myYLine[leftPos] = myYLine[rightPos];
	

	circle(newout, Point(myXLine[leftPos], myYLine[leftPos]), 5, Scalar(255, 255, 0), 2, 8);
	circle(newout, Point(myXLine[rightPos], myYLine[rightPos]), 5, Scalar(255, 255, 0), 2, 8);
	circle(newout, Point(myXLine[midPos], myYLine[midPos]), 5, Scalar(255, 255, 0), 2, 8);
	circle(newout, Point(myXLine[midLeftPos], myYLine[midLeftPos]), 5, Scalar(255, 255, 0), 2, 8);
	circle(newout, Point(myXLine[midRightPos], myYLine[midRightPos]), 5, Scalar(255, 255, 0), 2, 8);

	circle(newout, Point(midOfTop, topYLine), 7, Scalar(255, 255, 0), 1, 8);

	
	dynLine.leftX = myXLine[leftPos];
	dynLine.leftY = myYLine[leftPos];
	dynLine.rightX = myXLine[rightPos];
	dynLine.rightY = myYLine[rightPos];
	dynLine.topX = midOfTop;
	dynLine.topY = topYLine;
	dynLine.botX = myXLine[midPos];
	dynLine.botY = myYLine[midPos];
	
	


	float slopeLM = (((float)(myYLine[midPos]-myYLine[leftPos])) / (myXLine[midPos] - myXLine[leftPos]));
	float slopeMR = (((float)(myYLine[rightPos]-myYLine[midPos])) / (myXLine[rightPos] - myXLine[midPos]));

	float slopeLCl = (((float)(myYLine[midLeftPos]-myYLine[leftPos])) / (myXLine[midLeftPos] - myXLine[leftPos]));
	float slopeClC = (((float)(myYLine[midPos] - myYLine[midLeftPos])) / (myXLine[midPos] - myXLine[midLeftPos]));
	float slopeCCr = ((((float)myYLine[midRightPos] - myYLine[midPos])) / (myXLine[midRightPos] - myXLine[midPos]));
	float slopeCrR = (((float)(myYLine[rightPos] - myYLine[midRightPos])) / (myXLine[rightPos] - myXLine[midRightPos]));

	float slopeTopLeftMid = (((float)(topYLine - myYLine[midPos])) / (midOfTop - myXLine[midPos]));
	float slopeTopMidRight = (((float)(myYLine[rightPos] - topYLine)) / (myXLine[rightPos] - midOfTop));

	//cout << myYLine[midLeftPos] << endl;
//	cout << myYLine[leftPos] << endl;
//	cout << myXLine[midLeftPos] << endl;
//	cout << myXLine[leftPos] << endl;
	
//	cout << "analyze" << endl;
////	cout << myYLine[leftPos] << endl;
//	cout << myYLine[midLeftPos] << endl;
////	cout << myYLine[midPos] << endl;
//	cout << myYLine[midRightPos] << endl;
//	cout << myYLine[totalLoop-1] << endl;
	
	//cout << "end of analyze" << endl;



	
//	cout << "Slope Value" << endl;
	float resultslopeLM = atan(slopeLM) * 180 / PI;
	//cout << roundf(resultslopeLM * 100) / 100 << endl;

	float resultslopeMR = atan(slopeMR) * 180 / PI;
//	cout << roundf(resultslopeMR * 100) / 100 << endl;


	float resultslopeLCl = atan(slopeLCl) * 180 / PI;
	//cout << roundf(resultslopeLCl * 100) / 100 << endl;

	float resultslopeClC = atan(slopeClC) * 180 / PI;
	//cout << roundf(resultslopeClC * 100) / 100 << endl;

	float resultslopeCCr = atan(slopeCCr) * 180 / PI;
	//cout << roundf(resultslopeCCr * 100) / 100  << endl;

	float resultslopeCrR = atan(slopeCrR) * 180 / PI;
	//cout << roundf(resultslopeCrR * 100) / 100  << endl;

	float resultslopeLeftTopMid = atan(slopeTopLeftMid) * 180 / PI;
	//cout << roundf(resultslopeLeftTopMid * 100) / 100 << endl;

	float resultslopeTopMidRight = atan(slopeTopMidRight) * 180 / PI;
	//cout << roundf(resultslopeTopMidRight * 100) / 100 << endl;


	lineSlope.LeftToMid = roundf(resultslopeLM * 100) / 100;
	lineSlope.MidToRight = roundf(resultslopeMR * 100) / 100;
	
	lineSlope.LeftToLeftMid = roundf(resultslopeLCl * 100) / 100;
	lineSlope.LeftMidToMid = roundf(resultslopeClC * 100) / 100;
	lineSlope.MidToRightMid = roundf(resultslopeCCr * 100) / 100;
	lineSlope.RightMidToRight = roundf(resultslopeCrR * 100) / 100;

	lineSlope.LeftTopToMid = roundf(resultslopeLeftTopMid * 100) / 100;
	lineSlope.MidTopToRight = roundf(resultslopeTopMidRight * 100) / 100;


	//cout << "_________" << endl;
	namedWindow("showMe");
	imshow("showMe", newout);

	return lineSlope;
	
	
//	namedWindow("nwImg2");
//	imshow("nwImg2", nwImg2);

}



LineSlope LineDetection2(int* PixelRowCropped, int* PixelColCropped, Mat lineDetect, Mat newout) {
	LineSlope lineSlope;
	lineSlope.LeftToMid = 0;
	lineSlope.MidToRight = 0;
	lineSlope.LeftToLeftMid = 0;
	lineSlope.LeftMidToMid = 0;
	lineSlope.MidToRightMid = 0;
	lineSlope.RightMidToRight = 0;

	lineSlope.LeftTopToMid = 0;
	lineSlope.MidTopToRight = 0;



	int xStart = 0;
	int xStop = lineDetect.cols;
	int yStart = 0;
	int yStop = lineDetect.rows;
	int width = lineDetect.cols;
	int height = lineDetect.rows;

	for (int i = 0; i < lineDetect.cols; i++)
	{
		if (PixelColCropped[i] != 0) {
			xStart = i;
			break;
		}
	}
	for (int i = lineDetect.cols - 1; i > 0; i--)
	{
		if (PixelColCropped[i] != 0) {
			xStop = i;
			break;
		}
	}

	for (int i = 0; i < lineDetect.rows; i++)
	{
		if (PixelRowCropped[i] != 0) {
			yStart = i;
			break;
		}
	}
	for (int i = lineDetect.rows - 1; i > 0; i--)
	{
		if (PixelRowCropped[i] != 0) {
			yStop = i;
			break;
		}
	}

	width = xStop - xStart;
	height = yStop - yStart;

	Rect myROI(xStart, yStart, width, height);

	Mat nwImg = lineDetect(myROI);

	newout = newout(myROI);

	namedWindow("nwtop");
	imshow("nwtop", nwImg);

	//-------------End of recrop image---------------//

	//

	int mod = nwImg.cols % 4;
	int newSizeCol = nwImg.cols;
	

	int currentCheck = 0;
	int borderCheck = 1;

	//int * myXLine = new int[totalLoop];
	//int * myYLine = new int[totalLoop];

	//int xmid = (int)((double)nwImg.cols / (double)2);
//	int ymid = 0;

	int topcur = 0;
	int topstep = 1;
	int topwidth = topstep - topcur;
	int xTop = topcur;
	int yTop = 0;
	//int totalLoop = (int)((double)newSizeCol/(double)topstep);

	for (int i = 0; i < nwImg.cols; i++) {
		Rect nwROI(topcur, 0, topwidth, nwImg.rows);
		Mat nwImg2 = nwImg(nwROI);

		int *smallPixelRow = new int[nwImg2.rows];

		for (int i = 0; i < nwImg2.rows; i++)
		{
			smallPixelRow[i] = nwImg2.cols - countNonZero(nwImg2.row(i));

			if (smallPixelRow[i] != 0) {
				yTop = i;
				break;
			}

		}

		circle(newout, Point(xTop, yTop), 1, Scalar(255, 0, 0), 5, 1);
		
		topcur += topstep;

		if (topcur >= nwImg.cols) {
			int resultant = (topcur-nwImg.cols)+1;
			topcur -= resultant;
		}
		
	
		xTop = topcur;
	}

	




	

	//namedWindow("nwImg2");
	//imshow("nwImg2", nwImg2);

		namedWindow("showMe2");
	imshow("showMe2", newout);

	//cout << "Start Line" << endl;
	/*
	for (int i = 0; i < totalLoop; i++) {
		int nwWidth = borderCheck - currentCheck;
		int nwHeight = nwImg.rows;

		Rect nwROI(currentCheck, 0, nwWidth, nwHeight);

		Mat nwImg2 = nwImg(nwROI);

		int xLine = currentCheck ;
		int yLine = 0;

		//loop, get rows from y line, if pixelrows not 0, pick y
		int *PixelRow = new int[nwImg2.rows];

		for (int i = 0 ; i < nwImg2.rows; i++)
		{
			PixelRow[i] = nwImg2.cols - countNonZero(nwImg2.row(i));
			

			if (PixelRow[i] != 0) {
				yLine = i;
				break;
			}

		}

		currentCheck = borderCheck;
		borderCheck += 1;

		myXLine[i] = xLine;
		myYLine[i] = yLine;
		//cout << nwImg2.cols << endl;
		//cout << "#####" << endl;
		//	cout << xLine << endl;
		//	cout << yLine << endl;
		//cout << "______" << endl;
		//circle(newout, Point(xLine, yLine), 3, Scalar(255, 0, 255), 1, 8);

	}

	int total = 0;
	int average = 0;

	for (int i = 0; i < totalLoop; i++) {
		total += myYLine[i];
	}

	average = (int)(total / totalLoop);
	average -= (0.2*average);

	int midValX = myXLine[(int)(totalLoop / 2)];
	int midValY = myYLine[(int)(totalLoop / 2)];



	for (int i = 0; i < totalLoop; i++) {
		if (myYLine[i] < average) {

			int slope = (midValY - myYLine[i]) / (midValX - myXLine[i]);

			myYLine[i] = average;


		}
	}
	*/

//	cout << "analyze" << endl;
//	for (int i = 0; i < totalLoop; i++) {
	//	circle(newout, Point(myXLine[i], myYLine[i]), 3, Scalar(0, 255, 0), 1, 8);
	//	cout << myYLine[i] << endl;
//	}
	//cout << "end of analyze" << endl;
	

	//cout << "_________" << endl;
//	namedWindow("showMe2");
	//imshow("showMe2", newout);

	return lineSlope;


	//	namedWindow("nwImg2");
	//	imshow("nwImg2", nwImg2);

}


MinMax GetMinMaxVal(int *Data, int sizeData) {

	MinMax minMax;
	minMax.min = 1000000;
	minMax.max = 0;

	for (int i = 0; i < sizeData; i++)
	{
		if (Data[i]<minMax.min) {
			minMax.min = Data[i];
		}
		if (Data[i]>minMax.max) {
			minMax.max = Data[i];
		}
	}

	return minMax;
}

float NormalizeMinMax(int data, int min, int max, int newMin, int newMax) {

	return ((float)((data-min)*(newMax-newMin))/(max-min))+newMin;
}

double findVariance(int a[], int n, double mean)
{


	int * b = new int[n];

	for (int i = 0; i < n; i++)
		b[i] = pow((a[i] - mean),2);

	int sum = 0;

	for (int i = 0; i < n; i++)
		sum+=b[i];

	int m = n - 1;

	return (double)sum / (double)m;
}

double findStdev(int a[], int n, double mean) {
	double data = findVariance(a, n, mean);

	return sqrt(data);
}

double findMean(int a[], int n)
{
	int sum = 0;
	for (int i = 0; i < n; i++)
		sum += a[i];

	return (double)sum / (double)n;
}

// Function for calculating median 
double findMedian(int a[], int n)
{
	// First we sort the array 
	sort(a, a + n);

	// check for even case 
	if (n % 2 != 0)
		return (double)a[n / 2];

	return (double)(a[(n - 1) / 2] + a[n / 2]) / 2.0;
}

double findKurtosis(int a[], int n, double mean, double stdev) {
	
	double multiple = (double)(n*(n + 1)) / (double)((n - 1)*(n - 2)*(n - 3));
//	cout << multiple << endl;
	double minus = (double)(3*(pow((n - 1),2))) / (double)((n - 2)*(n - 3));
//	cout << minus << endl;
	
	double *b = new double[n];

	for (int i = 0; i < n; i++) 
		b[i] = pow(((double)(a[i] - mean) / stdev), 4);
		//cout << b[i] << endl;
	
	double sum=0;

	for (int i = 0; i < n; i++)
		sum+= b[i];

	//1cout << sum << endl;

	return (multiple*sum)-minus;
}


double findSkewness(double mean, double median, double stdev) {

	return (double)(mean - median) / (double)stdev;
}

MomentInvariant MomentInvariantCalculate(int * arr, int sizeData) {
	MomentInvariant momentInvariant;
	momentInvariant.Kurtosis = 0;
	momentInvariant.Mean = 0;
	momentInvariant.Median = 0;
	momentInvariant.Skewness = 0;
	momentInvariant.StDev = 0;
	momentInvariant.Variance = 0;
	//sort()

	MinMax minMax = GetMinMaxVal(arr, sizeData);

	int * newarr = new int[sizeData];

	for (int i = 0; i < sizeData; i++)
		newarr[i] = NormalizeMinMax(arr[i], minMax.min, minMax.max, 1, 1000);

//	cout << "Normalize Data" << endl;
//	for (int i = 0; i < sizeData; i++)
//		cout << newarr[i] << endl;
	

	int a[] = { 1, 3, 4, 2, 7, 5, 8, 6 };
	int n = sizeof(a) / sizeof(a[0]);
	
	

	momentInvariant.Mean = findMean(newarr, sizeData);
	momentInvariant.Median = findMedian(newarr, sizeData);
	momentInvariant.StDev = findStdev(newarr, sizeData, momentInvariant.Mean) ;
	momentInvariant.Kurtosis = findKurtosis(newarr, sizeData, momentInvariant.Mean, momentInvariant.StDev);
	momentInvariant.Skewness = findSkewness(momentInvariant.Mean, momentInvariant.Median, momentInvariant.StDev);


//	cout << "Mean = " << momentInvariant.Mean << endl;
//	cout << "Median = " << momentInvariant.Median << endl;
//	cout << "Stdev = " << momentInvariant.StDev << endl;
//	cout << "Kurtosis = " << momentInvariant.Kurtosis << endl;
//	cout << "Skewness = " << momentInvariant.Skewness << endl;
//	cout << "Min = " << minMax.min << endl;
//	cout << "Max = " << minMax.max << endl;
	

	return momentInvariant;
}

Mat EnhanceAndPreprop(Mat img) {


	namedWindow("before");
	imshow("before", img);

	bitwise_not(img, img);

	ImageDehazer dehazer;

	if (!dehazer.LoadImage(img)) {
		//	std::cout << "Load image failed" << endl;
	}

	if (!dehazer.Dehaze(1, 0.1, 1)) {
		//	std::cout << "Dehaze failed" << endl;
	}

	img = dehazer.WriteImage();

	bitwise_not(img, img);

	namedWindow("after");
	imshow("after", img);


	//cvtColor(img, img, COLOR_BGR2GRAY);
	//threshold(img, img, Thresh(img), 255, THRESH_BINARY);

	namedWindow("beforecrop");
	imshow("beforecrop", img);


	return img;
}

Rect Recrop(Mat img, int* PixelRowCropped, int *PixelColCropped) {
	

	int xStart = 0;
	int xStop = img.cols;
	int yStart = 0;
	int yStop = img.rows;
	int width = img.cols;
	int height = img.rows;

	for (int i = 0; i < img.cols; i++)
	{
		if (PixelColCropped[i] != 0) {
			xStart = i;
			break;
		}
	}
	for (int i = img.cols - 1; i > 0; i--)
	{
		if (PixelColCropped[i] != 0) {
			xStop = i;
			break;
		}
	}

	for (int i = 0; i < img.rows; i++)
	{
		if (PixelRowCropped[i] != 0) {
			yStart = i;
			break;
		}
	}
	for (int i = img.rows - 1; i > 0; i--)
	{
		if (PixelRowCropped[i] != 0) {
			yStop = i;
			break;
		}
	}

	width = xStop - xStart;
	height = yStop - yStart;

	Rect myROI(xStart, yStart, width, height);

	return myROI;
}

Rect RoiDetect(Mat img, int* PixelRowCropped, int *PixelColCropped) {


	int xStart = 0;
	int xStop = img.cols;
	int yStart = 0;
	int yStop = img.rows;
	int width = img.cols;
	int height = img.rows;

	int iMax = 0;
	int rowVal = 0;

	for (int i = 0; i < img.rows; i++)
	{
		if (PixelRowCropped[i] > rowVal) {
			rowVal = PixelRowCropped[i];
			iMax = i;
		}
	}



	for (int i = iMax+1; i < img.rows; i++)
	{
		if (PixelRowCropped[i] != 0) {
			yStop = i;	
		}
		else
			break;
	}






	width = xStop - xStart;
	height = yStop - yStart;
	//cout << width << endl;
	//cout << height << endl;

	Rect myROI(xStart, yStart, width, height);

	return myROI;
}



Position CenterPupilDetection(Mat afterThresh) {

	////////center//////////

	int centertotalNumberPixel = afterThresh.rows * afterThresh.cols;
	int centertotalPixel = centertotalNumberPixel - countNonZero(afterThresh);


	int *centerPixelRow = new int[afterThresh.rows];
	int *centerPixelCol = new int[afterThresh.cols];
	int centerjxPixelRow = 0;
	int centerjyPixelCol = 0;
	int centerjRowpixelblack = 0;
	int centerjColPixelblack = 0;

	for (int i = 0; i < afterThresh.rows; i++)
	{
		centerPixelRow[i] = afterThresh.cols - countNonZero(afterThresh.row(i));
	}

	for (int j = 0; j < afterThresh.cols; j++)
	{
		centerPixelCol[j] = afterThresh.rows - countNonZero(afterThresh.col(j));
	}

	for (int i = 0; i < afterThresh.rows; i++)
	{
		centerjxPixelRow += (i + 1)*centerPixelRow[i];
		centerjRowpixelblack += centerPixelRow[i];
	}
	for (int j = 0; j < afterThresh.cols; j++)
	{
		centerjyPixelCol += (j + 1)*centerPixelCol[j];
		centerjColPixelblack += centerPixelCol[j];
	}


	float wx = 0;
	float wy = 0;

	if (centertotalPixel != 0)
	{
		wy = centerjxPixelRow / centerjRowpixelblack;
		wx = centerjyPixelCol / centerjColPixelblack;
		//////center of pupil/////////

	}

	Position pos;
	pos.pointX = wx;
	pos.pointY = wy;

	return pos;
}

void ReadDataset() {
	std::ifstream  data("dataset.csv");
	std::string line;
	std::vector<std::vector<double> > parsedCsv;
	while (std::getline(data, line))
	{
		std::stringstream lineStream(line);
		std::string cell;
		std::vector<double> parsedRow;
		while (std::getline(lineStream, cell, ','))
		{
			parsedRow.push_back(atof(cell.c_str()));
		}

		parsedCsv.push_back(parsedRow);
	}

	
	vector<vector<double> > convertdouble;

	for (int i = 0; i < parsedCsv.size(); i++) {
		for (int j = 0; j < parsedCsv[i].size(); j++) {
			cout << parsedCsv[i][j] << " ";
		}
		cout << endl;
	}

	//for (int i = 0; i < 2; i++) {
	//	for (int j = 0; j < 6; j++) {
	//		cout << convertdouble[i][j] << " ";
	//	}
	//	cout << endl;
//	}


	

}

void EmotionDetection(Mat src) {

	//this is core mechanics of Emotion Detection. Getting All feature

	

	////////////////////////////########CORE#########////////////////////////////

#pragma region PreProcessing
	Mat realImage = src.clone();
	
	//enchance Image, dehazing, grayscaling, thresholding
	Mat img = EnhanceAndPreprop(src);
	/////////////////////////////////////////////////////

	//filter noise/////////////////////////
	int *PixelRowCropped = new int[img.rows];
	PixelRowCropped = NoiseFilter(img, true);

	int *PixelColCropped = new int[img.cols];
	PixelColCropped = NoiseFilter(img, false);
	///////////////////////////////////////

	///Recrop
	Rect Roi = Recrop(img, PixelRowCropped, PixelColCropped);
	//Mat normalAfterCrop = realImage(Roi);
	Mat normalAfterCrop = realImage;
//	Mat BWAfterCrop = img(Roi);
	Mat BWAfterCrop = img;

	int *PixelRowCroppedNw = new int[BWAfterCrop.rows];
	PixelRowCroppedNw = NoiseFilter(BWAfterCrop, true);

	int *PixelColCroppedNw = new int[BWAfterCrop.cols];
	PixelColCroppedNw = NoiseFilter(BWAfterCrop, false);
#pragma endregion

#pragma region Pupil Detection
	//Detect Position of Pupil
	Mat BwpupilImg = BWAfterCrop.clone();
	Position centerPupil = CenterPupilDetection(BwpupilImg);
	circle(normalAfterCrop, Point(centerPupil.pointX, centerPupil.pointY), 3, Scalar(255, 255, 255), 1, 8);
	//////////////end of Pupil///////////////////////
#pragma endregion

#pragma region Degree Calculate
	Position posEyelidRightCrop = getPosEyesideRight(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop, centerPupil.pointY);
	Position posEyelidBotCrop = getPosEyelidBot(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop);

	Position posEyelidTopCrop = getPosEyelidTop(PixelRowCroppedNw, BWAfterCrop);

	//	Position posEyelidRightCrop = getPosEyesideRight(PixelRowCropped, PixelColCropped, BWAfterCrop, centerPupil.pointY);

	circle(normalAfterCrop, Point(posEyelidTopCrop.pointX, posEyelidTopCrop.pointY), 5, Scalar(0, 255, 0), 1, 8);
	circle(normalAfterCrop, Point(posEyelidBotCrop.pointX, posEyelidBotCrop.pointY), 5, Scalar(0, 255, 0), 1, 8);
	circle(normalAfterCrop, Point(posEyelidRightCrop.pointX, posEyelidRightCrop.pointY), 5, Scalar(0, 255, 0), 1, 8);

	int valc = posEyelidBotCrop.pointY - posEyelidTopCrop.pointY;

	int distanceBRx = posEyelidRightCrop.pointX - posEyelidBotCrop.pointX;
	int distanceBRy=0; 

	if (posEyelidBotCrop.pointY > posEyelidRightCrop.pointY) 
		distanceBRy = posEyelidBotCrop.pointY - posEyelidRightCrop.pointY;
	else
		distanceBRy = posEyelidRightCrop.pointY - posEyelidBotCrop.pointY;
		
	int distanceTRx = posEyelidRightCrop.pointX - posEyelidTopCrop.pointX;
	int distanceTRy = 0;

	if (posEyelidTopCrop.pointY > posEyelidRightCrop.pointY)
		distanceTRy = posEyelidTopCrop.pointY - posEyelidRightCrop.pointY;
	else
		distanceTRy = posEyelidRightCrop.pointY - posEyelidTopCrop.pointY;

	int valb = sqrt(pow(distanceBRx,2) + pow(distanceBRy, 2));

	int vala = sqrt(pow(distanceTRx,2) + pow(distanceTRy, 2));

	double N = (double)((vala*vala) + (valb*valb) - (valc*valc));
	double M = (double)N / (double)(2*vala*valb);
	double degree = acos(M)* 180.0 / PI;


	//int b = (int)(sqrt(pow((posEyelidRightCrop.pointX - posEyelidBotCrop.pointX), 2) + pow((posEyelidRightCrop.pointY - posEyelidBotCrop.pointY), 2)));
	//int a = posEyelidBotCrop.pointY - posEyelidTopCrop.pointY;
	//int c = sqrt((a*a) + (b*b));

	//double N = ((b*b) + (c*c) - (a*a));
	//N = N / (2 * b*c);

	//double degree = acos(N)* 180.0 / PI;
	cout << "degree" << endl;
	cout << degree << endl;

#pragma endregion

#pragma region Slope
	LineSlope lineSlope = LineDetection(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop, normalAfterCrop);
	cout << "Slope" << endl;
	cout <<lineSlope.LeftMidToMid <<endl;
	cout <<lineSlope.LeftToLeftMid <<endl;
	cout <<lineSlope.LeftToMid <<endl;
	cout <<lineSlope.LeftTopToMid <<endl;
	cout <<lineSlope.MidTopToRight <<endl;
	cout <<lineSlope.MidToRight <<endl;
	cout <<lineSlope.MidToRightMid <<endl;
	cout <<lineSlope.RightMidToRight <<endl;

#pragma endregion

#pragma region Moment

	MomentInvariant momentRow = MomentInvariantCalculate(PixelRowCroppedNw, BWAfterCrop.rows);
	cout << "MomentRow" << endl;
	cout << momentRow.Kurtosis << endl;
	cout << momentRow.Mean << endl;
	cout << momentRow.Median << endl;
	cout << momentRow.Skewness << endl;
	cout << momentRow.StDev << endl;
	MomentInvariant momentCol = MomentInvariantCalculate(PixelColCroppedNw, BWAfterCrop.cols);

#pragma endregion

#pragma region ShowImage
	//namedWindow("normalAfterCrop");
	imshow("normalAfterCrop", normalAfterCrop);

	namedWindow("BWAfterCrop");
	imshow("BWAfterCrop", BWAfterCrop);

#pragma endregion

	
	////////////////////////////######## End of CORE#########////////////////////////////



	//get elapsed time
	


}

////new main/////


Mat dst, detected_edges;
int lowThreshold = 0;
const int max_lowThreshold = 100;
const int ratio = 3;
const int kernel_size = 3;
const char* window_name = "Edge Map";

Ptr<CLAHE> pFilter;
int tilesize;
int cliplimit;

static void TSize_Callback(int pos, void* /*data*/)
{
	if (pos == 0)
		pFilter->setTilesGridSize(Size(1, 1));
	else
		pFilter->setTilesGridSize(Size(tilesize, tilesize));
}

static void Clip_Callback(int, void* /*data*/)
{
	pFilter->setClipLimit(cliplimit);
}

bool record = false;
bool hascontain = false;
char key;

int ptotaldyn = 0;
int pindexdyn = 0;
int *pupXdyn = new int[1000];
int *pupYdyn = new int[1000];

int *topXdyn = new int[1000];
int *topYdyn = new int[1000];

int *botXdyn = new int[1000];
int *botYdyn = new int[1000];
int *leftXdyn = new int[1000];
int *leftYdyn = new int[1000];
int *rightXdyn = new int[1000];
int *rightYdyn = new int[1000];

string emosi = "happy";
int timeloop = 4000;
auto starttime = chrono::high_resolution_clock::now();


bool crop = false;
bool cropIsSet = false;
Rect firstRoi;
void foo()
{
	int asciival;
	while (1) {
		key = _getch();
		asciival = key;

		if (key == 'a') {
			record = !record;
			starttime = chrono::high_resolution_clock::now();
		}

		if (key == 'c') {
			crop = true;
		}

		

		if (!record) {
			//timeloop = 4000;
			
		}

		if (record) {
			
		}

		if (key == '1')
			emosi ="happy";
		else if (key == '2')
			emosi = "sad";
		else if (key == '3')
			emosi = "suprised";
		else if (key == '4')
			emosi = "distingued";
		else if (key == '5')
			emosi = "angry";
		else if (key == '6')
			emosi = "fear";

	}
	// do stuff...
}

Mat prevImg;
int sudutnormal = 25;
int selisih = selisih;


int main() {
	thread first(foo);

	dynLine.leftX = 0;
	dynLine.leftY = 0;
	dynLine.rightX = 0;
	dynLine.rightY = 0;
	dynLine.topX = 0;
	dynLine.topY = 0;
	dynLine.botX = 0;
	dynLine.botY = 0;

	VideoCapture cap(1);
	//ReadDataset();
/*
	//stream untuk menulis file
	ofstream myfile;

	//membuka file,
	//jika file tidak ditemukan maka file akan otomatis dibuat
	myfile.open("dataset.csv", ios::app);

	cout << "OPERASI FILE 1" << endl;
	cout << "--------------" << endl;

	//fail() -> untuk memeriksa suatu kesalahan pada operasi file
	if (!myfile.fail())
	{
		//menulis ke dalam file
		myfile << "roll" << ", "
			<< "2name" << ", "
			<< "2math" << ", "
			<< "2phy" << ", "
			<< "2chem" << ", "
			<< "2bio"
			<< endl;
		myfile.close(); //menutup file
		cout << "Text telah ditulis ke dalam File" << endl;
	}
	else {
		cout << "File tidak ditemukan" << endl;
	}
	*/
	
	bool motionstate = false;
	cv::Mat diffImage;
	//EmotionDetection(src);
	while (true) {

		//count start time
		double t = (double)getTickCount();
		Mat src;
		//src = imread("D:/z6.jpg");

	//	Mat asli = src.clone();
	//	cvtColor(asli, asli, COLOR_BGR2GRAY);
	//	threshold(asli, asli, Thresh(asli), 255, THRESH_BINARY);
	//	namedWindow("asli");
	//	imshow("asli", asli);
	//	
		//cap >> src;
		//Mat src;
		cap >> src;

		Rect myROI(50, 100, 500, 250);
		src = src(myROI);

	//	if (src.empty())
	//	{
	//		std::cout << "!!! Failed imread(): image not found" << std::endl;
			// don't let the execution continue, else imshow() will crash.
	//	}

		Mat a = src.clone();
		//cap >> a;
		//Rect myROI(50, 100, 500, 250);
		//a = a(myROI);

		a = EnhanceAndPreprop(a);

		Mat dst = a;

		pyrDown(a, dst, Size(a.cols / 2, a.rows / 2));


		

		Mat pry = dst.clone();


		cvtColor(dst, dst, COLOR_BGR2GRAY);

		Ptr<CLAHE> clahe = createCLAHE();
		clahe->setClipLimit(1);
		clahe->setTilesGridSize(Size(2, 2));


		clahe->apply(dst, dst);

		medianBlur(dst, dst, 11);

	//	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));

		//morphologyEx(dst, dst, MORPH_CLOSE, element,Point(-1,1), 3);

		threshold(dst, dst, Thresh(dst), 255, THRESH_BINARY);

		//Mat im_floodfill = im_th.clone();
		floodFill(dst, cv::Point(0, 0), Scalar(255));


		
		

//		Mat BWAfterCrop = dst;
//		int *PixelRowCroppedNw = new int[dst.rows];
//		PixelRowCroppedNw = WithoutNoiseFilter(dst, true);

	//	int *PixelColCroppedNw = new int[dst.cols];
	//	PixelColCroppedNw = WithoutNoiseFilter(dst, false);

		int *PixelRow1 = new int[dst.rows];
		PixelRow1 = NoiseFilter(dst, true);

		int *PixelCol1 = new int[dst.cols];
		PixelCol1 = NoiseFilter(dst, false);

		
		if (crop) {
			firstRoi = RoiDetect(dst, PixelRow1, PixelCol1);
			crop = false;
		}
		if(!cropIsSet) {
			firstRoi = Rect(0, 0, dst.cols, dst.rows);
			cropIsSet = true;
		}
		
		
		
		dst = dst(firstRoi);
		pry = pry(firstRoi);

		Mat pupil = dst.clone();


#pragma region Degree Calculate

		Mat BWAfterCrop = dst;

		int *PixelRowCroppedNw = new int[dst.rows];
		PixelRowCroppedNw = NoiseFilter(dst, true);

		int *PixelColCroppedNw = new int[dst.cols];
		PixelColCroppedNw = NoiseFilter(dst, false);

		Position posEyelidRightCrop = getPosEyesideRight(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop, 0);
		Position posEyelidBotCrop = getPosEyelidBot(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop);

		Position posEyelidTopCrop = getPosEyelidTop(PixelRowCroppedNw, BWAfterCrop);

		//	Position posEyelidRightCrop = getPosEyesideRight(PixelRowCropped, PixelColCropped, BWAfterCrop, centerPupil.pointY);

		circle(pry, Point(posEyelidTopCrop.pointX, posEyelidTopCrop.pointY), 5, Scalar(0, 255, 0), 1, 8);
		circle(pry, Point(posEyelidBotCrop.pointX, posEyelidBotCrop.pointY), 5, Scalar(0, 255, 0), 1, 8);
		circle(pry, Point(posEyelidRightCrop.pointX, posEyelidRightCrop.pointY), 5, Scalar(0, 255, 0), 1, 8);

		int valc = posEyelidBotCrop.pointY - posEyelidTopCrop.pointY;

		int distanceBRx = posEyelidRightCrop.pointX - posEyelidBotCrop.pointX;
		int distanceBRy = 0;

		if (posEyelidBotCrop.pointY > posEyelidRightCrop.pointY)
			distanceBRy = posEyelidBotCrop.pointY - posEyelidRightCrop.pointY;
		else
			distanceBRy = posEyelidRightCrop.pointY - posEyelidBotCrop.pointY;

		int distanceTRx = posEyelidRightCrop.pointX - posEyelidTopCrop.pointX;
		int distanceTRy = 0;

		if (posEyelidTopCrop.pointY > posEyelidRightCrop.pointY)
			distanceTRy = posEyelidTopCrop.pointY - posEyelidRightCrop.pointY;
		else
			distanceTRy = posEyelidRightCrop.pointY - posEyelidTopCrop.pointY;

		int valb = sqrt(pow(distanceBRx, 2) + pow(distanceBRy, 2));

		int vala = sqrt(pow(distanceTRx, 2) + pow(distanceTRy, 2));

		double N = (double)((vala*vala) + (valb*valb) - (valc*valc));
		double M = (double)N / (double)(2 * vala*valb);
		double degree = acos(M)* 180.0 / PI;


		

		//int b = (int)(sqrt(pow((posEyelidRightCrop.pointX - posEyelidBotCrop.pointX), 2) + pow((posEyelidRightCrop.pointY - posEyelidBotCrop.pointY), 2)));
		//int a = posEyelidBotCrop.pointY - posEyelidTopCrop.pointY;
		//int c = sqrt((a*a) + (b*b));

		//double N = ((b*b) + (c*c) - (a*a));
		//N = N / (2 * b*c);

		//double degree = acos(N)* 180.0 / PI;
	//	cout << "degree" << endl;
	//	cout << degree << endl;

#pragma endregion


//Main Slope
		//slope right to the top
		double slopeTop = (double)(posEyelidRightCrop.pointY - posEyelidTopCrop.pointY) / (double)(posEyelidRightCrop.pointX - posEyelidTopCrop.pointX);
		double degreeSlopeTop = atan(slopeTop) * 180 / PI;

		//slope right to the bot
		double slopeBot = (double)(posEyelidRightCrop.pointY - posEyelidBotCrop.pointY) / (double)(posEyelidRightCrop.pointX - posEyelidBotCrop.pointX);
		double degreeSlopeBot = atan(slopeBot) * 180 / PI;

	//	cout << "Slope" << endl;
	//	cout<< degreeSlopeTop <<endl;
	//	cout<< degreeSlopeBot <<endl;


		//LineSlope lineSlope = LineDetection(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop, pry);
		
	
	//	cout << lineSlope.LeftToMid << endl;
	//	cout << lineSlope.MidToRight << endl;

	//	cout << lineSlope.LeftToLeftMid << endl;
	//	cout<< lineSlope.LeftMidToMid << endl;
	//	cout << lineSlope.MidToRightMid << endl;
	//	cout << lineSlope.RightMidToRight << endl;
		
		//LineSlope lineSlope2 = LineDetection2(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop, pry);



	//	int YslopeLeft = posEyelidRightCrop.pointY; // point left
		//int YslopeRight = posEyelidRightCrop.pointY; //point right

	//	int YslopeCenterBot = posEyelidBotCrop.pointX; //center bot
	//	int YslopeCenterTop = posEyelidTopCrop.pointX; //center top



		
	

#pragma region Slope
		LineSlope lineSlope = LineDetection(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop, pry);
		//LineSlope lineSlope2 = LineDetection2(PixelRowCroppedNw, PixelColCroppedNw, BWAfterCrop, pry);
	//	cout << "Slope" << endl;
	//	cout << lineSlope.LeftMidToMid << endl;
	//	cout << lineSlope.LeftToLeftMid << endl;
	//	cout << lineSlope.LeftToMid << endl;
	//	cout << lineSlope.LeftTopToMid << endl;
	//	cout << lineSlope.MidTopToRight << endl;
	//	cout << lineSlope.MidToRight << endl;
	//	cout << lineSlope.MidToRightMid << endl;
	//	cout << lineSlope.RightMidToRight << endl;

#pragma endregion


		Position pos = CenterPupilDetection(pupil);
		circle(pry, Point(pos.pointX, pos.pointY), 3, Scalar(0, 255, 0), 2, 1);

		
			MomentInvariant momentRow = MomentInvariantCalculate(PixelRowCroppedNw, BWAfterCrop.rows);
		//	cout << "MomentRow" << endl;
		//	cout << momentRow.Kurtosis << endl;
		//	cout << momentRow.Mean << endl;
		//	cout << momentRow.Median << endl;
		//	cout << momentRow.Skewness << endl;
		//	cout << momentRow.StDev << endl;
			MomentInvariant momentCol = MomentInvariantCalculate(PixelColCroppedNw, BWAfterCrop.cols);
		//	cout << "MomentCol" << endl;
		//	cout << momentCol.Kurtosis << endl;
		//	cout << momentCol.Mean << endl;
		//	cout << momentCol.Median << endl;
		//	cout << momentCol.Skewness << endl;
		//	cout << momentCol.StDev << endl;


		if (record) {
			
			auto end = chrono::high_resolution_clock::now();
			chrono::duration<double, milli>Elapsed = end - starttime;
			if (Elapsed.count() > 5000) {
				cout << "end" << endl;
				record = false;
			}



			if (!hascontain) {
				cout << "_______" << endl;
				cout << "Recording" << endl;

				

				pindexdyn = 0;

				pupXdyn[pindexdyn] = 0;
				pupYdyn[pindexdyn] = 0;

				topXdyn[pindexdyn] = 0;
				topYdyn[pindexdyn] = 0;

				botXdyn[pindexdyn] = 0;
				botYdyn[pindexdyn] = 0;

				leftXdyn[pindexdyn] = 0;
				leftYdyn[pindexdyn] = 0;

				rightXdyn[pindexdyn] = 0;
				rightYdyn[pindexdyn] = 0;

				selisih = sudutnormal- degree;

			}
		//	MomentInvariant momentRow = MomentInvariantCalculate(PixelRowCroppedNw, BWAfterCrop.rows);
		//	cout << "MomentRow" << endl;
		//	cout << momentRow.Kurtosis << endl;
		//	cout << momentRow.Mean << endl;
		//	cout << momentRow.Median << endl;
		//	cout << momentRow.Skewness << endl;
		//	cout << momentRow.StDev << endl;
		//	MomentInvariant momentCol = MomentInvariantCalculate(PixelColCroppedNw, BWAfterCrop.cols);
		//	cout << "MomentCol" << endl;
		//	cout << momentCol.Kurtosis << endl;
		//	cout << momentCol.Mean << endl;
		//	cout << momentCol.Median << endl;
		//	cout << momentCol.Skewness << endl;
		//	cout << momentCol.StDev << endl;
			hascontain = true;
			pindexdyn++;

			pupXdyn[pindexdyn] = pos.pointX+1;
			pupYdyn[pindexdyn] = pos.pointY+1;
			
			topXdyn[pindexdyn] = dynLine.topX+1;
			topYdyn[pindexdyn] = dynLine.topY+1;

			botXdyn[pindexdyn] = dynLine.botX+1;
			botYdyn[pindexdyn] = dynLine.botY+1;

			leftXdyn[pindexdyn] = dynLine.leftX+1;
			leftYdyn[pindexdyn] = dynLine.leftY+1;

			rightXdyn[pindexdyn] = dynLine.rightX+1;
			rightYdyn[pindexdyn] = dynLine.rightY+1;
			
			



		//	cout << "Pupil" << endl;
		//	cout << pos.pointX << endl;
		//	cout << pos.pointY << endl;
		//	cout << "Pupil" << endl;
		//	cout << key << endl;
		//	cout << "_______" << endl;
		}
		else {
			if (hascontain) {
				cout << "Stop Recording" << endl;

				cout << emosi << endl;

				pindexdyn++;

				pupXdyn[pindexdyn] = 0;
				pupYdyn[pindexdyn] = 0;

				topXdyn[pindexdyn] = 0;
				topYdyn[pindexdyn] = 0;

				botXdyn[pindexdyn] = 0;
				botYdyn[pindexdyn] = 0;

				leftXdyn[pindexdyn] = 0;
				leftYdyn[pindexdyn] = 0;

				rightXdyn[pindexdyn] = 0;
				rightYdyn[pindexdyn] = 0;

				
				/*
				cout << "Data Record" << endl;
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << pupXdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << pupYdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << topXdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << topYdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << botXdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << botYdyn[i] << endl;
				}
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << leftXdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << leftYdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << rightXdyn[i] << endl;
				}
				cout << "_______________" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << rightYdyn[i] << endl;
				}
				cout << "_______________" << endl;
				*/


				/*
				cout << "All PupilY" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << pupYdyn[i] << endl;
				}
				cout << "All Top" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << topXdyn[i] << endl;
				}
				cout << "All Bot" << endl;
				for (int i = 0; i < pindexdyn; i++)
				{
					cout << botYdyn[i] << endl;
				}
				*/

				MomentInvariant mPupilX = MomentInvariantCalculate(pupXdyn, pindexdyn);
				MomentInvariant mPupilY = MomentInvariantCalculate(pupYdyn, pindexdyn);
				MomentInvariant mTopX = MomentInvariantCalculate(topXdyn, pindexdyn);
				MomentInvariant mTopY = MomentInvariantCalculate(topYdyn, pindexdyn);
				MomentInvariant mBotX = MomentInvariantCalculate(botXdyn, pindexdyn);
				MomentInvariant mBotY = MomentInvariantCalculate(botYdyn, pindexdyn);
				MomentInvariant mLeftX = MomentInvariantCalculate(leftXdyn, pindexdyn);
				MomentInvariant mLeftY = MomentInvariantCalculate(leftYdyn, pindexdyn);
				MomentInvariant mRightX = MomentInvariantCalculate(rightXdyn, pindexdyn);
				MomentInvariant mRightY = MomentInvariantCalculate(rightYdyn, pindexdyn);

				MomentInvariant MI[10];
				MI[0] = mPupilX;
				MI[1] = mPupilY;
				MI[2] = mTopX;
				MI[3] = mTopY;
				MI[4] = mBotX;
				MI[5] = mBotY;
				MI[6] = mLeftX;
				MI[7] = mLeftY;
				MI[8] = mRightX;
				MI[9] = mRightY;

				//degree eye. top bot right
				//cout << selisih << endl;
				cout << (degree+selisih) << endl;

				//degree slope. 
				cout << lineSlope.LeftToMid << endl;
				cout << lineSlope.MidToRight << endl;
				cout << lineSlope.LeftTopToMid << endl;
				cout << lineSlope.MidTopToRight << endl;


				//cout << lineSlope.LeftToLeftMid << endl;
				//cout << lineSlope.LeftMidToMid << endl;
				//cout << lineSlope.MidToRightMid << endl;
				//cout << lineSlope.RightMidToRight << endl;

				//moment allimage
				//cout << "MomentRow" << endl;
				cout << momentRow.Kurtosis << endl;
				cout << momentRow.Mean << endl;
				cout << momentRow.Median << endl;
				cout << momentRow.Skewness << endl;
				cout << momentRow.StDev << endl;
				//cout << "MomentCol" << endl;
				cout << momentCol.Kurtosis << endl;
				cout << momentCol.Mean << endl;
				cout << momentCol.Median << endl;
				cout << momentCol.Skewness << endl;
				cout << momentCol.StDev << endl;

				for (int i = 0; i < 10; i++) {
					cout << MI[i].Mean << endl;
					cout << MI[i].Median << endl;
					cout << MI[i].StDev << endl;
					cout << MI[i].Skewness << endl;
					cout << MI[i].Kurtosis << endl;
				}

				hascontain = false;
				pindexdyn = 0;
			}
		}

		namedWindow("autos");
		imshow("autos", dst);
		namedWindow("src");
		imshow("src", src);
		
		namedWindow("pry");
		imshow("pry", pry);

		

		waitKey(3);
		t = (double)cvGetTickCount() - t;
	//	printf("=============Execution time per frame = %gms\n", t / ((double)cvGetTickFrequency()*1000.));
	}
}

