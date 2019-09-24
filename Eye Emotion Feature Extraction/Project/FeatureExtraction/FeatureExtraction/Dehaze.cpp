#include "stdafx.h"
#include "Dehaze.h"
#include <queue>
#include <functional>
using namespace std;



bool ImageDehazer::LoadImage(Mat img) {

	m_Image = img;
	if (m_Image.empty()) {
		//cout << "Load image failed" << endl;
		return false;
	}

	m_Image.convertTo(m_DoubleImage, CV_64FC3);
	//cout << "Image is load." << endl;
	return true;
}



bool ImageDehazer::Dehaze(const int& _patchsize, const double& _t, const double& _w) {
	if (m_Image.empty())
		return false;
	DarkChannelImage_Create(_patchsize);
	m_AtmosLight = Atmospheric_Light_Estimate();
	TransMap_Create(_patchsize, _t, _w);
	return true;
}

Mat ImageDehazer::WriteImage() {
	//if (m_RecoveredImage.empty())
		
	return m_RecoveredImage;
}



void ShowImg(char *windowname, IplImage *image)
{
	cvNamedWindow(windowname, 1);
	cvShowImage(windowname, image);
	cvWaitKey(0);
	cvReleaseImage(&image);
	cvDestroyWindow(windowname);

}

void ImageDehazer::DarkChannelImage_Create(const int& _patchsize)
{


	m_DarkChannelImage.create(m_Image.rows, m_Image.cols, CV_8UC1);

	for (int i = 0; i < m_Image.rows; ++i) {
		for (int j = 0; j < m_Image.cols; ++j) {
			unsigned char DarkVal = 255;
			for (int m = i - _patchsize / 2; m <= i + _patchsize / 2; ++m) {
				for (int n = j - _patchsize / 2; n <= j + _patchsize / 2; ++n) {
					if (m < 0 || n < 0 || m >= m_Image.rows || n >= m_Image.cols)
						continue;

					DarkVal = std::min(std::min(m_Image.at<Vec3b>(m, n)[0], m_Image.at<Vec3b>(m, n)[1]), m_Image.at<Vec3b>(m, n)[2]);
				}
			}
			m_DarkChannelImage.at<uchar>(i, j) = DarkVal;
		}
	}
	imwrite("dark.jpg", m_DarkChannelImage);
	//cout << "The image dark channel is create!" << endl;
	return;
}

double ImageDehazer::Atmospheric_Light_Estimate()
{


	std::priority_queue<uchar, vector<uchar>, std::greater<uchar>> TopValues;

	//find out the 0.1% highest pixels in the dark channel
	int TopAmounts = m_DarkChannelImage.rows * m_DarkChannelImage.cols / 1000;
	double total = 0;
	for (int i = 0; i < m_DarkChannelImage.rows; i++)
	{
		for (int j = 0; j < m_DarkChannelImage.cols; j++)
		{
			uchar pixel = m_DarkChannelImage.at<uchar>(i, j);
			if (TopValues.size() < TopAmounts) {
				TopValues.push(pixel);
				total += pixel;
			}
			else {
				if (TopValues.top() >= pixel)
					continue;
				total -= TopValues.top();
				total += pixel;
				TopValues.pop();
				TopValues.push(pixel);
			}
		}
	}

	total /= TopAmounts;
	//cout << total << endl;
	return total;
}

void ImageDehazer::TransMap_Create(const int& _patchsize, const double& _t, const double& _w)
{
	cv::Mat TransmissionMap(m_Image.rows, m_Image.cols, CV_8UC1);
	m_RecoveredImage.create(m_Image.rows, m_Image.cols, CV_8UC3);

	for (int i = 0; i < m_Image.rows; i++)
	{
		for (int j = 0; j < m_Image.cols; j++)
		{

			double t = std::max(1 - (_w*m_DarkChannelImage.at<uchar>(i, j) / m_AtmosLight), _t);

			m_RecoveredImage.at<Vec3b>(i, j)[0] = static_cast<uchar>(std::min(((m_Image.at<Vec3b>(i, j)[0] - m_AtmosLight) / t + m_AtmosLight), 255.0));
			m_RecoveredImage.at<Vec3b>(i, j)[1] = static_cast<uchar>(std::min(((m_Image.at<Vec3b>(i, j)[1] - m_AtmosLight) / t + m_AtmosLight), 255.0));
			m_RecoveredImage.at<Vec3b>(i, j)[2] = static_cast<uchar>(std::min(((m_Image.at<Vec3b>(i, j)[2] - m_AtmosLight) / t + m_AtmosLight), 255.0));
		}
	}


	return;
}