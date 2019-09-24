#include <string>
#include <vector>
#include <iostream>
#include <cstdio>
#include <opencv/cv.h>
#include <opencv2\highgui.hpp>
#include <opencv/cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>
using namespace std;
using namespace cv;
class ImageDehazer {

public:
	bool LoadImage(Mat img);
	bool Dehaze(const int& _patchsize, const double& _t, const double& _w);
	Mat WriteImage();

private:
	IplImage * m_InputImage;
	cv::Mat m_Image;
	cv::Mat m_DoubleImage;
	cv::Mat m_DarkChannelImage;
	cv::Mat m_RecoveredImage;
	IplImage *InputImg;
	double m_AtmosLight;

	void DarkChannelImage_Create(const int& _patchsize);
	double Atmospheric_Light_Estimate();
	void TransMap_Create(const int& _patchsize, const double& _t, const double& _w);
};