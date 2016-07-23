#include<iostream>
#include<opencv2/imgproc/imgproc.hpp>
#include "opencv2/highgui/highgui.hpp"
#include<opencv2/core/utility.hpp>
#include<stdio.h>
#include<opencv2/imgcodecs.hpp>

using namespace std;
using namespace cv;

RNG rng(12345);

void insertionSort(int window[])
{
	int temp, i, j;
	for (i = 0; i < 9; i++){
		temp = window[i];
		for (j = i - 1; j >= 0 && temp < window[j]; j--){
			window[j + 1] = window[j];
		}
		window[j + 1] = temp;
	}
}
                                                

// Computes the y component of the gradient vector
// at a given point in a image
// returns gradient in the y direction


void convolution(Mat src, Mat dst, float Kernel[][3])
{
	float sum;
	for (int x = 1; x < src.rows-1; x++){
		for (int y= 1; y < src.cols-1; y++){
			sum = 0.0;
			for (int k = -1; k <= 1; k++){
				for (int j = -1; j <= 1; j++){
					sum = sum + Kernel[j + 1][k + 1] * src.at<uchar>(x-j, y-k);
				}
			}
			dst.at<uchar>(x, y) = sum;
		}
	}
}

void convolutionSobel(Mat src, Mat dst, float kernelsatu[][3], float kerneldua[][3])
{
	float sumX, sumY, sumTotal,teta;
	for (int x = 1; x < src.rows-1; x++)
	{
		for (int y = 1; y < src.cols-1; y++)
		{
			sumX = 0.0;
			sumY = 0.0;
			sumTotal = 0.0;
			for (int k = -1; k <= 1; k++)
			{
				for (int j = -1; j <= 1; j++)
				{
					sumX = sumX + kernelsatu[j + 1][k + 1] * src.at<uchar>(x-j, y-k);
					sumY = sumY + kerneldua[j + 1][k + 1] * src.at<uchar>(x-j, y-k);
					sumTotal = sqrt(pow(sumX,2.0)+pow(sumY,2.0));
					teta = atan(abs(sumX) / abs(sumY));
					if (sumTotal > 255)
					{
						sumTotal = 255;
					}
					else if (sumTotal < 0)
					{
						sumTotal = 0;
					}
				}
			}
			dst.at<uchar>(x, y) = sumTotal;
		}
	}
}

int main()
{

	Mat src, dst;

	
	/// Load an image
	src = imread("complicated.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	if (!src.data)
	{
		return -1;
	}

	namedWindow("initial");
	imshow("initial", src);
	

	//namedWindow("gaussian");
	//imshow("gaussian", dst);
	//smoothing

	//laplacian
	float KernelLap[3][3] = {
		{ 1.0,  1.0, 1.0 },
		{ 1.0, -8.0, 1.0 },
		{ 1.0,  1.0, 1.0 }
	};

	Mat med, max;
	vector< uchar> vecImg(src.rows*src.cols);
	int arrHis[256] = {};
	int srcSize = src.rows*src.cols;
	int histSize = 256;
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	vecImg.assign((uchar*)src.datastart, (uchar*)src.dataend);
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < srcSize; j++)
		{
			if (vecImg[j] == i) {
				arrHis[i] += 1;
			}
		}
	}

	medianBlur(src, med, 5);
	bilateralFilter(src, max, 75, 150, 37);
	//namedWindow("Median");
	//imshow("Median", med);


	float KernelGauss1[5][5] = {
		{ 2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0, 2 / 159.0 },
		{ 4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0, 4 / 159.0 },
		{ 5 / 159.0, 12 / 159.0, 15 / 159.0, 12 / 159.0, 5 / 159.0 },
		{ 4 / 159.0,  9 / 159.0, 12 / 159.0,  9 / 159.0, 4 / 159.0 },
		{ 2 / 159.0,  4 / 159.0,  5 / 159.0,  4 / 159.0, 2 / 159.0 }
	};
	
	//gaussian filter
	float KernelGauss[3][3] = {
		{ 0.0    , 1 / 6.0, 0.0 },
		{ 1 / 6.0, 2 / 6.0, 0.0 },
		{ 0.0    , 1 / 6.0, 0.0 }
	};
	/*
	//weight smoothing filter
	float Kernel[3][3] = {
		{ 1 / 16.0, 2 / 16.0, 1 / 16.0 },
		{ 2 / 16.0, 4 / 16.0, 2 / 16.0 },
		{ 1 / 16.0, 2 / 16.0, 1 / 16.0 }
	};
	//mean
	float Kernel[3][3] = {
		{ 1 / 9.0, 1 / 9.0, 1 / 9.0 },
		{ 1 / 9.0, 1 / 9.0, 1 / 9.0 },
		{ 1 / 9.0, 1 / 9.0, 1 / 9.0 }
	};
	*/
	
	dst = med.clone();
	for (int y = 0; y < med.rows; y++)
		for (int x = 0; x < med.cols; x++)
			dst.at<uchar>(y, x) = 0.0;
	convolution(med, dst, KernelGauss);

	//namedWindow("gaussian");
	//imshow("gaussian", dst);
	
	Mat dst2;
	dst2 = dst.clone();
	for (int x = 0; x < dst.rows; x++)
		for (int y = 0; y < dst.cols; y++)
			dst2.at<uchar>(x, y) = 0.0;

	convolution(dst, dst2, KernelGauss);

	//namedWindow("gaussian 2");
	//imshow("gaussian 2", dst2);
	

	//----------MEDIAN------------
	/*
	int window[9];

	dst= src.clone();
	for (int y = 0; y < src.rows; y++)
		for (int x = 0; x < src.cols; x++)
			dst.at<uchar>(y, x) = 0.0;

	for (int y = 1; y < src.rows - 1; y++){
		for (int x = 1; x < src.cols - 1; x++){

			// Pick up window element

			window[0] = src.at<uchar>(y - 1, x - 1);
			window[1] = src.at<uchar>(y, x - 1);
			window[2] = src.at<uchar>(y + 1, x - 1);
			window[3] = src.at<uchar>(y - 1, x);
			window[4] = src.at<uchar>(y, x);
			window[5] = src.at<uchar>(y + 1, x);
			window[6] = src.at<uchar>(y - 1, x + 1);
			window[7] = src.at<uchar>(y, x + 1);
			window[8] = src.at<uchar>(y + 1, x + 1);

			// sort the window to find median
			insertionSort(window);

			// assign the median to centered element of the matrix
			dst.at<uchar>(y, x) = window[4];
		}
	}
	/*
	for (int a = 1; a < src.rows - 1; a++)
	{
		for (int b = 1; b < src.cols - 1; b++)
		{
			cout << window[a-1] << endl;
		}
	}
	
	namedWindow("median");
	imshow("median", dst);
	*/
	float kernelYSobel[3][3] =
	{
		{ 1.0, 2.0, 1.0 },
		{ 0.0, 0.0, 0.0 },
		{ -1.0, -2.0, -1.0 }
	};

	float kernelXSobel[3][3] =
	{
		{ -1.0, 0.0, 1.0 },
		{ -2.0, 0.0, 2.0 },
		{ -1.0, 0.0, 1.0 }
	};

	//int gx, gy, sum;
	Mat dst3;
	dst3 = dst2.clone();
	for (int x = 0; x < dst2.rows; x++)
		for (int y = 0; y < dst2.cols; y++)
			dst3.at<uchar>(x, y) = 0.0;
	convolutionSobel(dst2, dst3, kernelXSobel, kernelYSobel);
	//namedWindow("sobel");
	//imshow("sobel", dst3);
	
	Mat thres1;
	thres1 = dst3.clone();
	threshold(dst3, thres1, 57, 255, THRESH_BINARY);
	//namedWindow("thresholding1");
	//imshow("thresholding1", thres1);
	Mat hue,hue2;
	hue = dst3.clone();
	//cvtColor(dst, gray, COLOR_BGR2GRAY);

	// Create a window
	//namedWindow("Edge map", 1);
	// create a toolbar
	//createTrackbar("Canny threshold", "Edge map", &edgeThresh);
	// Show the image
	Canny(hue, hue, 54, 100 * 3, 3);
	// Wait for a key stroke; the same function arranges events processing
	
	//Canny(dst3, dst4, 70, 250, 3);
	//cvtColor(dst3, color_dst, CV_BGR2GRAY);
	
	//createTrackbar("Canny threshold", "Edge map", &edgeThresh, 100, onTrackbar);
	//namedWindow("Canny Edge", CV_WINDOW_AUTOSIZE);
	//imshow("Canny Edge", hue);

	Mat thres2;
	thres2 = hue.clone();
	threshold(thres2, thres2, 57, 255, THRESH_BINARY);
	//namedWindow("thresholding2");
	//imshow("thresholding2", thres2);

	Mat thres3 = thres1.clone();
	Mat dsttt;
	Mat dilate_kernel(9, 9, CV_8UC1);
	dilate(thres3, dsttt, dilate_kernel);
	
	imshow("Thres3", dsttt);


	threshold(dsttt, dsttt, 57, 255, THRESH_BINARY_INV);
	imshow("invers", dsttt);
	//opening
	int morph_elem = 0;
	int morph_size = 23;
	int morph_operator = 0;
	int const max_operator = 4;
	int const max_elem = 2;
	int const max_kernel_size = 21;
	int operation = morph_operator + 2;

	Mat element = getStructuringElement(morph_elem, Size(2 * morph_size + 1, 2 * morph_size + 1), Point(morph_size, morph_size));
	Mat open = dsttt.clone();
	Mat dstttt;
	morphologyEx(open, dstttt, operation, element);


	imshow("opening", dstttt);
	threshold(dstttt, dstttt, 57, 255, THRESH_BINARY_INV);
	imshow("inver2", dstttt);
	Mat thres;
	thres = hue.clone();

	//adaptiveThreshold(thres, thres, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 1);
	threshold(thres, thres, 128, 255, THRESH_BINARY);
	namedWindow("thresholding");
	imshow("thresholding", thres);
	
	Mat hasil;
	hasil = thres.clone();
	for (int x = 0; x < dst.rows; x++)
		for (int y = 0; y < dst.cols; y++)
			hasil.at<uchar>(x, y) = thres1.at<uchar>(x, y) + thres.at<uchar>(x, y);
	
	namedWindow("hasil");
	imshow("hasil", hasil);
	/*
	Mat cann = dst3.clone();
	Mat dstt;
	Mat dilate_kernel(3, 3, CV_8UC1);
	dilate(cann, dstt, dilate_kernel);

	namedWindow("dilation");
	imshow("dilation", dstt);*/
	/*
	Mat img5 = thres.clone();
	Mat imgOutput = img5.clone();


	Rect bounding_rect;
	int largest_area = 430;
	int largest_contour_index = 0;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	findContours(imgOutput, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(imgOutput.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i], false);  //  Find the area of contour
		if (area < largest_area)
		{
			//contours.erase(contours.begin() + i);
			largest_area = area;
			largest_contour_index = i;
			bounding_rect = boundingRect(contours[i]);
		}
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, Point());
	}
	imshow("Contours", drawing);
	*/
	waitKey(0);


	return 0;
}
