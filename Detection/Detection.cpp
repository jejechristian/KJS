#include<iostream>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

int main()
{

	IplImage* img = cvLoadImage("tulaaang.jpg");

	//show the original image
	cvNamedWindow("Original");
	cvShowImage("Original", img);

	//converting the original image into grayscale
	IplImage* imgGrayScale = cvCreateImage(cvGetSize(img), 8, 1);
	cvCvtColor(img, imgGrayScale, CV_BGR2GRAY);

	//thresholding the grayscale image to get better results
	cvThreshold(imgGrayScale, imgGrayScale, 128, 255, CV_THRESH_BINARY);

	CvSeq* contours;  //hold the pointer to a contour in the memory block
	CvSeq* result;   //hold sequence of points of a contour
	CvMemStorage *storage = cvCreateMemStorage(0); //storage area for all contours

	//finding all contours in the image
	cvFindContours(imgGrayScale, storage, &contours, sizeof(CvContour), CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));

	//iterating through each contour
	int counter = 0;
	int initial = 2;
	cout << "Jumlah Tulang Initial = " << initial << endl;
	while (contours)
	{
		//obtain a sequence of points of contour, pointed by the variable 'contour'
		result = cvApproxPoly(contours, sizeof(CvContour), storage, CV_POLY_APPROX_DP, cvContourPerimeter(contours)*0.02, 0);

		//if there are 3  vertices  in the contour(It should be a triangle)
		if (result->total == 3)
		{
			//iterating through each point
			CvPoint *pt[3];
			for (int i = 0; i<3; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the triangle
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[2], *pt[0], cvScalar(0, 0, 255), 4);
			counter++;
			//cout << counter << " 3 ";
		}

		//if there are 4 vertices in the contour(It should be a quadrilateral)
		else if (result->total == 4)
		{
			//iterating through each point
			CvPoint *pt[4];
			for (int i = 0; i<4; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the quadrilateral
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 255, 255), 4);//kuning
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 0, 255), 4);//merah
			cvLine(img, *pt[2], *pt[3], cvScalar(255, 0, 255), 4);//magenta
			cvLine(img, *pt[3], *pt[0], cvScalar(255, 0, 0), 4);//biru
			cout << pt[1] << " " << pt[2] << endl;
			counter++;
			//cout << counter << " 4 ";

		}

		//if there are 7  vertices  in the contour(It should be a heptagon)
		else if (result->total == 7)
		{
			//iterating through each point
			CvPoint *pt[7];
			for (int i = 0; i<7; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the heptagon
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[2], *pt[3], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[3], *pt[4], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[4], *pt[5], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[5], *pt[6], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[6], *pt[0], cvScalar(0, 0, 255), 4);
			counter++;
			//cout << counter << " 7 ";
		}
		else if (result->total == 5)
		{
			//iterating through each point
			CvPoint *pt[5];
			for (int i = 0; i<5; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the heptagon
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 255, 255), 4);//kuning
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 0, 255), 4);//merah
			cvLine(img, *pt[2], *pt[3], cvScalar(255, 0, 255), 4);//magenta
			cvLine(img, *pt[3], *pt[4], cvScalar(255, 255, 255), 4);//putih
			cvLine(img, *pt[4], *pt[0], cvScalar(0, 255, 255), 4);//kuning
			cout << pt[0] << " " << pt[1] << endl;
			cout << pt[4] << " " << pt[0] << endl;

			counter++;

			//cout << counter << " 5 ";
		}
		else if (result->total == 6)
		{
			//iterating through each point
			CvPoint *pt[6];
			for (int i = 0; i<6; i++){
				pt[i] = (CvPoint*)cvGetSeqElem(result, i);
			}

			//drawing lines around the heptagon
			cvLine(img, *pt[0], *pt[1], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[1], *pt[2], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[2], *pt[3], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[3], *pt[4], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[4], *pt[5], cvScalar(0, 0, 255), 4);
			cvLine(img, *pt[5], *pt[0], cvScalar(0, 0, 255), 4);
			counter++;
			//cout << counter << " 6 ";
		}
		//obtain the next contour
		contours = contours->h_next;
		//	cout << counter<<" ";


	}
	cout << "Jumlah Tulang yang terdeteksi = " << counter << endl;
	if (counter >= initial)
	{
		cout << "Jenis Patah Tulang = Simple Bone Fracture";
	}
	//show the image in which identified shapes are marked   
	cvNamedWindow("Result");
	cvShowImage("Result", img);


	cvWaitKey(0); //wait for a key press

	//cleaning up
	cvDestroyAllWindows();
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&img);
	cvReleaseImage(&imgGrayScale);

	return 0;
}
