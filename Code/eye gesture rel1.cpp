

#include "stdafx.h"
#include "cv.h"
#include "highgui.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include <time.h>
#include <ctype.h>
#include<time.h>

#include "threshold.h"
#define USEROI 0
#define CAMSHIFT 0
int main_func();
IplImage *framecopy,*framecur;
double eye_threshold;
typedef struct {
  IplImage* hsv;     //input image converted to HSV
  IplImage* hue;     //hue channel of HSV image
  IplImage* mask;    //image for masking pixels
  IplImage* prob;    //face probability estimates for each pixel

  CvHistogram* hist; //histogram of hue in original face image

  CvRect prev_rect;  //location of face in previous frame
  CvBox2D curr_box;  //current face location estimate
} TrackedObj ;


//IplImage* imgOriginal=0;


char* cascade_name =
    "haarcascade_frontalface_default.xml";

//int x,y,width,height=0;

//HaarFaceDetect
//void haardetect( IplImage* imgOriginal,IplImage* frame  )
void cleanup (char* name,
              CvHaarClassifierCascade* cascade,
              CvMemStorage* storage) 
{
	//cleanup and release resources
	cvDestroyWindow(name);
	if(cascade) cvReleaseHaarClassifierCascade(&cascade);
	if(storage) cvReleaseMemStorage(&storage);
}

CvRect* haardetect( IplImage* frame,CvHaarClassifierCascade* cascade,
                     CvMemStorage* storage  )
{
	CvRect* rect=0;
	// int scale = 1;
   // CvPoint pt1, pt2,pt3,pt4,pt5,pt6,pt7,pt8;
   // int i;
	
	if(cascade)

	{
		CvSeq* faces = cvHaarDetectObjects( frame, cascade, storage,
											1.1, 2, CV_HAAR_DO_CANNY_PRUNING,
											cvSize(40, 40) );


		//  for( i = 0; i < (faces ? faces->total : 0); i++ )
		// {
		if(faces && faces->total)
			rect = (CvRect*) cvGetSeqElem(faces, 0);
	}
	return rect;
           

            
       /*     pt1.x = r->x*scale;
			x=pt1.x;
			width=r->width;
            pt2.x = (r->x+r->width)*scale;
            pt1.y = r->y*scale;
			y=pt1.y;
			height=r->height;
            pt2.y = (r->y+r->height)*scale;
pt3.x=pt1.x;
pt3.y=pt2.y;
pt4.x=pt2.x;
pt4.y=pt1.y;
pt5.x=(pt1.x+pt3.x)/2;
pt5.y=(pt1.y+pt3.y)/2;
pt6.x=(pt2.x+pt4.x)/2;
pt6.y=(pt2.y+pt4.y)/2;
pt7.x=(pt1.x+pt4.x)/2;
pt7.y=(pt1.y+pt4.y)/2;
pt8.x=(pt2.x+pt3.x)/2;
pt8.y=(pt2.y+pt3.y)/2;
if((pt6.x-pt5.x)!=0 && (pt8.x-pt7.x)!=0 )
{
float m1=(pt6.y-pt5.y)/(pt6.x-pt5.x);
float m2=(pt8.y-pt7.y)/(pt8.x-pt7.x);
float tanangle=(m2-m1)/(1+m1*m2);
}

           cvLine(frame,pt5,pt6,CV_RGB(255,255,0),3,8,0);
		   cvLine(frame,pt7,pt8,CV_RGB(255,255,0),3,8,0);
            cvRectangle( frame, pt1, pt2, CV_RGB(255,0,0), 3, 8, 0 );
					
					
  
    

  IplImage* imgTemplate = cvLoadImage("i0.jpg",0);
	IplImage* imgTemplate1 = cvLoadImage("i1.jpg",0);
	//IplImage* imgTemplate=cvCreateImage(cvSize(imgTemp->width,imgTemp->height),IPL_DEPTH_8U,1);
//IplImage* imgTemplate1=cvCreateImage(cvSize(imgTemp1->width,imgTemp1->height),IPL_DEPTH_8U,1);
	//cvCvtColor( imgTemp, imgTemplate, CV_RGB2GRAY );
	//cvCvtColor( imgTemp1, imgTemplate1, CV_RGB2GRAY );
CvRect rect = cvRect(x,y,width,height);
 cvSetImageROI(imgOriginal,rect);
	cvSetImageROI(frame,rect);
   IplImage* imgResult = cvCreateImage(cvSize(width-imgTemplate->width+1,height-imgTemplate->height+1), IPL_DEPTH_32F, 1);
cvZero(imgResult);

	cvMatchTemplate(imgOriginal, imgTemplate, imgResult,CV_TM_CCORR_NORMED);


	double min_val=0, max_val=0;
    CvPoint min_loc, max_loc;

    cvMinMaxLoc(imgResult, &min_val, &max_val, &min_loc, &max_loc);

	cvRectangle(frame, max_loc, cvPoint(max_loc.x+imgTemplate->width, max_loc.y+imgTemplate->height), cvScalar(0), 1);

	

	cvMatchTemplate(imgOriginal, imgTemplate1, imgResult, CV_TM_CCORR_NORMED);
	cvMinMaxLoc(imgResult, &min_val, &max_val, &min_loc, &max_loc);
	
	
	cvRectangle(frame, max_loc, cvPoint(max_loc.x+imgTemplate1->width, max_loc.y+imgTemplate1->height), cvScalar(0), 1);
	 cvResetImageROI(imgOriginal);
	 cvResetImageROI(frame);*/
		//}
}

//updatehue
void updatehue (const IplImage* image, TrackedObj* obj) {
	//limits for calculating hue
	int vmin = 65, vmax = 256, smin = 55;

	//convert to HSV color model
	cvCvtColor(image, obj->hsv, CV_BGR2HSV);

	//mask out-of-range values
	cvInRangeS(obj->hsv,                               //source
			 cvScalar(0, smin, MIN(vmin, vmax), 0),  //lower bound
			 cvScalar(180, 256, MAX(vmin, vmax) ,0), //upper bound
			 obj->mask);                             //destination

	//extract the hue channel, split: src, dest channels
	cvSplit(obj->hsv, obj->hue, 0, 0, 0 );
}

//CreateFaceObject
TrackedObj* makeobject (IplImage* image, CvRect* region)
{
	TrackedObj* obj;

	//allocate memory for tracked object struct
	if((obj = (TrackedObj*)malloc(sizeof(*obj))) != NULL) {
		//create-image: size(w,h), bit depth, channels
		obj->hsv  = cvCreateImage(cvGetSize(image), 8, 3);
		obj->mask = cvCreateImage(cvGetSize(image), 8, 1);
		obj->hue  = cvCreateImage(cvGetSize(image), 8, 1);
		obj->prob = cvCreateImage(cvGetSize(image), 8, 1);

		int hist_bins = 30;           //number of histogram bins
		float hist_range[] = {0,180}; //histogram range
		float* range = hist_range;
		obj->hist = cvCreateHist(1,             //number of hist dimensions
								 &hist_bins,    //array of dimension sizes
								 CV_HIST_ARRAY, //representation format
								 &range,        //array of ranges for bins
								 1);            //uniformity flag
	}
	updatehue(image, obj);

	float max_val = 0.f;

	//create a histogram representation for the face
	cvSetImageROI(obj->hue, *region);
	cvSetImageROI(obj->mask, *region);
	cvCalcHist(&obj->hue, obj->hist, 0, obj->mask);
	cvGetMinMaxHistValue(obj->hist, 0, &max_val, 0, 0 );
	cvConvertScale(obj->hist->bins, obj->hist->bins,
				 max_val ? 255.0/max_val : 0, 0);
	cvResetImageROI(obj->hue);
	cvResetImageROI(obj->mask);

	//store the previous face location
	obj->prev_rect = *region;

	return obj;
}

//DestroyFaceObject
void destroyobject (TrackedObj* obj) 
{
	cvReleaseImage(&obj->hsv);
	cvReleaseImage(&obj->hue);
	cvReleaseImage(&obj->mask);
	cvReleaseImage(&obj->prob);
	cvReleaseHist(&obj->hist);

	free(obj);
}

//CamshiftTracking

CvBox2D camshifttracking (IplImage* image, TrackedObj* obj) 
{
	CvConnectedComp components;

	//create a new hue image
	updatehue(image, obj);

	//create a probability image based on the face histogram
	cvCalcBackProject(&obj->hue, obj->prob, obj->hist);
	cvAnd(obj->prob, obj->mask, obj->prob, 0);

	//use CamShift to find the center of the new face probability
	cvCamShift(obj->prob, obj->prev_rect,
			 cvTermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1),
			 &components, &obj->curr_box);

	//update face location and angle
	obj->prev_rect = components.rect;
	obj->curr_box.angle = -obj->curr_box.angle;
	return obj->curr_box;
}

void TrackEye(IplImage *frame,int x,int y,float width,float height)
{
	IplImage* frame_gray=0;IplImage* imgResult=0;
	CvRect rect123;
	rect123= cvRect(x,y,width,height);
	IplImage* imgTemplate = cvLoadImage("i0.jpg",0);
	IplImage* imgTemplate1 = cvLoadImage("i1.jpg",0);

	frame_gray=cvCreateImage(cvSize(frame->width,frame->height),IPL_DEPTH_8U,1);
	cvCvtColor( frame, frame_gray, CV_RGB2GRAY );
if (USEROI)
	{
		

		if (x<0)
			x = 0;
		if (y<0)
			y = 0;
		if (x+width > frame->width)
			width = frame->width - x - 1;
		if (y+height > frame->height)
			height = frame->height - y - 1;
		if (width < imgTemplate->width || height < imgTemplate->height)
			return;
		if (x<0 || y<0 || x+width > frame->width || y+height > frame->height)
		{
			printf("%d %f %d %f %d %d\n",x, width, y, height, frame->width, frame->height);
		}
		cvRectangle(frame, cvPoint(x, y), cvPoint(x+width, y+height), cvScalar(0), 1);
		rect123= cvRect(x,y,width,height);
		cvSetImageROI(frame_gray,rect123);
		imgResult = cvCreateImage(cvSize(width-imgTemplate->width+1,height-imgTemplate->height+1), IPL_DEPTH_32F, 1);
	}
else
{
	imgResult = cvCreateImage(cvSize(frame->width-imgTemplate->width+1,frame->height-imgTemplate->height+1), IPL_DEPTH_32F, 1);
	cvZero(imgResult);
}
	cvMatchTemplate(frame_gray, imgTemplate, imgResult,CV_TM_CCORR_NORMED);


	double min_val=0, max_val=0;
    CvPoint min_loc1, max_loc1,min_loc2, max_loc2;

    cvMinMaxLoc(imgResult, &min_val, &max_val, &min_loc1, &max_loc1);
	if(max_val<eye_threshold)
		main_func();
		
	if (USEROI)
	{
		max_loc1.x += x;
		max_loc1.y += y;
	}

	

	

	cvMatchTemplate(frame_gray, imgTemplate1, imgResult, CV_TM_CCORR_NORMED);
	cvMinMaxLoc(imgResult, &min_val, &max_val, &min_loc2, &max_loc2);
	if(max_val<eye_threshold)
		main_func();
		 
	if (USEROI)
	{
		max_loc2.x += x;
		max_loc2.y += y;
	}

	if(abs(max_loc1.y-max_loc2.y)==0)
	{
	cvRectangle(frame, max_loc1, cvPoint(max_loc1.x+imgTemplate->width, max_loc1.y+imgTemplate->height), cvScalar(0), 1);
	cvRectangle(frame, max_loc2, cvPoint(max_loc2.x+imgTemplate1->width, max_loc2.y+imgTemplate1->height), cvScalar(0), 1);
	}
	if(abs(max_loc1.y-max_loc2.y)>0 && abs(max_loc1.y-max_loc2.y)<5)
	{
	cvRectangle(frame, max_loc1, cvPoint(max_loc1.x+imgTemplate->width, max_loc1.y+imgTemplate->height), CV_RGB(0,0,255), 1);
	cvRectangle(frame, max_loc2, cvPoint(max_loc2.x+imgTemplate1->width, max_loc2.y+imgTemplate1->height), CV_RGB(0,0,255), 1);
	}
	if(abs(max_loc1.y-max_loc2.y)>=5 && abs(max_loc1.y-max_loc2.y)<10)
	{
	cvRectangle(frame, max_loc1, cvPoint(max_loc1.x+imgTemplate->width, max_loc1.y+imgTemplate->height), CV_RGB(0,192,0), 1);
	cvRectangle(frame, max_loc2, cvPoint(max_loc2.x+imgTemplate1->width, max_loc2.y+imgTemplate1->height), CV_RGB(0,192,0), 1);
	}
	if(abs(max_loc1.y-max_loc2.y)>=10 && abs(max_loc1.y-max_loc2.y)<15)
	{
	cvRectangle(frame, max_loc1, cvPoint(max_loc1.x+imgTemplate->width, max_loc1.y+imgTemplate->height), CV_RGB(0,192,255), 1);
	cvRectangle(frame, max_loc2, cvPoint(max_loc2.x+imgTemplate1->width, max_loc2.y+imgTemplate1->height), CV_RGB(0,192,255), 1);
	}

	if(USEROI)
	{
	cvResetImageROI(frame_gray);
	}
	cvReleaseImage(&frame_gray);

cvReleaseImage(&imgTemplate);

cvReleaseImage(&imgTemplate1);
	//cvResetImageROI(frame_gray);
	//cvResetImageROI(frame); 
}

//Getcopyofcapture
IplImage* getcapturecopy(CvCapture* capture)
{
	int boo=!cvGrabFrame( capture );
    if( boo)
        return 0;
    framecur = cvRetrieveFrame( capture );

    if( !framecur )
        return 0;
		
	framecopy = cvCreateImage( cvGetSize(framecur),IPL_DEPTH_8U, framecur->nChannels );

     // if(!frame_gray)
		//	frame_gray = cvCreateImage( cvSize( frame->width, frame->height ), IPL_DEPTH_8U, 1 );
       
	//	cvCvtColor( frame, frame_gray, CV_RGB2GRAY );

      // if( !imgOriginal )
	//imgOriginal = cvCreateImage( cvSize(frame_gray->width,frame_gray->height), IPL_DEPTH_8U, frame_gray->nChannels );

	//cvCopy( frame_gray,imgOriginal , 0 );
	cvCopy(framecur,framecopy,NULL);
	framecopy->origin = framecur->origin;
	if (framecopy->origin == 1) {
		cvFlip(framecopy, 0, 0);
		framecopy->origin = 0;
	}
	return framecopy;
}
int main_func()
{
	
	
	time_t t1;
	time_t t2;
	CvHaarClassifierCascade* cascade=0;
	CvRect* facerect = 0;
	CvMemStorage* storage=0;
	cascade=(CvHaarClassifierCascade*)cvLoad( cascade_name, 0, 0, 0 );
	if( !cascade )
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        return -1;
    }
	//double width1, height1, width2, height2;
	
	storage = cvCreateMemStorage(0);
	cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);

	CvCapture* capture = cvCaptureFromCAM(0);
	IplImage* frame = 0;
	if( capture )
    {
        for(;;)
        {
			loop:
			t1=time(NULL);
			frame=getcapturecopy(capture);
			if(!frame)
			   break;
			//haardetect( imgOriginal,frame_copy );
					   
			facerect=haardetect( frame,cascade,storage );
			if(facerect)
			{
				cvRectangle(frame, cvPoint(facerect->x, facerect->y), cvPoint(facerect->x+facerect->width, facerect->y+facerect->height), cvScalar(0), 1);
				if(!CAMSHIFT)
				{
					eye_threshold=threshold();
					TrackEye(frame,facerect->x,facerect->y,facerect->width,facerect->height);
				}
			}
			cvShowImage("Result",frame);
			if(facerect && CAMSHIFT)
				break;
				if ((char)27 == cvWaitKey(10)) {
					cvReleaseCapture(&capture);
					cleanup("Result", cascade, storage);
					exit(0);
			}	
		}//closefor

			TrackedObj* faceobject = makeobject(frame, facerect);
			CvBox2D facebox;

		for(;;)
		{//startingfor
			frame=getcapturecopy(capture);
			facebox=camshifttracking(frame, faceobject);
			if (faceobject->prev_rect.height <=0 || faceobject->prev_rect.width <=0)
				goto loop;
			facebox.angle += 90;
			cvEllipseBox(frame, facebox, CV_RGB(255,0,0), 3, CV_AA, 0);
			float height=facebox.size.height;
			float width=facebox.size.width;
			CvPoint2D32f center=facebox.center;
			CvPoint p1,p2;
			
			p2.x=center.x-(width/2);
			p2.y=center.y+(height/2);
			

			
			eye_threshold=threshold();
			TrackEye(frame,p2.x,p2.y,width,height);
			cvShowImage("Result", frame);

			if ((char)27 == cvWaitKey(10)) break;
			t2=time(NULL);
			//printf("%d time",t2-t1);
if((t2-t1)>10)
{ 
	goto loop;
}
		}//endingfor

		destroyobject(faceobject);
		cvReleaseCapture(&capture);
	}//closeif
	cleanup("Result", cascade, storage);

}

int _tmain(int argc, _TCHAR* argv[])

{ 
	main_func();
	return 1;

}

