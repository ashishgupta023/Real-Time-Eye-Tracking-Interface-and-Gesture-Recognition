double threshold()
{
	double img[6][6];
	int i,j;
	char *str="";
	IplImage *imgResult;
	char *form=".jpg";
	for(i=0;i<6;i++)
	{
		IplImage *src=0,*temp=0;
		char c1[10]="eye";
		sprintf(c1,"%s%d",c1,i);
		sprintf(c1,"%s%s",c1,".jpg");
		if((src=cvLoadImage(c1,0))==0)
			printf("Error loading source");
		printf("Source: %s\n",c1);
		for(j=0;j<6;j++)
		{	
			char c2[10]="eye";
		sprintf(c2,"%s%d",c2,j);
		sprintf(c2,"%s%s",c2,".jpg");
		printf("Template:%s\n",c2);
			if((temp=cvLoadImage(c2,0))==0)
				printf("Error loading template\n");
			
		imgResult = cvCreateImage(cvSize(src->width-temp->width+1,src->height-temp->height+1), IPL_DEPTH_32F, 1);
cvZero(imgResult);
			cvMatchTemplate(src, temp, imgResult,CV_TM_CCORR_NORMED );
			// cvNormalize(imgResult,imgResult,1,0,CV_MINMAX);
			double min_val, max_val;
			CvPoint min_loc, max_loc;
			cvMinMaxLoc(imgResult, &min_val, &max_val, &min_loc, &max_loc);
			img[i][j]=max_val;
			printf("minval=%f\n",min_val);
			printf("maxval=%f\n",max_val);
			
			cvReleaseImage(&imgResult);
	cvReleaseImage(&temp);
	
		}
		cvReleaseImage(&src);
		
	}
	double min_of_max=img[0][0];
	double max_of_max=img[0][1];
	for(int i=0;i<6;i++)
	{
			for(int j=0;j<6;j++)
			{
				if(j==i)
					;
				else
				{
					if(img[i][j]<min_of_max)
						min_of_max=img[i][j];
					if(img[i][j]>max_of_max)
					{
						if(img[i][j]==1)
							;
						else
						max_of_max=img[i][j];
					}

				}
			}
	}
	printf("<<minofmax: %f>>",min_of_max);
	printf("<<maxofmax: %f>>",max_of_max);
	
	//min_of_max=min_of_max-(0.05*min_of_max);
	max_of_max=max_of_max-(0.2*max_of_max);
	return max_of_max;

	
}