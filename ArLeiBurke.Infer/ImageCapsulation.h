#pragma once

#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

struct ImageCapsulation
{
	string Path;

	Mat Image;

	vector<float> InputTensor;

	vector<float> OutputTensor;


	ImageCapsulation(string Path)
	{
		Image = imread(Path);

	}

};



