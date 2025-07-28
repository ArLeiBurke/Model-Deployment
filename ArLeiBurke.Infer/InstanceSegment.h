#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

#include <windows.h>
#include <string>

#include <iostream>


using namespace cv;
using namespace std;
using namespace Ort;


class InstanceSegment
{

	// 参考文章！！！ https://blog.csdn.net/yangyu0515/article/details/142057357  这个是部署分割相关的！！！


public:
	InstanceSegment();
	Mat transformation(const Mat& image, const Size& targetSize, const Scalar& mean, const Scalar& std);



};


