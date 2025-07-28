#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#include <direct.h>

#include <windows.h>
#include <string>
#include <iostream>

#include "InstanceSegment.h"
#include "ObjectDetection.h"

#include <filesystem>  // C++17


using namespace cv;
using namespace std;
using namespace Ort;

using namespace cv::dnn;
namespace fs = std::filesystem;


void main(int args)
{

	ObjectDetection od;
	od.ClassFilePath = "FuChiOCR.txt";
	od.ReadClasses();
	string rootpath = "C:\\Users\\Administrator\\Desktop\\Pic\\temp";
	od.ReadImages(rootpath);
	od.PreProcessing();

	od.Inference();


}



