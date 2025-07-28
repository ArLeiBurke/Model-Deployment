#pragma once

#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <provider_options.h>
#include <onnxruntime_cxx_api.h>
#include <filesystem>  // C++17

#include <random>		// 生成随机数

#include <optional>	// 延时构造

#include "onnxruntime_cxx_api.h"



using namespace cv;
using namespace std;
using namespace Ort;
using namespace cv::dnn;

using namespace filesystem;


class ObjectDetection
{
	/*
	* 
	

	用 Onnx Runtime 去推理模型的时候 都需要去执行哪些 预处理相关的操作呢？？？
		1.Resize Image Size
		2.BGR -> RGB ！！！因为 OpneCV 是以 BGR的形式读取图片的！！
		3.数据类型转换 + 归一化！！ 将uint8(0-255) 转换成 float32,并且将对应的数据归一化到[0,1]
		4.通道排列 HWC -> CHW ！！！因为 Onnx 模型默认使用的是 CHW 格式！！！而OpenCV加载图像的时候，默认格式是 HWC！！！


	Onnx Runtime 多Batch 部署，我传递给模型的 Tensor 的 Shape 是 [X,3,640,640] 这个样子的 ！！！我通过 Session.Run(XXX) 得到的Tensor 的 Shape 是什么样子的呢？？？？这个Tensor 应该怎么解析呢？？？

	Onnx Runtime 多Batch 部署的时候，对 输入 Tensor 都有什么要求么？？？必须是连续的？

	导出的模型如果输入是 [1,3,640,640] 的话 是不支持 多batch部署的，，除非是 动态batch 或者是 [X,3,640,640] 样式的，才可以支持动态batch ！！！！

	将.pt格式的模型文件 转换成 .onnx 格式的模型文件 会有一定成都的精度损失(这里的精度损失是.pt转换成.onnx 的时候导致的，这个我还没有实际测试，只是道途听说)！
		这里的 精度损失 指的是 对同一张图片践行推理, .pt 的推理结果 跟 .onnx 的推理结果不一致！！！

	为了限制.onnx 模型文件的分发！！需要对 .onnx 文件进行加密！！ 实现此功能的方式 是  非对称加密 + 动态秘钥！！！
	1.不同电脑上面硬件的序列号是不一样的！ 我可以把多个硬件的序列号整合在一起，然后加密一下，说出加密后的序列号
	2.用加密后的序列号  去 加密 .onnx 模型以得到 加密后的 .bin 文件
	3.解密用的密码 是根据 加密后的序列号来的  每一台电脑解密用的 密码都是不一样的！！！
	4. 后面可以的话 直接做成 动态秘钥的形式，把时间戳加上！！！不同的时间 解密用到的 秘钥都是不一样的！！

	*/

public:
	ObjectDetection();

	void ReadClasses();
	void ReadClasses(string path);		// 读取指定.txt 文件中的  类别！！！

	void ReadImage(string path);		// 读取单张图片
	void ReadImages(string folder);		// 通过递归的方式 遍历一个文件夹，读取其中的素有图片，并将其放入到 ImageList 里面！！！

	vector<float> PreProcessing(string path);	// 预处理！
	void PreProcessing(Mat& img);
	void PreProcessing();				// 直接对 ImageList 中的 所有图片全都去做 预处理！！！！

	void PostProcessing();		// 后处理 相关操作
	void PostProcessing(const vector<Value>& outputtensor);
	
	void Inference();

	Mat Inference(vector<float> tensor);	// 推理单张图片

	string GenerateRandomString(size_t length);
	string GenerateNewFileName(const string& absolutePath);





public:
	vector<float> Input_Tensors;
	vector<string> Classes;		// 类别
	string ImagePath;	// 图片路径
	string ClassFilePath;
	const ORTCHAR_T* ModelPath = L"FN.onnx";		// .onnx 文件的路径！  这里是相对路径 直接从对应的 .exe 目录下面读取对应的 模型文件！
	vector<string> ImageList;	

	vector<float> SingleTensor;		// 单张图片张量

	vector<Mat> InputImages;				// 采集到的图片！！！
	vector<vector<float>> InputTensors;		// 多张图片对应的 张量！！！ 输入张量
	vector<vector<Value>> OutputTensors;	//  推理后得到的多个输出张量
	vector<float> MultiBatchTensor;		// 多Batch张量


	float ConfThreshold = 0.8;		// 置信度
	float IouThreshold = 0.6;		// Iou 阈值


	size_t Index;


private:
	Mat ResizeImage(Mat srcimg, int* newh, int* neww, int* top, int* left);
	bool IsImageFile(const path& file);
	



	// onnx runtime 相关操作！
public:
	bool IsEnableCUDA;		// 指示是否 使用 GPU 去推理！！！

private:
	void InitializeOrt();

	shared_ptr<char> inputName;
	shared_ptr<char> outputName;

	vector<const char*> inputnodenames_;	// 输入节点名字
	vector<const char*> outputnodenames_;	// 输出节点名字
	size_t inputnodenum_;					// 输入节点数量
	vector<int64_t> inputnodedims_;			// 输入节点维度
	size_t inputtensorsize_;				// Input Tensor Size

	optional<Session> session_;		// Session 没有默认的构造函数！而且还需要在指定逻辑执行完之后 才能够执行相关初始化工作！！！但是我又想要将其放到头文件里面,所以就只能 通过 延迟构造的方式来实现了！！！
	optional<Env> env_;
	optional<MemoryInfo> memoryinfo_;

	SessionOptions sessionoptions_;
	OrtCUDAProviderOptions cuda_options_;
	AllocatorWithDefaultOptions allocator_;



private:
	void TestInference();		// 测试用！！！！ 不能删



};

