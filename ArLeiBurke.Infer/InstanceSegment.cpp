#include "InstanceSegment.h"


 Mat InstanceSegment::transformation(const Mat& image, const Size& targetSize, const Scalar& mean, const Scalar& std) {

	cv::Mat resizedImage;
	//图片尺寸缩放
	cv::resize(image, resizedImage, targetSize, 0, 0, cv::INTER_AREA);
	cv::Mat normalized;
	resizedImage.convertTo(normalized, CV_32F);
	cv::subtract(normalized / 255.0, mean, normalized);
	cv::divide(normalized, std, normalized);
	return normalized;
}


InstanceSegment::InstanceSegment()
{

	string ImagePath = "C:\\Users\\Administrator\\Desktop\\Pic\\S00013_C01_P001_L0_PI139_G1_M1_202407100906.bmp";
	string ONNXPath = "FN.onnx";

	//const ORTCHAR_T* ModelPath = L"FN.onnx";		一开始我是这么写的，我觉的 它是能够直接 从 .exe 所在目录下面去 读取对应的文件的！！！但是后来发现，it is not work！！！！

	const ORTCHAR_T* ModelPath = L"C:\\Users\\Administrator\\Desktop\\ONNX\\FN.onnx";
	//const wchar_t* ModelPath = L"FN.onnx";

	Scalar mean(0.485, 0.456, 0.406); // 均值
	Scalar std(0.229, 0.224, 0.225);  // 标准差
	Mat frame = imread(ImagePath);
	wstring modelPath = wstring(ONNXPath.begin(), ONNXPath.end());
	SessionOptions session_options;
	//Env env = Env(ORT_LOGGING_LEVEL_ERROR, "default");
	Env env;

	// 设定单个操作(op)内部并行执行的最大线程数,可以提升速度
	session_options.SetIntraOpNumThreads(28);	// 同一操作内部线程数
	session_options.SetInterOpNumThreads(2);  // 操作之间的并发数
	//session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
	cout << "onnxruntime inference try to use GPU Device" << endl;

	//Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
	//Ort::SessionOptions session_options;
	// 是否使用GPU
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);			// 这行代码一直给我报错，，我都不知道是因为什么原因导致的！！！
	// 
	//Session session_(env, modelPath.c_str(), session_options);
	Session session_(env, ModelPath, session_options);

	int input_nodes_num = session_.GetInputCount();
	int output_nodes_num = session_.GetOutputCount();
	vector<string> input_node_names;
	vector<string> output_node_names;
	AllocatorWithDefaultOptions allocator;


	int input_h = 0;
	int input_w = 0;

	// 获得输入信息
	for (int i = 0; i < input_nodes_num; i++) {
		auto input_name = session_.GetInputNameAllocated(i, allocator);
		input_node_names.push_back(input_name.get());
		auto inputShapeInfo = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		int ch = inputShapeInfo[1];
		input_h = inputShapeInfo[2];
		input_w = inputShapeInfo[3];
		std::cout << "input format: " << ch << "x" << input_h << "x" << input_w << std::endl;
	}

	// 获得输出信息 多输出
	for (int i = 0; i < output_nodes_num; i++) {
		auto output_name = session_.GetOutputNameAllocated(i, allocator);
		output_node_names.push_back(output_name.get());
		auto outShapeInfo = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		int ch = outShapeInfo[1];
		int output_h = outShapeInfo[2];
		int output_w = outShapeInfo[3];
		std::cout << "output format: " << ch << "x" << output_h << "x" << output_w << std::endl;
	}

	// 图象预处理 - 格式化操作
	int64 start = cv::getTickCount();
	cv::Mat rgbImage;
	cv::cvtColor(frame, rgbImage, cv::COLOR_BGR2RGB);
	cv::Size targetSize(input_w, input_h);
	// 对原始图像resize和归一化
	cv::Mat normalized = transformation(rgbImage, targetSize, mean, std);
	cv::Mat blob = cv::dnn::blobFromImage(normalized);
	size_t tpixels = input_w * input_h * 3;
	std::array<int64_t, 4> input_shape_info{ 1, 3, input_h, input_w };

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob.ptr<float>(), tpixels, input_shape_info.data(), input_shape_info.size());
	// 输入一个数据
	const std::array<const char*, 1> inputNames = { input_node_names[0].c_str() };
	// 输出多个数据
	const std::array<const char*, 3> outNames = { output_node_names[0].c_str(),output_node_names[1].c_str(),output_node_names[2].c_str() };
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, inputNames.size(), outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}
	// 选择最后一个输出作为最终的mask
	const float* mask_data = ort_outputs[2].GetTensorMutableData<float>();
	auto outShape = ort_outputs[2].GetTensorTypeAndShapeInfo().GetShape();
	int num_cn = outShape[1];
	int out_h = outShape[2];
	int out_w = outShape[3];

	int step = out_h * out_w;
	// 逐像素判断是背景还是前景
	cv::Mat result = cv::Mat::zeros(cv::Size(out_w, out_h), CV_8UC1);
	for (int row = 0; row < out_h; row++) {
		for (int col = 0; col < out_w; col++) {
			float c1 = mask_data[row * out_w + col];
			if (c1 > 0.5) {
				result.at<uchar>(row, col) = 255;
			}
			else {
				result.at<uchar>(row, col) = 0;
			}
		}
	}
	cv::Mat mask, binary;
	cv::resize(result, mask, cv::Size(frame.cols, frame.rows));
	cv::threshold(mask, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
	float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
	std::cout << "Total Testing Time : " << t << std::endl;
	cv::imshow("mask", binary);
	cv::waitKey(0);
	// 释放资源
	session_options.release();
	session_.release();


}