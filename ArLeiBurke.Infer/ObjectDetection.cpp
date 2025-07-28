#include "ObjectDetection.h"





ObjectDetection::ObjectDetection()
{
	InitializeOrt();
}


void ObjectDetection::InitializeOrt()
{
	memoryinfo_.emplace(MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
	env_.emplace(ORT_LOGGING_LEVEL_WARNING, "yolov5s-5.0");

	sessionoptions_.SetIntraOpNumThreads(0);	// 为什么修改 节点内部线程数 会提高推理速度呢？？？但是改的太大的话又会导致 推理速度变慢，	问了一下GPT，说这个参数仅对CPU有效，用GPU推理的话，这个就没用了！
	sessionoptions_.SetExecutionMode(ORT_PARALLEL);		// 设置执行模式！！！ ORT_SEQUENTIAL 顺序执行所有节点！  ORT_PARALLEL 并行执行无依赖的节点！！！
	sessionoptions_.EnableCpuMemArena();		//  启用 CPU 内存 Arena 分配器			启用时，Arena 会持续占用内存，不会主动释放。启用这个会导致内存占用过大？？？？
	sessionoptions_.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	cuda_options_.device_id = 0;
	cuda_options_.arena_extend_strategy = 0;
	cuda_options_.gpu_mem_limit = (size_t)1 * 1024 * 1024 * 1024;
	cuda_options_.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;	//  OrtCudnnConvAlgoSearchDefault  OrtCudnnConvAlgoSearchHeuristic	 OrtCudnnConvAlgoSearchExhaustive  三个可选的值，不会的话直接问ChatGPT
	cuda_options_.do_copy_in_default_stream = 1;
	if (IsEnableCUDA)
		sessionoptions_.AppendExecutionProvider_CUDA(cuda_options_);
	sessionoptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
	session_.emplace(env_.value(), ModelPath, sessionoptions_);

	inputnodenum_ = session_->GetInputCount();
	inputName = move(session_->GetInputNameAllocated(0, allocator_));
	inputnodenames_.push_back(inputName.get());
	outputName = move(session_->GetOutputNameAllocated(0, allocator_));
	outputnodenames_.push_back(outputName.get());

	TypeInfo type_info = session_->GetInputTypeInfo(0);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	inputnodedims_ = tensor_info.GetShape();

	size_t input_tensor_size = 1;
	for (auto dim : inputnodedims_) {
		input_tensor_size *= dim;
	}
	inputtensorsize_ = input_tensor_size;


}

Mat ObjectDetection::Inference(vector<float> tensor)
{
	Mat mat;
	return mat;
}

void ObjectDetection::Inference()
{
	Index = 0;
	for (auto outer = InputTensors.begin(); outer != InputTensors.end(); ++outer,++Index)
	{
		vector<float> inputtensorvalues = *outer;
		Value input_tensor = Value::CreateTensor<float>(memoryinfo_.value(), inputtensorvalues.data(), inputtensorsize_, inputnodedims_.data(), inputnodedims_.size());
		vector<Value> ort_input;
		ort_input.push_back(move(input_tensor));

		/*
		下面这行代码会给我报错 给我报 error C2280 : 'X::X(const X &)' : attempting to reference a deleted function

		下面这行代码还会给我报另一个错误，给我报  inputnodenames_ 无效  或者  outputnodenames_ 无效，，后来发现是因为 函数执行完之后 shared_ptr<char> 类型的变量释放 导致对应的 内存也跟着一起析构了！！！

		*/
		//auto output_tensor = session_->Run(RunOptions{ nullptr }, intputnodenames_.data(), ort_input.data(), intputnodenames_.size(), outputnodenames_.data(), outputnodenames_.size());	// 推理
		vector<Value> output_tensor = session_->Run(RunOptions{ nullptr }, inputnodenames_.data(), ort_input.data(), inputnodenames_.size(), outputnodenames_.data(), outputnodenames_.size());	// 推理

		PostProcessing(output_tensor);		//		 后处理！

	}
}



void ObjectDetection::PostProcessing(const vector<Value>& output_tensors)
{
	const float* rawOutput = output_tensors[0].GetTensorData<float>();
	vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	vector<float> output(rawOutput, rawOutput + count);

	vector<cv::Rect> boxes;
	vector<float> confs;
	vector<int> classIds;
	int numClasses = (int)outputShape[2] - 5;
	int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

	for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
	{
		float clsConf = *(it + 4);//object scores
		if (clsConf > ConfThreshold)
		{
			int centerX = (int)(*it);
			int centerY = (int)(*(it + 1));
			int width = (int)(*(it + 2));
			int height = (int)(*(it + 3));
			int x1 = centerX - width / 2;
			int y1 = centerY - height / 2;
			boxes.emplace_back(cv::Rect(x1, y1, width, height));

			// first 5 element are x y w h and obj confidence
			int bestClassId = -1;
			float bestConf = 0.0;

			for (int i = 5; i < numClasses + 5; i++)
			{
				if ((*(it + i)) > bestConf)
				{
					bestConf = it[i];
					bestClassId = i - 5;
				}
			}

			//confs.emplace_back(bestConf * clsConf);
			confs.emplace_back(clsConf);
			classIds.emplace_back(bestClassId);
		}
	}


	vector<int> indices;
	NMSBoxes(boxes, confs, ConfThreshold, IouThreshold, indices);		// Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences

	//随机数种子
	RNG rng((unsigned)time(NULL));
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int index = indices[i];
		int colorR = rng.uniform(0, 255);
		int colorG = rng.uniform(0, 255);
		int colorB = rng.uniform(0, 255);

		//保留两位小数
		float scores = round(confs[index] * 100) / 100;
		std::ostringstream oss;
		oss << scores;

		rectangle(InputImages[Index], Point(boxes[index].tl().x, boxes[index].tl().y), Point(boxes[index].br().x, boxes[index].br().y), Scalar(colorR, colorG, colorB), 1.5);
		putText(InputImages[Index], Classes[classIds[index]] + " " + oss.str(), Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(colorR, colorG, colorB), 2);
	
	
	}

	string combinestring = GenerateNewFileName(ImageList[Index]);
	imwrite(combinestring, InputImages[Index]);


}

string ObjectDetection::GenerateRandomString(size_t length)
{
	const std::string characters =
		"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
	std::random_device rd;  // 用于获取随机种子
	std::mt19937 generator(rd());  // Mersenne Twister 随机数引擎
	std::uniform_int_distribution<> distrib(0, characters.size() - 1);

	std::string result;
	for (size_t i = 0; i < length; ++i) {
		result += characters[distrib(generator)];
	}

	return result;
}

string ObjectDetection::GenerateNewFileName(const string& absolutePath)
{
	namespace fs = std::filesystem;

	fs::path pathObj(absolutePath);
	std::string stem = pathObj.stem().string();         // 文件名，不含扩展名
	std::string extension = pathObj.extension().string(); // 扩展名（.txt 之类）
	std::string randomSuffix = GenerateRandomString(8);

	std::string newFileName = stem + "_" + randomSuffix + extension;
	fs::path newPath = pathObj.parent_path() / newFileName;

	return newPath.string();
}





void ObjectDetection::ReadClasses(string Path)
{
	//string filename = "FuChiOCR.txt";
	std::ifstream ifs(Path.c_str());
	std::string line;
	while (getline(ifs, line)) Classes.push_back(line);

}

void ObjectDetection::ReadClasses()
{
	std::ifstream ifs(ClassFilePath.c_str());
	std::string line;
	while (getline(ifs, line)) Classes.push_back(line);
}

Mat ObjectDetection::ResizeImage(Mat srcimg, int* newh, int* neww, int* top, int* left)
{

	int srch = srcimg.rows, srcw = srcimg.cols;
	int inpHeight = 640;
	int  inpWidth = 640;
	*newh = inpHeight;
	*neww = 640;
	bool keep_ratio = true;
	Mat dstimg;
	if (keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = inpHeight;
			*neww = int(inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, inpWidth - *neww - *left, BORDER_CONSTANT, 114);
		}
		else {
			*newh = (int)inpHeight * hw_scale;
			*neww = inpWidth;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 114);
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	return dstimg;


}


void ObjectDetection::PreProcessing(Mat& srcimg)
{
	vector<float> InputTensor(inputtensorsize_);
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = ResizeImage(srcimg, &newh, &neww, &padh, &padw);//Padded resize

	InputImages.push_back(dstimg);

	// 主要是做 归一化处理 + NHWC ➜ NCHW 排列转换 
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < 640; i++)
		{
			for (int j = 0; j < 640; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				InputTensor[c * 640 * 640 + i * 640 + size_t(j)] = pix / 255.0;
			}
		}
	}

	SingleTensor = InputTensor;
}

// 读取某个文件夹下面的所有图片！！！！
vector<float> ObjectDetection::PreProcessing(string Path)
{
	vector<float> InputTensor(inputtensorsize_);
	Mat srcimg = imread(Path);
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = ResizeImage(srcimg, &newh, &neww, &padh, &padw);//Padded resize

	InputImages.push_back(dstimg);


	// 主要是做 归一化处理 + NHWC ➜ NCHW 排列转换 
	for (int c = 0; c < 3; c++)
	{
		for (int i = 0; i < 640; i++)
		{
			for (int j = 0; j < 640; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				InputTensor[c * 640 * 640 + i * 640 + size_t(j)] = pix / 255.0;
			}
		}
	}

	return InputTensor;
}


void ObjectDetection::PreProcessing()
{
	//遍历 IamgeList 集合 以获取其中的每一个变量！
	for (vector<string>::const_iterator it = ImageList.begin(); it != ImageList.end(); ++it)
	{
		string path = *it;
		auto tensor = PreProcessing(path);
		InputTensors.push_back(tensor);
	}

}


bool ObjectDetection::IsImageFile(const path& File)
{
	std::string ext = File.extension().string();
	std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
	return ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp" || ext == ".tiff";
}

void ObjectDetection::ReadImages(string rootPath)
{
	for (const auto& entry : recursive_directory_iterator(rootPath)) {
		if (is_regular_file(entry) && IsImageFile(entry.path())) {
			ImageList.push_back(entry.path().string());
		}
	}

}



void ObjectDetection::TestInference()
{
	string ImagePath = "C:\\Users\\Administrator\\Desktop\\Pic\\FN1\\S00018_C01_P001_L0_PI139_G1_M1_202407100906.bmp";
	//const ORTCHAR_T* ModelPath = L"C:\\Users\\Administrator\\Desktop\\ONNX\\FN.onnx";
	const ORTCHAR_T* ModelPath = L"FN.onnx";
	string ONNXPath = "FN.onnx";


	vector<string> ClassName = {
	"FF",
	"1",
	"2",
	"3",
	"4",
	"FN",
	"FM",
	"FK",
	"FD",
	"FH",
	"FJ",
	"FG",
	"FL",
	};


	Env env(ORT_LOGGING_LEVEL_WARNING, "yolov5s-5.0");

	SessionOptions session_options;
	session_options.SetIntraOpNumThreads(0);	// 为什么修改 节点内部线程数 会提高推理速度呢？？？但是改的太大的话又会导致 推理速度变慢，	问了一下GPT，说这个参数仅对CPU有效，用GPU推理的话，这个就没用了！
	session_options.SetExecutionMode(ORT_PARALLEL);		// 设置执行模式！！！ ORT_SEQUENTIAL 顺序执行所有节点！  ORT_PARALLEL 并行执行无依赖的节点！！！
	session_options.EnableCpuMemArena();		//  启用 CPU 内存 Arena 分配器			启用时，Arena 会持续占用内存，不会主动释放。启用这个会导致内存占用过大？？？？
	session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

	/*

一开始的时候下面这行代码给我报错，， FAIL : LoadLibrary failed with error 126 "" when trying to load  onnxruntime_providers_cuda.dll  后来我直接百度了一下，找到了CSDN 上面的一篇文章，
文章链接		https://blog.csdn.net/weixin_41700859/article/details/144679388
文章里面讲 说是因为 ONNX Runtime / CUDA / CUDNN 版本之间不一致导致的！！！所以我就把适配的给下载了  然后运行了一下 下面的这行代码就不给我报错了！！！
Onnx Runtime 1.18.1
CUDA 12.8
CUDNN 9.8.0

*/

	OrtCUDAProviderOptions cuda_options;
	cuda_options.device_id = 0;
	cuda_options.arena_extend_strategy = 0;
	cuda_options.gpu_mem_limit = (size_t)1 * 1024 * 1024 * 1024;
	cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;	//  OrtCudnnConvAlgoSearchDefault  OrtCudnnConvAlgoSearchHeuristic	 OrtCudnnConvAlgoSearchExhaustive  三个可选的值，不会的话直接问ChatGPT
	cuda_options.do_copy_in_default_stream = 1;
	session_options.AppendExecutionProvider_CUDA(cuda_options);
	session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);


	Session session(env, ModelPath, session_options);
	AllocatorWithDefaultOptions allocator;
	size_t num_input_nodes = session.GetInputCount();


	// 下面这个地方 可以优化一下！！！！
	//vector<const char*> input_node_names = { "images" };
	//vector<const char*> output_node_names = { "output0" };
	vector<const char*> input_node_names;
	vector<const char*> output_node_names;
	shared_ptr<char> inputName = move(session.GetInputNameAllocated(0, allocator));
	input_node_names.push_back(inputName.get());
	shared_ptr<char> outputName = move(session.GetOutputNameAllocated(0, allocator));
	output_node_names.push_back(outputName.get());


	//  获取输入张量的 Shape ！！！		session.GetInputTypeInfo(0); 获取输入模型第零个节点的类型信息！！如果输入模型是动态维度，或者有多个节点的话，，那下面这行代码就不能够正常 Work 了！！！
	//  先获取一下输入的 Shape ！！！！
	//	vector<int64_t> input_node_dims = { 1, 3, 640, 640 };
	TypeInfo type_info = session.GetInputTypeInfo(0);
	auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
	vector<int64_t> input_node_dims = tensor_info.GetShape();

	size_t input_tensor_size = 1 * 3 * 640 * 640;

	vector<float> input_tensor_values(input_tensor_size);		// 这个地方是真的离谱，一开始的时候我并没有这样子显示分配内存，代码是能够正常 work的！！后来我动了一下其他地方的地方，结果这里给我报错了！！我是真的服!!!
	//vector<float> input_tensor_values;
	Mat srcimg = imread(ImagePath);
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = ResizeImage(srcimg, &newh, &neww, &padh, &padw);//Padded resize
	for (int c = 0; c < 3; c++)		// 主要是做 归一化处理 + NHWC ➜ NCHW 排列转换 
	{
		for (int i = 0; i < 640; i++)
		{
			for (int j = 0; j < 640; j++)
			{
				float pix = dstimg.ptr<uchar>(i)[j * 3 + 2 - c];
				input_tensor_values[c * 640 * 640 + i * 640 + size_t(j)] = pix / 255.0;

			}
		}
	}



	auto memory_info = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Value input_tensor = Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), input_node_dims.size());

	vector<Value> ort_inputs;
	ort_inputs.push_back(move(input_tensor));


	for (int i = 0; i < 10; ++i) {
		auto start = std::chrono::high_resolution_clock::now();
		session.Run(RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());
		auto end = std::chrono::high_resolution_clock::now();
		double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
		std::cout << "Run " << i << " time: " << elapsed << " ms" << std::endl;
	}

	clock_t start = clock();

	// 推理
	vector<Value> output_tensors = session.Run(RunOptions{ nullptr }, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());


	// Get pointer to output tensor float values
	const float* rawOutput = output_tensors[0].GetTensorData<float>();
	vector<int64_t> outputShape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	size_t count = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
	vector<float> output(rawOutput, rawOutput + count);

	vector<cv::Rect> boxes;
	vector<float> confs;
	vector<int> classIds;
	int numClasses = (int)outputShape[2] - 5;
	int elementsInBatch = (int)(outputShape[1] * outputShape[2]);

	float confThreshold = 0.5;
	for (auto it = output.begin(); it != output.begin() + elementsInBatch; it += outputShape[2])
	{
		float clsConf = *(it + 4);//object scores
		if (clsConf > confThreshold)
		{
			int centerX = (int)(*it);
			int centerY = (int)(*(it + 1));
			int width = (int)(*(it + 2));
			int height = (int)(*(it + 3));
			int x1 = centerX - width / 2;
			int y1 = centerY - height / 2;
			boxes.emplace_back(cv::Rect(x1, y1, width, height));

			// first 5 element are x y w h and obj confidence
			int bestClassId = -1;
			float bestConf = 0.0;

			for (int i = 5; i < numClasses + 5; i++)
			{
				if ((*(it + i)) > bestConf)
				{
					bestConf = it[i];
					bestClassId = i - 5;
				}
			}

			//confs.emplace_back(bestConf * clsConf);
			confs.emplace_back(clsConf);
			classIds.emplace_back(bestClassId);
		}
	}







	float iouThreshold = 0.5;
	vector<int> indices;
	NMSBoxes(boxes, confs, confThreshold, iouThreshold, indices);		// Perform non maximum suppression to eliminate redundant overlapping boxes with lower confidences

	//随机数种子
	RNG rng((unsigned)time(NULL));
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int index = indices[i];
		int colorR = rng.uniform(0, 255);
		int colorG = rng.uniform(0, 255);
		int colorB = rng.uniform(0, 255);

		//保留两位小数
		float scores = round(confs[index] * 100) / 100;
		std::ostringstream oss;
		oss << scores;

		rectangle(dstimg, Point(boxes[index].tl().x, boxes[index].tl().y), Point(boxes[index].br().x, boxes[index].br().y), Scalar(colorR, colorG, colorB), 1.5);
		putText(dstimg, ClassName[classIds[index]] + " " + oss.str(), Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(colorR, colorG, colorB), 2);
	}

	clock_t end = clock();
	double elapsed_secs = double(end - start) / CLOCKS_PER_SEC;
	cout << elapsed_secs << endl;

	imshow("检测结果", dstimg);
	cv::waitKey();

}







