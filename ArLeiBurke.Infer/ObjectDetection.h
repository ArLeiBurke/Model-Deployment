#pragma once

#include <fstream>
#include <sstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <provider_options.h>
#include <onnxruntime_cxx_api.h>
#include <filesystem>  // C++17

#include <random>		// ���������

#include <optional>	// ��ʱ����

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
	

	�� Onnx Runtime ȥ����ģ�͵�ʱ�� ����Ҫȥִ����Щ Ԥ������صĲ����أ�����
		1.Resize Image Size
		2.BGR -> RGB ��������Ϊ OpneCV ���� BGR����ʽ��ȡͼƬ�ģ���
		3.��������ת�� + ��һ������ ��uint8(0-255) ת���� float32,���ҽ���Ӧ�����ݹ�һ����[0,1]
		4.ͨ������ HWC -> CHW ��������Ϊ Onnx ģ��Ĭ��ʹ�õ��� CHW ��ʽ��������OpenCV����ͼ���ʱ��Ĭ�ϸ�ʽ�� HWC������


	Onnx Runtime ��Batch �����Ҵ��ݸ�ģ�͵� Tensor �� Shape �� [X,3,640,640] ������ӵ� ��������ͨ�� Session.Run(XXX) �õ���Tensor �� Shape ��ʲô���ӵ��أ����������Tensor Ӧ����ô�����أ�����

	Onnx Runtime ��Batch �����ʱ�򣬶� ���� Tensor ����ʲôҪ��ô�����������������ģ�

	������ģ����������� [1,3,640,640] �Ļ� �ǲ�֧�� ��batch����ģ��������� ��̬batch ������ [X,3,640,640] ��ʽ�ģ��ſ���֧�ֶ�̬batch ��������

	��.pt��ʽ��ģ���ļ� ת���� .onnx ��ʽ��ģ���ļ� ����һ���ɶ��ľ�����ʧ(����ľ�����ʧ��.ptת����.onnx ��ʱ���µģ�����һ�û��ʵ�ʲ��ԣ�ֻ�ǵ�;��˵)��
		����� ������ʧ ָ���� ��ͬһ��ͼƬ��������, .pt �������� �� .onnx ����������һ�£�����

	Ϊ������.onnx ģ���ļ��ķַ�������Ҫ�� .onnx �ļ����м��ܣ��� ʵ�ִ˹��ܵķ�ʽ ��  �ǶԳƼ��� + ��̬��Կ������
	1.��ͬ��������Ӳ�������к��ǲ�һ���ģ� �ҿ��԰Ѷ��Ӳ�������к�������һ��Ȼ�����һ�£�˵�����ܺ�����к�
	2.�ü��ܺ�����к�  ȥ ���� .onnx ģ���Եõ� ���ܺ�� .bin �ļ�
	3.�����õ����� �Ǹ��� ���ܺ�����к�����  ÿһ̨���Խ����õ� ���붼�ǲ�һ���ģ�����
	4. ������ԵĻ� ֱ������ ��̬��Կ����ʽ����ʱ������ϣ�������ͬ��ʱ�� �����õ��� ��Կ���ǲ�һ���ģ���

	*/

public:
	ObjectDetection();

	void ReadClasses();
	void ReadClasses(string path);		// ��ȡָ��.txt �ļ��е�  ��𣡣���

	void ReadImage(string path);		// ��ȡ����ͼƬ
	void ReadImages(string folder);		// ͨ���ݹ�ķ�ʽ ����һ���ļ��У���ȡ���е�����ͼƬ����������뵽 ImageList ���棡����

	vector<float> PreProcessing(string path);	// Ԥ����
	void PreProcessing(Mat& img);
	void PreProcessing();				// ֱ�Ӷ� ImageList �е� ����ͼƬȫ��ȥ�� Ԥ����������

	void PostProcessing();		// ���� ��ز���
	void PostProcessing(const vector<Value>& outputtensor);
	
	void Inference();

	Mat Inference(vector<float> tensor);	// ������ͼƬ

	string GenerateRandomString(size_t length);
	string GenerateNewFileName(const string& absolutePath);





public:
	vector<float> Input_Tensors;
	vector<string> Classes;		// ���
	string ImagePath;	// ͼƬ·��
	string ClassFilePath;
	const ORTCHAR_T* ModelPath = L"FN.onnx";		// .onnx �ļ���·����  ���������·�� ֱ�ӴӶ�Ӧ�� .exe Ŀ¼�����ȡ��Ӧ�� ģ���ļ���
	vector<string> ImageList;	

	vector<float> SingleTensor;		// ����ͼƬ����

	vector<Mat> InputImages;				// �ɼ�����ͼƬ������
	vector<vector<float>> InputTensors;		// ����ͼƬ��Ӧ�� ���������� ��������
	vector<vector<Value>> OutputTensors;	//  �����õ��Ķ���������
	vector<float> MultiBatchTensor;		// ��Batch����


	float ConfThreshold = 0.8;		// ���Ŷ�
	float IouThreshold = 0.6;		// Iou ��ֵ


	size_t Index;


private:
	Mat ResizeImage(Mat srcimg, int* newh, int* neww, int* top, int* left);
	bool IsImageFile(const path& file);
	



	// onnx runtime ��ز�����
public:
	bool IsEnableCUDA;		// ָʾ�Ƿ� ʹ�� GPU ȥ��������

private:
	void InitializeOrt();

	shared_ptr<char> inputName;
	shared_ptr<char> outputName;

	vector<const char*> inputnodenames_;	// ����ڵ�����
	vector<const char*> outputnodenames_;	// ����ڵ�����
	size_t inputnodenum_;					// ����ڵ�����
	vector<int64_t> inputnodedims_;			// ����ڵ�ά��
	size_t inputtensorsize_;				// Input Tensor Size

	optional<Session> session_;		// Session û��Ĭ�ϵĹ��캯�������һ���Ҫ��ָ���߼�ִ����֮�� ���ܹ�ִ����س�ʼ����������������������Ҫ����ŵ�ͷ�ļ�����,���Ծ�ֻ�� ͨ�� �ӳٹ���ķ�ʽ��ʵ���ˣ�����
	optional<Env> env_;
	optional<MemoryInfo> memoryinfo_;

	SessionOptions sessionoptions_;
	OrtCUDAProviderOptions cuda_options_;
	AllocatorWithDefaultOptions allocator_;



private:
	void TestInference();		// �����ã������� ����ɾ



};

