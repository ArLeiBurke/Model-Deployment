# 部署模型

通过 **C++ + Onnx Runtime** 的方式去部署 yolov5 导出的 .onnx 格式的模型文件。<br> 

相关代码我已经封装成一个类,将项目拉到本地之后,需要配置一下环境，然后修改几个参数就能够正常运行了。

# 依赖版本
我电脑上面安装的是以下<br>
**CUDNN 9.8.0** <br>
**通过网盘分享的文件：CUDNN
链接: https://pan.baidu.com/s/1ERgFkiKzUG5nBqNKnTkcIA?pwd=p629 提取码: p629 ** <br>

**CUDA 12.8** <br>
**通过网盘分享的文件：CUDA
链接: https://pan.baidu.com/s/1vXNpZsANhRjdRYM_ru-oFg?pwd=c3s3 提取码: c3s3 **<br>

**ONNX Runtime 1.18.1** <br>
**通过网盘分享的文件：ONNX Runtime
链接: https://pan.baidu.com/s/1vgdqTIru-jWTvsueypzQAg 提取码: x9bu ** <br>


# 碎碎念
一开始我电脑上面安装的 Onnx Runtime 版本是 1.20.1 ,运行代码的时候一直给我报错，报什么错我忘了，后来我卸载了之后 重新安装 1.18.1版本的 ONNX Runtime 就不给我报错了<br>
