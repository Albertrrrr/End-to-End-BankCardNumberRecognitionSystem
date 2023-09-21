# An End-to-End Bank Card Number Recognition System
## Acknowledgment
@inproceedings{dong2019method,
  title={A Method of Bank Card Number Recognition and End-to-End Integration Based on Deep Learning},
  author={Dong, Qinqin and Zhang, Ruixian and Jing, Chao and Wang, Kun and Zhang, Wuping and Li, Fuzhong},
  booktitle={2019 Chinese Automation Congress (CAC)},
  pages={5060--5063},
  year={2019},
  organization={IEEE}
}

## 简介
基于改进的CTPN+DenseNet的银行卡识别系统

* 文本检测：CTPN
* 文本识别：DenseNet + CTC

使用环境Ubuntu 18.04/Ubuntu 16.04 

GCC版本：5.5

实现语言:Python 3.6

深度学习框架:TensorFlow Keras

部署虚拟环境:Anaconda3

Nvidia CUDA版本：CUDA9.0

VGGnet_fast_rcnn_iter_50000.ckpt 预训练文件

链接：https://pan.baidu.com/s/1P7HdTvHFWMDxtLCBKVfwYw 提取码：dhgo

## 环境部署
进入SourceCode目录后 创建新的anaconda虚拟环境并进入

``` Bash
conda create -n BCNet python=3,6
source activate BCNet
```

添加所需要的第三方库
``` Bash
sh setup.sh
```

```bash
pip install numpy scipy matplotlib pillow
pip install easydict opencv-python keras h5py PyYAML
pip install cython==0.24
pip install tensorflow-gpu==1.3.0
```

第三方库安装完成后进入目录/SourceCode/CardPositioning/lib/utils 

务必在BCNet的虚拟环境中运行make.sh
```bash
chmod +x make.sh
sh make.sh
```

需注意的是在Ubuntu18.04环境下需要将gcc降级到5.5版本才可正常运行

## Demo
将测试图片放入test_images目录，检测结果会保存到test_result中

``` Bash
python demo.py
```


### DenseNet + CTC训练

#### 1. 数据准备

数据集：链接: https://pan.baidu.com/s/1DVXnK5oum7fE2mBqenomgA 提取码: 8hbg

按照官方的1084的图片进行数据增强 1084*80 = 86720张银行卡图片

图片解压后放置到train/images目录下，描述文件放到train目录下

测试集与训练集已经划分完成且已经加好label


#### 2. 结果

|     acc     | loss |
| -----------| ---------- |
| 0.996 | 0.14 |

| val acc | loss |
| -----------| ---------- |
| 0.992 | 0.092 |


![](http://m.qpic.cn/psb?/V10Rz2Ti3Ob5wF/aHHUuVZo7btYxcoNAobdZcBRlQpT9I8xa7TUERo*qRE!/b/dAcBAAAAAAAA&bo=sQGcAgAAAAADBww!&rf=viewer_4)

![](http://m.qpic.cn/psb?/V10Rz2Ti3Ob5wF/tagS6DHdfesIIacuceEmmzcSC2R8VFk6veXCL5d0u3I!/b/dL4AAAAAAAAA&bo=CgKdAgAAAAADF6U!&rf=viewer_4)

* GPU: GTX TITAN X *2 
* Keras Backend: Tensorflow


#### 3.GUI展示

此系统中的GUI以网页形式呈现 前端为H5+CSS+JQuery 后端为Flask

进入之前的anaconda虚拟环境(BCNet)中安装Flask
```bash
pip install flask
```


安装成功后修改源代码，使用GUI时必须将源代码中的相对路径引用改为绝对路径引用

* SourceCode/CardPositioning/text_detect.py 

第24行 cfg.TEST.checkpoints_path = (os.getcwd() + r'/CardPositioning/checkpoints')

改为SourceCode/CardPositioning/checkpoints的绝对路径


* SourceCode/CardRecognition/model.py

第29行 modelPath = os.path.join(os.getcwd(), os.getcwd() + r'/train/models/weights_densenet-09-0.11.h5')

改为SourceCode/train/models/weights_densenet-09-0.11.h5的绝对路径

* SourceCode/demoSupportOCR.py

第88行  cfg_from_file(os.getcwd() + r'/CardPositioning/ctpn/text.yml')

改为SourceCode/CardPositioning/ctpn/text.yml的绝对路径

改完后再运行app.py程序即可看到GUI


