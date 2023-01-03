# 人脸识别与人脸质量评估项目
MagFace: A Universal Representation for Face Recognition and Quality Assessment    

**From 樊一超**     

这是一个基于pytorch的开源项目，用于进行人脸识别，同时带有人脸质量评估的功能。可以认为是对ArcFace的优化。ArcFace提出增加角度间隔尽可能的拉大不同类别之间的角度间隔，从而达到压缩类内距离拉大类间距离的目的。但是ArcFace存在缺点，即仅使用了一个固定常量间隔，且只考虑了类间间隔并没有对类内分布做更精细化的处理，没有考虑模长等维度的信息。鉴于此，提出了ArgFace，充分利用角度和模长两个维度信息，既可以在识别问题上获得更好的性能，又可以利用模长信息来衡量人脸质量问题，同时更好的类内分布也有利于聚类任务。    

**项目链接：** https://github.com/IrvingMeng/MagFace  
**论文链接：** https://arxiv.org/abs/2103.06627  
**知乎解读** https://zhuanlan.zhihu.com/p/475775106   
**此外** ，关于一些此模型的解读均整理在doc目录下。

**训练测试环境：** recognition，获取方式docker pull fanacio/recognition:v0 或者本地保存镜像recognition.tar

## 1. 数据集的准备
### 1.1 用于测试的数据集 

1. 数据集[下载地址](https://pan.baidu.com/s/1ekuNnIWa6kwtsPuwRZaDmQ), 提取码：fz3h。同时将其下载，并直接放在eval/eval_recognition目录下，名称为lfw_cfp_agedb.tar。

### 1.2 用于训练的数据集

1. 参考EQFace模型中的README.md文件，将数据集进行下载并使用相关docker将其转换成images格式备用。本人选择的是比较小的数据集CASIA-Webface。且将训练集存放在dataset/images目录下。

## 2. 预训练权重的准备

**下面列出官方给出的一些权重列表**  
| Parallel Method | Loss | Backbone | Dataset | Split FC? | Model | Log File | name |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DDP | MagFace | iResNet100 | MS1MV2 | Yes | [BaiduDrive](https://pan.baidu.com/s/1E9miCIesvf2NBBhCTyS8IQ) code:fz3h | / | magface_epoch_00025.pth |
| DDP | MagFace | iResNet50 | MS1MV2 | Yes | [BaiduDrive](https://pan.baidu.com/s/1yIL63OeLZAIHRYoAhu_gqQ) code: fz3h| [BaiduDrive](https://pan.baidu.com/s/1MGAmhtOangqr8nHxIFmNvg), code: 66j1 | magface_iresnet50_MS1MV2_ddp_fp32.pth |
| DDP | Mag-CosFace | iResNet50 | MS1MV2 | Yes | [BaiduDrive](https://pan.baidu.com/s/17zlNzNS_ZjB2cH39ZCr-hw) code: fz3h| [BaiduDrive](https://pan.baidu.com/s/10EQjRydQLJMAU98q7lH10w), code: ejec | mag-cosface_iresnet50_MS1MV2_ddp_fp32.pth |
| DP | MagFace | iResNet50 | MS1MV2 | No | [BaiduDrive](https://pan.baidu.com/s/1atuZZDkcCX3Bl14J8Ss_YQ) code: tvyv | [BaiduDrive](https://pan.baidu.com/s/1T6_TkEh9v9Vtf4Sw-chT2w), code: hpbt | magface_iresnet50_MS1MV2_dp.pth |
| DP | MagFace | iResNet18 | CASIA-WebFace | No | [BaiduDrive](https://pan.baidu.com/s/1TvTGYMFTCN3LL4CEKebObA) code: fz3h | [BaiduDrive](https://pan.baidu.com/s/1bdfE7W2ffUB8ehDaOt-tBw), code: qv2x | magface_iresnet18_casia_dp.pth |
| DP | ArcFace | iResNet18 | CASIA-WebFace | No | [BaiduDrive](https://pan.baidu.com/s/1GZzucQVItyYaddMCktaDNw) code: fz3h | [BaiduDrive](https://pan.baidu.com/s/1lp4wAlz85w2Y29DT8RqGfQ), code: 756e | arcface_iresnet18_casia_dp.pth |

**注意：** 本地将这些权重均保存在weights目录下。

## 3. 测试
### 3.1 Face Recognition
进入eval/eval_recognition目录，首先将tar数据集进行解压，执行解压命令如下：
```bash  
tar -xf lfw_cfp_agedb.tar   
```
然后执行如下脚本eval.sh，例如(**第三个参数100表示iresnet100，如pth文件是resnet18训练所得，则为18**)：
```bash
./eval.sh ../../weights/magface_epoch_00025.pth official 100
```  
执行结果：  
```bash
...
...
=> torch version : 1.8.1+cu111
=> ngpus : 2
=> modeling the network ...
=> loading pth from magface_epoch_00025.pth ...
=> building the dataloader ...
=> preparing dataset for inference ...
=> starting inference engine ...
=> embedding features will be saved into ./features/magface_iresnet100//agedb_official.list
2022-12-27 09:25:03.950 | INFO     | utils.utils:display:55 - Extract Features: [ 0/47] Time  4.823 ( 4.823)    Data  0.570 ( 0.570)
evaluate lfw
    Accuracy: 0.99817+-0.00229
evaluate cfp
    Accuracy: 0.98400+-0.00451
evaluate agedb
    Accuracy: 0.98217+-0.00853
```
### 3.2 Quality Assessment
进入inference目录，然后执行如下命令获得特征信息（共512维）：  
```bash
python gen_feat.py --inf_list toy_imgs/img.list --feat_list toy_imgs/feat.list --resume ../weights/magface_epoch_00025.pth
```
获得结果保存至inference/toy_imgs/feat.list文件中，为tensor信息。  
执行如下命令获取质量得分：  
```bash
python Quality_Assessment.py
```
获得如下结果：
```bash
toy_imgs/0.jpg 21.78
toy_imgs/1.jpg 22.16
toy_imgs/2.jpg 24.09
toy_imgs/3.jpg 24.4
toy_imgs/4.jpg 25.9
toy_imgs/5.jpg 27.11
toy_imgs/6.jpg 27.41
toy_imgs/7.jpg 28.22
toy_imgs/8.jpg 28.73
toy_imgs/9.jpg 30.25
```
**画图：the error-versus-reject curve**  

在**3.1** 中已经已经获得了Face Recognition特征，存放在eval/eval_recognition/features/magface_iresnet100目录下的**.list文件中，此时可以根据list保存值来获得**the error-versus-reject curve**  

在eval/eval_quality目录下，执行如下命令即可：  
```bash
./eval_quality.sh lfw
```
## 4. 训练

### 4.1 检测依赖环境
所需的环境依赖都在raw/requirements.txt文件中，查看是否缺少，如缺少则安装即可。
训练结果均保存在run/test目录下，包含所得**权重**及**log信息**。

### 4.2 数据集准备
如果训练开源数据集则按照1.2中准备即可，如果训练自己的数据集则需要将数据集进行调整，调整方式如下：  
Align images to 112x112 pixels with 5 facial landmarks ([code](https://github.com/deepinsight/insightface/blob/cdc3d4ed5de14712378f3d5a14249661e54a03ec/python-package/insightface/utils/face_align.py)).

### 4.3 数据集标签制定   
在EQFace模型训练时候使用的标签格式为.txt文件，内容为：  
/home/FaceQuality-master/rec2image/images/0_495950/165.jpg;0  

而在此模型中，对其进行修改，将.txt格式改为.list格式，内容改为：  
/home/FaceRocg/MagFace/dataset/images/0_495950/165.jpg 0

将训练所需要的标签放在dataloader目录下，dataloader/face_train_CASIA_WebFace.list备用。

### 4.4 训练
调整run.sh中的参数信息，并执行其进行训练即可。   
主要参数信息如下所示：主要修改了MODEL_ARC为**iresnet18**，train_list为**face_train_CASIA_WebFace.list**，此list文件是根据训练集所获取。
```bash
# settings
MODEL_ARC=iresnet18
OUTPUT=./test/

mkdir -p ${OUTPUT}/vis/

python -u trainer.py \
    --arch ${MODEL_ARC} \
    --train_list /home/FaceRocg/MagFace/dataloader/face_train_CASIA_WebFace.list \
    --workers 8 \
    --epochs 25 \
    --start-epoch 0 \
    --batch-size 512 \
    --embedding-size 512 \
    --last-fc-size 85742 \
    --arc-scale 64 \
    --learning-rate 0.1 \
    --momentum 0.9 \
    --weight-decay 5e-4 \
    --lr-drop-epoch 10 18 22 \
    --lr-drop-ratio 0.1 \
    --print-freq 100 \
    --pth-save-fold ${OUTPUT} \
    --pth-save-epoch 1 \
    --l_a ${la} \
    --u_a ${ua} \
    --l_margin ${lm} \
    --u_margin ${um} \
    --lambda_g ${lg} \
    --vis_mag 1    2>&1 | tee ${OUTPUT}/output.log  
```
最终获得的训练权重重命名为**self_CASIA_WebFace_iresnet18.pth**，测试时使用此权重即可。
## 5. 训练后测试
### 5.1 对标3.1测试精度
eval/eval_recognition目录下执行命令：
```bash
 ./eval.sh ../../run/test/self_CASIA_WebFace_iresnet18.pth official 18
```
获得结果如下：
```bash
evaluate lfw
    Accuracy: 0.99167+-0.00342
evaluate cfp
    Accuracy: 0.92086+-0.01337
evaluate agedb
    Accuracy: 0.92783+-0.01197
```
### 5.2 对标3.2测试质量分数
进入inference目录，执行如下命令：
```bash
python gen_feat.py --arch iresnet18 --inf_list toy_imgs/img.list --feat_list toy_imgs/feat.list --resume ../run/test/self_CASIA_WebFace_iresnet18.pth

python Quality_Assessment.py
```
获得如下结果：
```bash
toy_imgs/1.jpg 7.85
toy_imgs/0.jpg 9.96
toy_imgs/3.jpg 13.31
toy_imgs/5.jpg 14.06
toy_imgs/7.jpg 15.07
toy_imgs/4.jpg 15.18
toy_imgs/2.jpg 15.61
toy_imgs/6.jpg 16.43
toy_imgs/8.jpg 17.25
toy_imgs/9.jpg 22.4
```
**总而言之，测试结果较为满意。**

## 6. 转onnx模型并使用onnx模型推理
### 6.1 转onnx模型
在inference/gen_feat.py文件中设置了转onnx的开关及功能，将`--pytorch2onnx`设置为`True`，后执行如下命令即可生成onnx模型：
```bash
python gen_feat.py --arch iresnet18 --inf_list toy_imgs/img.list --feat_list toy_imgs/feat.list --resume ../run/test/self_CASIA_WebFace_iresnet18.pth
```
### 6.2 关于使用onnx模型推理
此部分内容在export目录下，可自行学习。

## 关于报错
### 报错1
- 报错内容
```bash
    File "../../utils/utils.py", line 16, in <module>
    from termcolor import cprint
    ModuleNotFoundError: No module named 'termcolor'
```
- 解决方案  
执行命令安装此库
```bash
pip --default-timeout=1000 install termcolor -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
```

### 报错2
- 报错内容(训练时执行run.sh时的报错)
```bash
Traceback (most recent call last):
  File "trainer.py", line 258, in <module>
    main(args)
  File "trainer.py", line 106, in main
    main_worker(ngpus_per_node, args)
  File "trainer.py", line 127, in main_worker
    train_loader = dataloader.train_loader(args)
  File "../dataloader/dataloader.py", line 51, in train_loader
    train_dataset = MagTrainDataset(
  File "../dataloader/dataloader.py", line 21, in __init__
    self.init()
  File "../dataloader/dataloader.py", line 32, in init
    self.targets.append(int(data[2]))
IndexError: list index out of range
```
- 解决方案
```python
        for line in f.readlines():
            data = line.strip().split(' ')
            self.im_names.append(data[0])
            self.targets.append(int(data[1]))
```
将dataloader.pyL32中的2改为1。


## 参考
关于转onnx和tensorrt的参考[链接](https://github.com/tonhathuy/tensorrt-triton-magface)。
