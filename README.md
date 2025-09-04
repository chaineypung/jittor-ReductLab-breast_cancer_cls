| 第五届计图挑战赛赛道一：rank1

# Jittor 超声肿瘤乳腺癌Bi-Rads分级 


## 简介

针对超声乳腺分类任务，我们从多方面着手优化：  1、 ArcMargin 辅助头损失拉大不同类的特征空间；  2、 “类概率“标签蒸馏；  3、 类别不平衡+难样本损失；  4、 改进ConvNeXt输出头；...... 

## 安装 

#### 运行环境
- ubuntu 22.04 LTS
- python >= 3.10
- jittor >= 1.3.9.14

#### 安装依赖
执行以下命令安装 python 依赖
```
conda create -n jittor python=3.11.2
conda activate jittor 
pip install -r requirements.txt
```

## 训练

单卡训练可运行以下命令：
```
python train_convnext.py
python trian_effv2.py
```


## 推理

生成测试集上的结果可以运行以下命令：

```
python infer.py
```


## 项目地址

```
https://github.com/chaineypung/jittor-ReductLab-breast_cancer_cls
```
