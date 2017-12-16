# 环境

> Python 环境： 3.5


# 说明
该项目是参考[用机器学习玩转恶意URL检测]后的改进版本，在svm，logistics分类器的基础上加入了neuralnetwork,而svm，logistics依赖的库可以参见那篇文章。
运行test.py即可知道svm,logistic回归怎么运行,而运行neural_network.py则知道神经网络如何训练,网络的构架在dnn_app_utils_v2.py文件中。
详细的设计思路及分析可见
[通过机器学习识别恶意url](http://blog.csdn.net/solo_ws/article/details/77095341)

## 数据集
good_fromE 某系统的某天的正常访问url，已去重
good_fromE2 同上
bad_fromE 利用sql注入某系统产生的url记录
badqueries 来源于网上数据
goodqueries 来源于网上数据


## 参数设定
neural_network.py中有关网络参数的调整以及kmean的聚类维数都在py文件中,比较懒，参数全放在py文件中前面设定。

## 模型
.label 文件保存的是分词结果，只与数据集有关
.pickle是训练后的模型，各位按需自取

