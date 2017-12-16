from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.cluster import KMeans
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
import dnn_app_utils_v2 as dnn
import pickle
import urllib
import time
from PIL import Image
from scipy import ndimage


# 各十条数据作为例子分析
# 输入参数部分
good = 'data/good_fromE2.txt'
bad = 'data/badqueries.txt'
n = 2
use_k = True
k = 80


def printT(word):
    a = time.strftime('%Y-%m-%d %H:%M:%S: ', time.localtime(time.time()))
    print(a+str(word))


# return[good,bad]
def getdata():

    with open(good,'r') as f:
        good_query_list = [i.strip('\n') for i in f.readlines()[:]]
    with open(bad,'r') as f:
        bad_query_list = [i.strip('\n') for i in f.readlines()[:]]
    return [good_query_list, bad_query_list]


class IDS(object):
    pass


# In[8]:

#     训练模型基类
class Baseframe(object):

    def __init__(self):

        printT('读入数据，good：'+good+' bad:'+bad)
        data = getdata()
        printT('done, good numbers:'+str(len(data[0]))+' bad numbers:'+str(len(data[1])))
        # 打标记
        good_y = [0 for i in range(len(data[0]))]
        bad_y = [1 for i in range(len(data[1]))]
        
        y = good_y + bad_y

        #     向量化
        # converting data to vectors  定义矢量化实例
        self.vectorizer = TfidfVectorizer(tokenizer=self.get_ngrams)
        # 把不规律的文本字符串列表转换成规律的 ( [i,j],weight) 的矩阵X [url条数，分词总类的总数，理论上少于256^n] 
        # i表示第几条url，j对应于term编号（或者说是词片编号）
        # 用于下一步训练分类器 lgs
           
        
        X = self.vectorizer.fit_transform(data[0]+data[1])
        printT('向量化后维度：'+str(X.shape))
        # 通过kmeans降维 返回降维后的矩阵
        if use_k:
            X = self.transform(self.kmeans(X))

            printT('降维完成')
        
       
        #定义网络的结构
        n_x = 80
        n_h_1 = 80
        n_h_2 = 80
        
        n_y = 1
        layers_dims = (n_x,n_h_1,n_h_2,n_y)
        
        """print(type(X))
        print(X.shape)"""
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        #转化成深度学习可以使用的特征矩阵
        """混合的训练样本"""
        X_train = X_train.todense()
        X_train = X_train.T
        X_train = X_train.tolist()
        X_train = np.array(X_train)
        printT('X_train降维后维度：'+str(X_train.shape))
        Y_train = np.array(Y_train).T
        Y_train = Y_train.reshape((1,len(Y_train)))
        printT('y_train标签的维度：'+str(Y_train.shape))
        """混合的测试样本"""
        X_test = X_test.todense()
        X_test = X_test.T
        X_test = X_test.tolist()
        X_test = np.array(X_test)
        printT('X_test降维后维度：'+str(X_test.shape))
        Y_test = np.array(Y_test).T
        Y_test = Y_test.reshape((1,len(Y_test)))
        printT('y_test标签的维度：'+str(Y_test.shape))
        
        X = X.todense()
        X = X.T
        X = X.tolist()
        X = np.array(X)
        Y = np.array(y).T
        Y = Y.reshape((1,len(Y)))
        """纯的好样本"""
        X_good = X[:,0:len(data[0])]
        Y_good = Y[:,0:len(data[0])]
        printT('X_good降维后维度：'+str(X_good.shape))
        printT('Y_good降维后维度：'+str(Y_good.shape))
        """纯的差样本"""
        X_bad = X[:,len(data[0]):len(X[0])]
        Y_bad = Y[:,len(data[0]):len(Y[0])]
        printT('X_bad降维后维度：'+str(X_bad.shape))
        printT('Y_bad降维后维度：'+str(Y_bad.shape))
        
        
        
        
        
        #开始训练模型并且计算准确度”
        self.parameters = self.L_layer_model(X_train, Y_train, layers_dims, num_iterations = 20000, print_cost = True)
        print("在混合的训练集上:")
        predictions_train = dnn.predict(X_train, Y_train, self.parameters)
        print("在混合的交叉验证集上:")
        predictions_test = dnn.predict(X_test, Y_test, self.parameters)
        print("在好的样本上:")
        predictions_train = dnn.predict(X_good, Y_good, self.parameters)
        print("在差的样本上:")
        predictions_train = dnn.predict(X_bad, Y_bad, self.parameters)
        
        
    def L_layer_model(self,X, Y, layers_dims, learning_rate = 0.01, num_iterations = 10000, print_cost=False):
        np.random.seed(1)
        costs = []
        parameters = dnn.initialize_parameters_deep(layers_dims)
        for i in range(0, num_iterations):
            AL, caches = dnn.L_model_forward(X, parameters)
            cost = dnn.compute_cost(AL, Y)
            grads = dnn.L_model_backward(AL, Y, caches)
            parameters = dnn.update_parameters(parameters, grads, learning_rate)
            if print_cost and i % 100 == 0:
                print("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                costs.append(cost)
            
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        
        return parameters

        
        
        
        
        
        
        
    def get_ngrams(self, query):
        tempQuery = str(query)
        ngrams = []
        for i in range(0, len(tempQuery)-n):
            ngrams.append(tempQuery[i:i+n])
        return ngrams

    def kmeans(self, weight):

        printT('kmeans之前矩阵大小： ' + str(weight.shape))
        weight = weight.tolil().transpose()
        # 同一组数据 同一个k值的聚类结果是一样的。保存结果避免重复运算
        try:

            with open('model/k' + str(k) + '.label', 'r') as input:

                printT('loading kmeans success')
                a = input.read().split(' ')

                self.label = [int(i) for i in a[:-1]]

        except FileNotFoundError:

            printT('Start Kmeans ')

            clf = KMeans(n_clusters=k, precompute_distances=False )

            s = clf.fit(weight)
            printT(s)

            # 保存聚类的结果
            self.label = clf.labels_

            # with open('model/' + self.getname() + '.kmean', 'wb') as output:
            #     pickle.dump(clf, output)
            with open('model/k' + str(k) + '.label', 'w') as output:
                for i in self.label:
                    output.write(str(i) + ' ')
        printT('kmeans 完成,聚成 ' + str(k) + '类')
        return weight

    #     转换成聚类后结果 输入转置后的矩阵 返回转置好的矩阵
    def transform(self, weight):

        from scipy.sparse import coo_matrix

        a = set()
        # 用coo存 可以存储重复位置的元素
        row = []
        col = []
        data = []
        # i代表旧矩阵行号 label[i]代表新矩阵的行号
        for i in range(len(self.label)):
            if self.label[i] in a:
                continue
            a.add(self.label[i])
            for j in range(i, len(self.label)):
                if self.label[j] == self.label[i]:
                    temp = weight[j].rows[0]
                    col += temp
                    temp = [self.label[i] for t in range(len(temp))]
                    row += temp
                    data += weight[j].data[0]

        # print(row)
        # print(col)
        # print(data)
        newWeight = coo_matrix((data, (row, col)), shape=(k,weight.shape[1]))
        return newWeight.transpose()
    
    
    
  
        
    
This=Baseframe()
parameters =This.parameters