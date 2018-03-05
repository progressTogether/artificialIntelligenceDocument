#-*- coding: utf-8 -*-
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler    #引入归一化的包

#一、使用sklearn.preprocessing.scale()函数，可以直接将给定数据进行标准化。
X = np.array( [[1.0,2.0,3.0],
               [4.0,5.0,6.0],
               [7.0,8.0,9.0]])
X_scaled = preprocessing.scale(X)
print ('经过标准化的X_scaled：\n',X_scaled)
print ('经过标准化之后的数据的期望：' ,X_scaled.mean(axis=0))
print ('经过标准化之后的数据的标准差：' ,X_scaled.std(axis=0))

#二、使用sklearn.preprocessing.StandardScaler类，使用该类的好处在于可以保存训练集中的参数（均值、方差）直接使用其对象转换测试集数据
# 归一化操作
scaler = StandardScaler()
scaler.fit(X)


test_result = scaler.transform(np.array([[7,5,3]]))#将保存的训练数据的参数应用到测试数据
print ('将保存的训练数据的参数应用到测试数据结果为：',test_result)
print ('期望',scaler.mean_)

#三、正则化（Normalization）
#可以使用preprocessing.normalize()函数对指定数据进行转换：
#正则化的过程是将每个样本缩放到单位范数（每个样本的范数为1），如果后面要使用如二次型（点积）或者其它核方法计算两个样本之间的相似性这个方法会很有用。
#Normalization主要思想是对每个样本计算其p-范数，然后对该样本中每个元素除以该范数，
# 这样处理的结果是使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1。
#p-范数的计算公式：||X||p=(|x1|^p+|x2|^p+...+|xn|^p)^1/p
#该方法主要应用于文本分类和聚类中。例如，对于两个TF-IDF向量的l2-norm进行点积，就可以得到这两个向量的余弦相似性。

#1、可以使用preprocessing.normalize()函数对指定数据进行转换：
X_normalized = preprocessing.normalize(X, norm='l2')
print ('经过正则化\n',X_normalized)

#2、可以使用processing.Normalizer()类实现对训练集和测试集的拟合和转换：
normalizer = preprocessing.Normalizer().fit(X)  # fit does nothing
normalizer.transform(X)
normalizer_in_test = normalizer.transform([[-1.,  1., 0.]])#
print ('经过正则化后测试集结果\n',normalizer_in_test)

#矩阵的乘法
#对应元素相乘 element-wise product: np.multiply(), 或 *
#在Python中，实现对应元素相乘，有2种方式，一个是np.multiply()，另外一个是*
Y = np.array( [[1.0,2.0,3.0],
               [4.0,5.0,6.0],
               [7.0,8.0,9.0]])

# 对应元素相乘 element-wise product
elment_wise_one = X * Y
print ('对应元素相乘方法* elment_wise_one:\n',elment_wise_one)
# 对应元素相乘 element-wise product
elment_wise_two = np.multiply(X , Y)
print ('对应元素相乘方法multiply elment_wise_two:\n',elment_wise_two)

#线性代数中矩阵乘法的定义： np.dot()
dot_x_y = np.dot(X,Y)
print ('矩阵相乘：\n',dot_x_y)

# 1-D array
one_dim_vec_one = np.array([1, 2, 3])
one_dim_vec_two = np.array([4, 5, 6])
one_result_res = np.dot(one_dim_vec_one, one_dim_vec_two)
print('one_result_res: %s' %(one_result_res))

#矩阵的加法
print ('矩阵的加法 按行：',elment_wise_one.sum(axis = 0))
print ('矩阵的加法 按列：',elment_wise_one.sum(axis = 1))
print ('矩阵的加法 按所有：',elment_wise_one.sum())


#写了一个测试函数模拟了preprocessing.Normalizer()
#更好的了解preprocessing.Normalizer()
def normalize_fun(X):
    elment_wise_X = X *X
    elment_sum = elment_wise_X.sum(axis = 1)
    elment_sqrt = [np.sqrt(elment_sum[i])  for i in range(len(elment_sum))]
    result = X
    for i in range(len(X)):
        for j in range(len(X[i])):
            result[i,j] = X[i,j]/elment_sqrt[i]
    print ('模拟Normalizer为了更好的理解Normalizer:\n',result)
    return result


normalize_fun(X)