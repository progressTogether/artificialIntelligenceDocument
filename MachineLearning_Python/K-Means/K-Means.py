#-*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy import io as spio
from scipy import misc      # 图片操作
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
font =  FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',size=20) # 解决ubuntu环境下画图汉字乱码问题
rcParams['axes.unicode_minus']=False #解决负号‘-‘显示为方块的问题


'''
    K-means 理论基础及具体步骤
    1、给定数据                                                 
    2、确定类别数目K（如K=5），并初始化K个类的中心（如随机选择K个点）   -----对应函数----kMeansInitCentroids
    3、对每个样本点，计算离其最近的类（使得每个类拥有自己的一些数据）
       该样本距离那个聚类中心近则将其划到该类下                      -----对于函数----findClosestCentroids
    4、对每个类，计算其所有数据的中心，并跳到新的中心                 -----对于函数----computerCentroids           
    5、重复3、4，直到数据点所属类别不再改变。
'''

def KMeans():
    '''二维数据聚类过程演示'''
    print (u'聚类过程展示...\n')
    data = spio.loadmat("data.mat")
    X = data['X']
    K = 3   # 总类数
    #initial_centroids = np.array([[3,3],[6,2],[8,5]])   # 初始化类中心
    initial_centroids = kMeansInitCentroids(X,K)
    max_iters = 5
    runKMeans(X,initial_centroids,max_iters,True,'test')       # 执行K-Means聚类算法
    '''
    图片压缩
    '''
    print (u'K-Means压缩图片\n')
    img_data = misc.imread("bird.png")  # 读取图片像素数据
    img_data = img_data/255.0             # 像素值映射到0-1
    img_size = img_data.shape
    X = img_data.reshape(img_size[0]*img_size[1],3)    # 调整为N*3的矩阵，N是所有像素点个数
    
    K = 16
    max_iters = 5
    initial_centroids = kMeansInitCentroids(X,K)
    centroids,idx = runKMeans(X, initial_centroids, max_iters, True,'bird')
    print (u'\nK-Means运行结束\n')
    print (u'\n压缩图片...\n')
    idx = findClosestCentroids(X, centroids)
    X_recovered = centroids[idx,:]
    X_recovered = X_recovered.reshape(img_size[0],img_size[1],3)
    
    print (u'绘制图片...\n')
    plt.subplot(1,2,1)
    plt.imshow(img_data)
    plt.title(u"原先图片",fontproperties=font)
    plt.subplot(1,2,2)
    plt.imshow(X_recovered)
    plt.title(u"压缩图像",fontproperties=font)
    plt.savefig('./origin_compass_bird.png')
    print (u'运行结束！')
    
    
# 找到每条数据距离哪个类中心最近    
def findClosestCentroids(X,initial_centroids):
    m = X.shape[0]                  # 数据条数
    K = initial_centroids.shape[0]  # 类的总数
    dis = np.zeros((m,K))           # 存储计算每个点分别到K个类的距离
    idx = np.zeros((m,1))           # 要返回的每条数据属于哪个类
    
    '''计算每个点到每个类中心的距离'''
    for i in range(m):
        for j in range(K):
            '''reshape(1,-1)指定行数为1，列数设置为-1为default，会在满足行数的情况下满足列数'''
            '''np.dot()  输入为向量（即  nx1）得到为俩个向量相乘在相加，例如np.dot([1,2],[3,4]) = 11'''
            dis[i,j] = np.dot((X[i,:]-initial_centroids[j,:]).reshape(1,-1),(X[i,:]-initial_centroids[j,:]).reshape(-1,1))
            '''dis[i,j],样本数为i对第j个类别的距离'''
    '''返回dis每一行的最小值对应的列号，即为对应的类别
    - np.min(dis, axis=1)返回每一行的最小值ss
    - np.where(dis == np.min(dis, axis=1).reshape(-1,1)) 返回对应最小值的坐标
     - 注意：可能最小值对应的坐标有多个，where都会找出来，所以返回时返回前m个需要的即可（因为对于多个最小值，属于哪个类别都可以）
    '''
    dummy,idx = np.where(dis == np.min(dis, axis=1).reshape(-1,1))
    #idx为样本对应的类别，其长度为样本的数目，所以可以实现将每个样本对应到指定的类别下
    print ('The shape of idx is',idx.shape[0])
    print ('dis.shape[0]=',dis.shape[0])
    # 注意截取一下，可能这里会有特殊情况，但从log来看基本上 dis.shape[0] == idx.shape[0]
    #特殊情况 比如某个样本到俩个或者俩个以上的距离相等
    return idx[0:dis.shape[0]]
             

# 计算类中心
def computerCentroids(X,idx,K):
    n = X.shape[1]
    print ('特征数目',n)
    centroids = np.zeros((K,n))
    '''
    大致思路为将所有样本中idx相同的同一特征数相加求平均值，构成类中心
    举个例子  样本1 = [1,2,3,4,5,6]  其idx = 1
             样本2 = [3,4,5,6,7,8] 其idx = 1
    那么idx=1的类中心为[2,3,4,5,6,7] 
    备注：idx = 1情况下只有2个样本
    '''
    for i in range(K):

        '''
        idx = np.array([0 ,1 ,2,1,2])
        
        test3 = np.array([[1,2,3],
                          [3,4,5],
                          [7,8,9],
                          [9,10,11],
                          [12,13,14]])
        print (test3.shape[1]) #test3.shape[1] = 3 所以之后会生成 3x3 类别数目x特征数目
        for i in range(3):
            print ( np.mean( test3[np.ravel(idx==i),:],axis=0) )#axis=0是按行求平均值
        #输出结果 [1. ,2. ,3] [6.,7.,8.] [9.5,10.5,11.5] 类别数目x特征数目
        #当 i = 0时 取idx为0的项[1,2,3] 所以mean = [1. ,2. ,3]
        #当 i = 1时 取idx为1的项[3,4,5]  [9,10,11]所以mean = [6.,7.,8.]
        #当 i = 2时 取idx为2的项[7,8,9]  [12,13,14]所以mean = [9.5,10.5,11.5]
        '''
        centroids[i,:] = np.mean(X[np.ravel(idx==i),:], axis=0).reshape(1,-1)   # 索引要是一维的,axis=0为每一列，idx==i一次找出属于哪一类的，然后计算均值
    return centroids    #centroids结果为Kxn，即类别数目x特征数目 

# 聚类算法
def runKMeans(X,initial_centroids,max_iters,plot_process,filename):
    m,n = X.shape                   # 数据条数和维度
    K = initial_centroids.shape[0]  # 类数
    centroids = initial_centroids   # 记录当前类中心
    previous_centroids = centroids  # 记录上一次类中心
    idx = np.zeros((m,1))           # 每条数据属于哪个类
    all_centroids = []
    all_centroids.append(centroids)
    for i in range(max_iters):      # 迭代次数
        print (u'迭代计算次数：%d'%(i+1) )
        idx = findClosestCentroids(X, centroids)
        if plot_process:    # 如果绘制图像
            plt = plotProcessKMeans(X,centroids,previous_centroids) # 画聚类中心的移动过程
            print (previous_centroids.shape)
            previous_centroids = centroids  # 重置
            result_filename = 'after_compress' + filename + str(i)
            plt.savefig(result_filename)
            plt.clf()
        centroids = computerCentroids(X, idx, K)    # 重新计算类中心
        all_centroids.append(centroids)
    plotAllProcessKMeans(X,all_centroids,filename)

    #if plot_process:    # 显示最终的绘制结果
        #plt.show()
    return centroids,idx    # 返回聚类中心和数据属于哪个类

def plotAllProcessKMeans(X,all_centroids,filename):
    plt.scatter(X[:,0], X[:,1])     # 原数据的散点图
    number,Row,column = np.shape(all_centroids)
    for i in range(number):
        plt.plot(all_centroids[i][:,0],all_centroids[i][:,1],'rx',markersize=10,linewidth=5.0)
        for j in range(Row-1):
            p1 = all_centroids[i][j,:]
            p2 = all_centroids[i][j + 1,:]
            plt.plot([p1[0],p2[0]],[p1[1],p2[1]],"->",linewidth=2.0)
    filename = filename + 'all.png'
    plt.savefig(filename)


# 画图，聚类中心的移动过程        
def plotProcessKMeans(X,centroids,previous_centroids):
    plt.scatter(X[:,0], X[:,1])     # 原数据的散点图
    plt.plot(previous_centroids[:,0],previous_centroids[:,1],'rx',markersize=10,linewidth=5.0)  # 上一次聚类中心
    plt.plot(centroids[:,0],centroids[:,1],'rx',markersize=10,linewidth=5.0)                    # 当前聚类中心
    for j in range(centroids.shape[0]): # 遍历每个类，画类中心的移动直线
        p1 = centroids[j,:]
        p2 = previous_centroids[j,:]
        plt.plot([p1[0],p2[0]],[p1[1],p2[1]],"->",linewidth=2.0)
    return plt


# 初始化类中心--随机取K个点作为聚类中心
def kMeansInitCentroids(X,K):
    m = X.shape[0]
    m_arr = np.arange(0,m)      # 生成0-m-1
    centroids = np.zeros((K,X.shape[1]))
    np.random.shuffle(m_arr)    # 打乱m_arr顺序    
    rand_indices = m_arr[:K]    # 取前K个
    centroids = X[rand_indices,:]  #从X中随机选取K个样本作为centroids
    return centroids

if __name__ == "__main__":
    KMeans()
    