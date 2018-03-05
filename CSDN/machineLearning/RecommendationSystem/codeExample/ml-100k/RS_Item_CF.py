# coding: utf-8

#item based CF

from __future__ import division  
import numpy as np  
import scipy as sp  
class  Item_based_CF:  
    def __init__(self, X):  
        self.X = X  #评分表
        self.mu = np.mean(self.X[:,2])  #average rating

        self.ItemsForUser={}   #用户打过分的所有Item
        self.UsersForItem={}   #给Item打过分的所有用户
        
        for i in range(self.X.shape[0]):  
            uid=self.X[i][0]  #user id
            i_id=self.X[i][1] #item_id 
            rat=self.X[i][2]  #rating
            
            self.UsersForItem.setdefault(i_id,{})  
            self.ItemsForUser.setdefault(uid,{}) 
            
            self.UsersForItem[i_id][uid]=rat  
            self.ItemsForUser[uid][i_id]=rat
            
            #self.similarity.setdefault(i_id,{}) 
            
        pass  
    
        n_Items = len(self.UsersForItem)+1 #数组的索引从0开始，浪费第0个元素
        print n_Items-1
        self.similarity = np.zeros((n_Items, n_Items), dtype=np.float)
        self.similarity[:,:] = -1
           
    
    #计算Item i_id1和i_id2之间的相似性
    def sim_cal(self, i_id1, i_id2):
        if self.similarity[i_id1][i_id2]!=-1:  #如果已经计算好
            return self.similarity[i_id1][i_id2]  
        
        si={}  
        for user in self.UsersForItem[i_id1]:  #所有对Item1打过分的的user
            if user in self.UsersForItem[i_id2]:  #如果该用户对Item2也打过分
                #print self.UsersForItem[i_id2]
                si[user]=1  #user为一个有效用用户
        
        #print si
        n=len(si)   #有效用户数，有效用户为即对Item1打过分，也对Item2打过分
        if (n==0):  #没有共同打过分的用户，相似度设为1.因为最低打分为1？
            self.similarity[i_id1][i_id2]=0  
            self.similarity[i_id1][i_id1]=0  
            return 0  
        
        #所有有效用户对Item1的打分
        s1=np.array([self.UsersForItem[i_id1][u] for u in si])  
        
        #所有有效用户对Item2的打分
        s2=np.array([self.UsersForItem[i_id2][u] for u in si])  
        
        sum1=np.sum(s1)  
        sum2=np.sum(s2)  
        sum1Sq=np.sum(s1**2)  
        sum2Sq=np.sum(s2**2)  
        pSum=np.sum(s1*s2)  
        
        #分子
        num=pSum-(sum1*sum2/n)  
        
        #分母
        den=np.sqrt((sum1Sq-sum1**2/n)*(sum2Sq-sum2**2/n))  
        if den==0:  
            self.similarity[i_id1][i_id2]=0  
            self.similarity[i_id2][i_id1]=0  
            return 0  
        
        self.similarity[i_id1][i_id2]=num/den  
        self.similarity[i_id2][i_id1]=num/den  
        return num/den  
            
    #预测用户uid对Item i_id的打分
    def pred(self,uid,i_id):  
        sim_accumulate=0.0  
        rat_acc=0.0  
        
        if(i_id == 599):    
            print self.UsersForItem[i_id]
            
        for item in self.ItemsForUser[uid]:  #用户uid打过分的所有Item
            sim = self.sim_cal(item,i_id)    #该Item与i_id之间的相似度
            if sim<0:continue  
            #print sim,self.user_movie[uid][item],sim*self.user_movie[uid][item]  
            
            rat_acc += sim * self.ItemsForUser[uid][item]  
            sim_accumulate += sim  
        
        #print rat_acc,sim_accumulate  
        if sim_accumulate==0: #no same user rated,return average rates of the data  
            return  self.mu  
        return rat_acc/sim_accumulate  
    
    #测试
    def test(self,test_X):  
        test_X=np.array(test_X) 
        output=[]  
        sums=0  
        print "the test data size is ",test_X.shape  
        for i in range(test_X.shape[0]):  
            uid = test_X[i][0]  #user id
            i_id = test_X[i][1] #item_id 
        
            #设置默认值，否则用户或item没在训练集中出现时会报错
            self.UsersForItem.setdefault(i_id,{})  
            self.ItemsForUser.setdefault(uid,{})
            
            pre=self.pred(uid, i_id)  
            output.append(pre)  
            #print pre,test_X[i][2]  
            sums += (pre-test_X[i][2])**2  
        rmse=np.sqrt(sums/test_X.shape[0])  
        print "the rmse on test data is ",rmse  
        return output  