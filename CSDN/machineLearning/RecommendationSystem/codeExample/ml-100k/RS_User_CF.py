# coding: utf-8

#user based CF

from __future__ import division  
import numpy as np  
import scipy as sp  
class  User_based_CF:  
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
    
        n_Users = len(self.ItemsForUser)+1  #数组的索引从0开始，浪费第0个元素
        print n_Users-1
        self.similarity = np.zeros((n_Users, n_Users), dtype=np.float)
        self.similarity[:,:] = -1
           
    
    #计算user uid1和uid2之间的相似性
    def sim_cal(self, uid1, uid2):
        if self.similarity[uid1][uid2]!=-1:  #如果已经计算好
            return self.similarity[uid1][uid2]  
        
        si={}  
        for item in self.ItemsForUser[uid1]:  #uid1所有打过分的Item1
            if item in self.ItemsForUser[uid2]:  #如果uid2也对该Item打过分
                si[item]=1  #item为一个有效item
        
        #print si
        n=len(si)   #有效item数，有效item为即对uid对Item打过分，uid2也对Item打过分
        if (n==0):  #没有共同打过分的item，相似度设为1.因为最低打分为1？
            self.similarity[uid1][uid2]=0  
            self.similarity[uid1][uid2]=0  
            return 0  
        
        #用户uid1打过分的所有有效的item
        s1=np.array([self.ItemsForUser[uid1][item] for item in si])  
        
        #用户uid2打过分的所有有效的Item
        s2=np.array([self.ItemsForUser[uid2][item] for item in si])  
        
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
            self.similarity[uid1][uid2]=0  
            self.similarity[uid2][uid1]=0  
            return 0  
        
        self.similarity[uid1][uid2]=num/den  
        self.similarity[uid2][uid1]=num/den  
        return num/den  
            
    #预测用户uid对Item i_id的打分
    def pred(self,uid,i_id):  
        sim_accumulate=0.0  
        rat_acc=0.0  
            
        for user in self.UsersForItem[i_id]:  #对i_id打过分的所有用户
            sim = self.sim_cal(user,uid)    #该user与uid之间的相似度
            if sim<=0:continue  
            #print sim,self.user_movie[uid][item],sim*self.user_movie[uid][item]  
            
            rat_acc += sim * self.UsersForItem[i_id][user] 
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
            #print i
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