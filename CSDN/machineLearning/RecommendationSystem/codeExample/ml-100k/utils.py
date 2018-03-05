# coding: utf-8

# 特征编码工具
#该事件涉及性别、职业和zip code等信息的处理

from __future__ import division 
import numpy as np
from collections import defaultdict

import numpy as np  
import scipy as sp  
from numpy.random import random  

#数据集中的用户数目和电影数目
n_Users = 943
n_Movies = 1682

#类别型特征编码
#这里写成类的形式，因为编码字典要在不同的文件中使用
class FeatureEng:
  def __init__(self):
   
    # 载入 sex id 字典
    ##缺失补0，性别未知
    self.SexIdMap = defaultdict(int, {'NaN': 0, "M":1, "F":2})
    
    # 载入occupation 字典
    self.occupationIdMap = defaultdict(int)
    filename = "u.occupation"
    i= 0
    for line in open(filename,'r'):  #对每条记录
        occupation = line.strip('\n')
        self.occupationIdMap[occupation.lower()] = i+1
        i += 1
    #print self.occupationIdMap

  def getSexId(self, sexStr):
    return self.SexIdMap[sexStr]

  def getoccupationId(self, occupation_str):
    return self.occupationIdMap[occupation_str.lower()]

  def getzip_codeInt(self, zip_code):
    try:
      return int(zip_code)
    except:
      return 0

  def getDate(self, date):
    try:
      if date == NaT:  #缺失值处理
        return 0
    
      n_years = date.year - 1995
      n_months = n_years*12 + date.month - 1
      n_days = date.day - 1 
      
      n_days += n_months*30
    
      return n_days
   
    except:
      return 0