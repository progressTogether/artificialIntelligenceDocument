# 决策树
非参数模型：不能用有限个参数来描述，随样本数量变化。

**优点：**

- 容易解释
- 可扩展到大规模数据，
- 不要求对特征做预处理
	- 能处理离散和连续值混合的输入
	- 对特征的单调变换，如log、标准化等，不敏感，只与数据的排序有关；
	- 能自动进行特征选择；
	- 可处理缺失数据等。

**缺点：**

- 预测正确率不高。+boosing=GBDT
- 模型不稳定，输入数据小的变化（如一两个数据点的取值变化）会带来树结构的变化。+bagging=RamdomForest
- 样本太少时容易过拟合

## 如何建树

### 建树目标
使训练集上模型的预测值与真值差距越来越小

### 建树过程
 1. 根节点包含全部样本
 2. 分裂。目标：减小该节点的<a href="#不纯净度量" target="_blank"> **[ 不纯净度]**</a>。方法：对特征j和阈值T，小于的样本分到左子节点，大于的样本分到右子节点。对左右节点分别计算节点的不纯净度，加权平均作为分裂后的总不纯净度，与父节点的不纯净度进行比较。选择**【分裂后的总不纯净度】**最小的特征j和阈值T进行分裂。sklearn中DecisionTree穷举搜索所有特征的所有可能取值，把连续特征当作离散特征处理，没有实现剪枝。
 3. 继续对左右子节点进行分裂。
 4. 停止分裂。（1）不纯净度减少太少；（2）树的深度超过了最大深度，或叶子节点超过一定数目；（3）左右分支的样本分布足够纯净；（4）左右分支中样本数目足够少。

### 建树指标
#### <a name="不纯净度量"> 不纯净度</a>
- **分类决策树**
	- 错分率：$H(D)=\frac{1}{|D|} \sum_{i \in D} I(y_i \neq \hat{y})=1-\hat{\pi}_\hat{y}$
	- 熵：$H(D)=-\sum_{c=1}^{C} \hat{\pi}_c log \hat{\pi}_c$
	- Gini系数：$H(D)=\sum_{c=1}^{C} \hat{\pi}_c (1-\hat{\pi}_c)=1-\sum_C \hat{\pi}_c^2 $
	
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;其中$\hat{\pi}_c=\frac{1}{|D|}\sum_{i \in D} I(y_i =c)$
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;以5个样本为例，类别分别为[1,1,2,3,4]，$\hat{\pi}_1=\frac{2}{5}$，$\hat{\pi}_2=\hat{\pi}_3=\hat{\pi}_4=\frac{1}{5}$，$\hat{y}=1$，
错分率为$\frac{3}{5}$，熵为$-(\frac{2}{5} log \frac{2}{5}+\frac{3}{5} log \frac{1}{5})$,Gini系数为$1-((\frac{2}{5}) ^2+3*(\frac{1}{5}) ^2)$

- **回归决策树**
	- 属于某一结点的所有样本的y的方差，即L2损失

## 如何剪枝

### 剪枝描述
使用校验集来进行剪枝，类似线性模型中的正则项，保证模型复杂度不要太高，防止过拟合

### 剪枝准则：Cost complexity pruning

$$CC(T)=Err(T)+\alpha |T|$$
其中$Err(T)$代表树的错误率，$\alpha$是正则因子，$|T|$是树的节点数目。形同机器学习模型的目标函数：$J(\theta)=\sum_{i=1}^{N} L(f(X_i;\theta),y_i)+\lambda \Omega(\theta)$

### 剪枝过程
自底向上进行剪枝，直至根节点。
当$\alpha$从0开始增大，树的一些分支被剪掉，得到不同$\alpha$对应的树。采用交叉验证得到最佳$\alpha$。

----------


# GBDT（Gradient boosting descision tree)

## Boosting 与 AdaBoost

http://www.jianshu.com/p/a6426f4c4e64
boosting描述：模型输出为多个弱学习器的加权平均
adaptive boosting描述：初始每个样本的权重（分布概率）均为$\frac{1}{N}$，训练得到一个分类器后对样本做预测。对错误率<$\frac{1}{2}$的分类器，降低正确分类的样本权重，提高误分样本的权重；对错误率>$\frac{1}{2}$的分类器，增加正确分类的样本权重，降低误分样本的权重；错误率=$\frac{1}{2}$的分类器的权重为0。继续训练下一个分类器。

**如何选择弱学习器的权重$\alpha$和样本的权重$w$?**

----------

### 确定样本权重w

目标：指数损失最小。考虑两类分类问题，样本标签$y_i \in \{-1,1\}$。多类问题可通过ovr策略进行扩展。
$$
\begin{align}
ERR_{train} & =\frac{1}{N} \sum_{i=0}^{N}
\begin{cases}
1 & y_i \neq sgn(f(X_i)) \\
0 & else
\end{cases}\\
& \leq \frac{1}{N} \sum_{i=0}^N exp(-y_if(X_i))\\
& = \frac{1}{N} \sum_{i=0}^N exp(-y_i (\alpha_1 f_1(X_i)+\alpha_2 f_2(X_i)+ \dots +\alpha_M f_M(X_i)))\\
& = \frac{1}{N} \sum_{i=0}^N [\prod_{m=1}^{M}  exp(-y_i \alpha_m f_m(X_i)]\\
假设存在w_{m,i}使\\
& = \prod_{m=1}^{M} [\sum_{i=0}^N w_{m,i} exp(-y_i \alpha_m f_m(X_i))]
\end{align}
$$
如何求解$w_{m,i}$?设
$$
R_{M,i}=\frac{\prod_{m=1}^{M}  exp(-y_i \alpha_m f_m(X_i))}{ \prod_{m=1}^{M} [\sum_{i=0}^N w_{m,i} exp(-y_i \alpha_m f_m(X_i))]}
$$
原问题等价于求解$w_{m,i}$使得$\sum_{i=1}^N R_{M,i}=\frac{1}{N}$。

当$M=1$时
$$
\frac{1}{N} \sum_{i=0}^{N} exp(-y_i \alpha f(X_i))=\sum_{i=0}^N w_{1,i} exp(-y_i \alpha f(X_i))
$$
可得$w_{1,i}=\frac{1}{N}$

对于$M>=2$
$$
R_{M+1,i}=\frac{exp(-y_i \alpha_{M+1} f_{M+1}(X_i))}{ \sum_{i=0}^N w_{M+1,i} exp(-y_i \alpha_{M+1} f_{M+1}(X_i))} R_{M,i}
$$
对所有样本求和，可得
$$
\sum_{i=0}^N R_{M,i}exp(-y_i \alpha_{M+1} f_{M+1}(X_i))=\frac{1}{N} \sum_{i=0}^N w_{M+1,i} exp(-y_i \alpha_{M+1} f_{M+1}(X_i))
$$
若令$w_{M+1,i}=\frac{1}{N}R_{M,i}$，则上式成立。进而有
$$
\begin{align}
w_{M+1,i} 
& =\frac{1}{N} R_{M,i}\\
& =\frac{1}{N} \frac{exp(-y_i \alpha_{M} f_{M}(X_i))}{ \sum_{i=0}^N w_{M,i} exp(-y_i \alpha_{M} f_{M}(X_i))} R_{M-1,i}\\
& =\frac{exp(-y_i \alpha_{m} f_{m}(X_i))}{ \sum_{i=0}^N w_{m,i} exp(-y_i \alpha_{m} f_{m}(X_i))} w_{M,i}
\end{align}
$$

即为样本权重更新公式。


----------

### 确定弱学习器权重$\alpha$

目标：指数损失最小。此时损失已可以表述为
$$
\begin{align}
ERR_{train} & =\prod_{m=1}^{M} [\sum_{i=0}^N w_{m,i} exp(-y_i \alpha_m f_m(X_i))]\end{align}
$$
令$Z_m=\sum_{i=0}^N w_{m,i} exp(-y_i \alpha_m f_m(X_i))$，对$\alpha$求偏导令其为0
$$
\begin{align}
\frac{\partial Z_m}{\partial \alpha_m}
& =-\sum_{i=0}^N w_{m,i} y_i  f_m(X)exp(-y_i \alpha_m f_m(X_i))\\
& =
\begin{cases}
-\sum_{X_i \in A} w_{m,i} exp(-\alpha_m ) & if X_i \in A,A=\{X_i:y_i f_m(X_i)=1\}  & 分类正确样本集合\\
\sum_{X_i \in \bar{A}} w_{m,i} exp(\alpha_m ) & if X_i \in \bar{A},\bar{A}=\{X_i:y_i f_m(X_i)=-1\}  & 分类错误样本集合\\
\end{cases}\\
&=0
\end{align}
$$

$$
\sum_{X_i \in A} w_{m,i} exp(-\alpha_m ) =\sum_{X_i \in \bar{A}} w_{m,i} exp(\alpha_m ) 
$$

$$
\sum_{X_i \in A} w_{m,i} =\sum_{X_i \in \bar{A}} w_{m,i} exp(2\alpha_m )
$$

$$
\alpha_m=\frac{1}{2} log \frac{\sum_{X_i \in A} w_{m,i} }{\sum_{X_i \in \bar{A}} w_{m,i} }=\frac{1}{2}log \frac{1-\epsilon_m}{\epsilon_m}
$$

其中$\epsilon_m=\frac{\sum_{X_i \in \bar{A}} w_{m,i}}{\sum w_{m,i}}$为第m个分类器的分类误差，即正确率高的弱分类器权重更大。


## Gradient Boosting

$$f_m(x)=f_{m-1}(x)+\eta \phi_m(x)$$
其中$f_m(x)$是第m次迭代获得的分类器，$\phi_m(x)$通过拟合损失函数对f(x)的负梯度得到，$\eta$是学习率，也称步长。该算法的思想源于一阶泰勒展开。

----------


# Random Forest(bagging)
模型描述：
$$\hat{f}_{avg}(x)=\frac{1}{B} \sum_{b=1}^B \hat{f}^b (x)$$
其中$\hat{f}^b (x)$是森林中的某棵决策树的预测结果。每一棵决策树通过（1）随机选择一部分特征；（2）随机选择一部分样本，对原N个样本的N次有放回抽样，重复B次以获得训练B颗树的数据，即Bootstrap Aggregating，训练得到。

