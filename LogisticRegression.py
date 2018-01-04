#coding=utf-8

from numpy import *
from pylab import *
#from matplotlib.pyplot import *
from scipy.optimize import minimize
#####加载数据#######
def loaddata():
	data = loadtxt('C:\Users\DELL\Desktop\logistic\data1.txt', delimiter=',')
	x= data[:, 0:2] 
	y= data[:, 2:]	
	x0=ones([x.shape[0],1])    #在特征矩阵前增加一列，x0=1
	X=column_stack([x0,x]) 
	return X,y

#######绘出数据分布#######
def plotData(X,y):
	pos = where(y == 1)
	neg = where(y == 0)
	plot(X[pos, 1], X[pos, 2], 'bo')
	plot(X[neg, 1], X[neg, 2], 'rx')
	xlabel('Feature1/Exam 1 score')
	ylabel('Feature2/Exam 2 score')
	show()

#####决策边界#####
def plotDecisionBoundary(theta,X,y):
	pos = where(y == 1)
	neg = where(y == 0)
	plot(X[pos, 1], X[pos, 2], 'bo')
	plot(X[neg, 1], X[neg, 2], 'rx')
	xlabel('Feature1/Exam 1 score')
	ylabel('Feature2/Exam 2 score')
	
	x1_min, x1_max = X[:,1].min(), X[:,1].max()
	x2_min, x2_max = X[:,2].min(), X[:,2].max()
	xx1, xx2 = meshgrid(linspace(x1_min, x1_max), linspace(x2_min, x2_max))
	h = sigmoid(c_[ones((xx1.ravel().shape[0],1)), xx1.ravel(), xx2.ravel()].dot(theta))
	print h.shape
	h = h.reshape(xx1.shape)
	contour(xx1, xx2, h, [0.5], linewidths=1, colors='b') 
	
	show()
		
#####假设函数#####
def sigmoid(X):
	N=1.0
	D=1.0+exp(X*(-1))
	return N/D

#####代价函数#####
def costFunction(theta, X, y):
	m = y.size
	h = sigmoid(X.dot(theta))
	J = -1.0*(1.0/m)*(log(h).T.dot(y)+log(1-h).T.dot(1-y))
	if isnan(J[0]):
		return(inf)
	return J[0]
	
#####梯度下降#####
def grad(theta,X,y):
    m = y.size
    h = sigmoid(X.dot(theta.reshape(-1,1)))  
    grad =(1.0/m)*X.T.dot(h-y)
    return(grad.flatten())
	
#####预测#####
def predict(theta,X):
	m=X.shape[0]
	h=sigmoid(X.dot(theta))
	p=zeros([m,1])
	for i in range(m):
		if h[i]>=0.5:
			p[i]=1
		else:
			p[i]=0
	return p,h

#===========================================功能测试==============================
X,y=loaddata()                        #加载数据
init_theta=zeros([X.shape[1],1])      #初始化参数
init_cost=costFunction(init_theta,X,y)#初始代价 
print('init_cost=: ',init_cost)

res = minimize(costFunction, init_theta, args=(X,y), jac=grad, options={'maxiter':400})
theta=res.x
print theta

stu1=mat(array([1,45,85]))         #预测一个同学是否能被录取
print('prediction of stu1(exam1=45,exam2=85) is  :',predict(theta,stu1)) 

p,h=predict(theta,X)               #计算模型的准确率
print('precision of model is : ', mean(double(p==y))*100)

plotData(X, y)
plotDecisionBoundary(theta,X,y)