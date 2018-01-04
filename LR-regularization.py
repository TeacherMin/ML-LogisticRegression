#coding=utf-8

from numpy import *
from pylab import *
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures

#####加载数据#######
def loaddata():
	data = loadtxt('C:\Users\DELL\Desktop\logistic\data2.txt', delimiter=',')
	X= data[:, 0:2] 
	y= data[:, 2:]	
	return data,X,y

#######绘出数据分布#######
def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # 获得正负样本的下标(即哪些是正样本，哪些是负样本)
    neg = data[:,2] == 0
    pos = data[:,2] == 1
    
    if axes == None:
        axes = gca()
    axes.scatter(data[pos][:,0], data[pos][:,1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:,0], data[neg][:,1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon= True, fancybox = True);
	
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
	D=1.0+exp(-X)
	return N/D

#####代价函数#####
def costFunction(theta,lamda,*args):
	m = y.size
	h = sigmoid(X.dot(theta))
	
	J = -1.0*(1.0/m)*(log(h).T.dot(y)+log(1-h).T.dot(1-y)) + (lamda/(2.0*m))*sum(square(theta[1:]))
	return J
	
#####梯度#####
def gradient(theta,lamda,*args):
	m = y.size
	h = sigmoid(X.dot(theta.reshape(-1,1)))     
	grad = (1.0/m)*X.T.dot(h-y) + (lamda/m)*r_[[[0]],theta[1:].reshape(-1,1)]
	return(grad.flatten())
	
#####预测#####
def predict(theta, X, threshold=0.5):
    p = sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))

#=====================================================功能测试==============================
data,x,y=loaddata()                        #加载数据

poly = PolynomialFeatures(6)          #使用sklearn中的方法进行特征映射
X = poly.fit_transform(x)

init_theta=zeros([X.shape[1],1])      #初始化参数为0
init_cost=costFunction(init_theta,1,X,y)
print init_cost


lam=0                                 #调节lambda以观察决策边界的变化
res = minimize(costFunction, init_theta, args=(lam, X, y), jac=gradient, options={'maxiter':3000})

#======================================================画出决策边界===========================
plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0')
x1_min, x1_max = x[:,0].min(), x[:,0].max()
x2_min, x2_max = x[:,1].min(), x[:,1].max()
xx1, xx2 = meshgrid(linspace(x1_min, x1_max), linspace(x2_min, x2_max))
h = sigmoid(poly.fit_transform(c_[xx1.ravel(), xx2.ravel()]).dot(res.x))
h = h.reshape(xx1.shape)
contour(xx1, xx2, h, [0.5], linewidths=1, colors='g'); 
show()




