import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=",")
x_data = data[:,0]
y_data = data[:,1]
plt.scatter(x_data,y_data)
plt.show()
print(x_data.shape)


k=input("输入k的数值：")
k=int(k)

#中心化
def zeroMean(dataMat):
    # 按列求平均，即各个特征的平均
    meanVal = np.mean(dataMat, axis=0) 
    newData = dataMat - meanVal
    return newData, meanVal
#求协方差矩阵
newData,meanVal=zeroMean(data)  
covMat = np.cov(newData, rowvar=0)

#求矩阵的特征值和特征向量
eigVals, eigVects = np.linalg.eig(np.mat(covMat))

#对特征值排序
eigValIndice = np.argsort(eigVals)

#取最大的k个特征值下标
n_eigValIndice = eigValIndice[-1:-(k+1):-1]

#最大的k个特征值对应的特征向量
n_eigVect = eigVects[:,n_eigValIndice]
lowDDataMat = newData*n_eigVect
reconMat = (lowDDataMat*n_eigVect.T) + meanVal


# data = np.genfromtxt("data.csv", delimiter=",")
# x_data = data[:,0]
# y_data = data[:,1]
# plt.scatter(x_data,y_data)

# 重构的数据
x_data = np.array(reconMat)[:,0]
y_data = np.array(reconMat)[:,1]
plt.scatter(x_data,y_data,c='r')
plt.show()
