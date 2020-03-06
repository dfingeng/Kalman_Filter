
#此示例使用自己写的公式
import numpy as np
import matplotlib.pyplot as plt

#https://www.cnblogs.com/USTC-ZCC/p/10018911.html

List1=np.arange(0,102)
data=np.arange(0,102,dtype='float32')
data2=np.zeros(102,dtype='float32')
noise = np.random.normal(0, 10, 102) #生成0~5,20个随机数
Y=30
# Temp=[23,24,25,26,24,26,28,25,32,34,26,28,36,32,42,38,38,42,44,36]  #测量值

# data.dtype('float32')
# data2.dtype('float32')


for i in range(0,102):
    data[i]=noise[i]+List1[i]
data2[0]=1

Temp_e=8 #温度计不确定度是4
Me_e=4  #自己不确定度是4
frist_e=3 #上一刻不确定度是3，为初始化的值
piancha=(frist_e**2+Me_e**2)**0.5

for i in range(0,100):
    kg=(piancha**2/(piancha**2+Temp_e**2))**0.5
    data2[i+1]=data[i]+kg*(data[i]-data2[i])
    piancha=(((1-kg)*piancha)**2)**0.5
    print(kg)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()  # 指明ax2是ax1的镜像

# plt.subplot(3,1,2) #1行2列的图。设置图片2的位置为3
ax1.plot(List1[0:101],data2[0:101],alpha=0.8,c='Red')

# plt.subplot(3,1,3) #1行2列的图。设置图片2的位置为3
ax2.scatter(x=List1[0:101],y=data[0:101],s=10,alpha=0.8,c='Green')

plt.show()
