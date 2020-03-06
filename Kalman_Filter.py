
#此示例使用网上代码
import numpy as np
import matplotlib.pyplot as plt

# 创建一个0-99的一维矩阵
z = [i for i in range(100)]
z_watch = np.mat(z)  #观测值

# 创建一个方差为1的高斯噪声，精确到小数点后两位
noise = np.round(np.random.normal(0, 5, 100), 2)
noise_mat = np.mat(noise)

# 将z的观测值和噪声相加
z_mat = z_watch + noise_mat

# 定义x的初始状态
x_mat = np.mat([[0, ], [0, ]])
print('x_mat',x_mat)

# 定义初始状态协方差矩阵
p_mat = np.mat([[1, 0], [0, 1]])
print('p_mat',p_mat)

# 定义状态转移矩阵，因为每秒钟采一次样，所以delta_t = 1
f_mat = np.mat([[1, 1], [0, 1]])
print('f_mat',f_mat)

# 定义状态转移协方差矩阵，这里我们把协方差设置的很小，因为觉得状态转移矩阵准确度高
q_mat = np.mat([[0.0001, 0], [0, 0.0001]])
print('q_mat',q_mat)

# 定义观测矩阵
h_mat = np.mat([1, 0])
print('h_mat',h_mat)

# 定义观测噪声协方差
r_mat = np.mat([1])
print('r_mat',r_mat)

result=np.arange(0,100,1,dtype='float32')
plt.subplot(3, 1, 1) #1行2列的图。设置图片1的位置为1
for i in range(100):
    x_predict = f_mat * x_mat  #由上一状态根据状态转移矩阵预测的当前状态
    p_predict = f_mat * p_mat * f_mat.T + q_mat
    kalman = p_predict * h_mat.T / (h_mat * p_predict * h_mat.T + r_mat)
    x_mat = x_predict + kalman * (z_mat[0, i] - h_mat * x_predict)
    p_mat = (np.eye(2) - kalman * h_mat) * p_predict

    # 将数据置入图片1
    # plt.plot(x_mat[0, 0], x_mat[1, 0], 'ro', markersize=1)
    result[i]=x_mat[0, 0]
    print('x_mat',x_mat[0, 0])


num=len(result)
Y2=np.arange(0,num) #建一个X轴，绘图用

#打印实际采样值

S_Date=noise_mat.A #将矩阵转化为array数组类型,否则无法使用plot生产图形
c_mat=z_mat.A  #将mat矩阵转化为array数组类型

plt.figure(dpi=300)
print('result',result)
print('c_mat',c_mat)
print('z_watch',z_watch)

#绘制合成图
plt.subplot(3,1,1)
plt.plot(Y2,result,'red')
plt.twinx()
# plt.scatter(Y2,c_mat,s=10,alpha=0.8,c='Green')
plt.plot(Y2,c_mat[0,0:100],'Green')

#打印滤波之后结果
plt.subplot(3,1,2)#1行2列的图。设置图片2的位置为2
plt.title('Filte_Result')
plt.scatter(x=Y2,y=result,s=5,alpha=0.8,c='red') #生成散点噪声图，s:尺寸，alpha透明度

#打印含噪声的观测值
plt.subplot(3,1,3) #1行2列的图。设置图片2的位置为3
plt.title('Filte_Before')
plt.scatter(x=Y2,y=c_mat[0,0:100],s=5,alpha=0.8,c='Green') #生成散点观测值图，s:尺寸，alpha透明度


#图形输出
plt.show()