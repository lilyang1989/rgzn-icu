import numpy as np
import  matplotlib.pyplot as plt

def my_func(x):  #计算最小值的函数
    return x**2-2*x

def grad_func(x):  #函数
    return 2*x-2

eta=0.1   #学习系数
x=4.0     #为x设定初始值
record_x=[] #x的记录
record_y=[] #y的记录
for i in range(20): #将x更新20次
    y=my_func(x)
    record_x.append(x)
    record_y.append(y)
    x-=eta*grad_func(x)

x_f=np.linspace(-2,4) #显示范围
y_f=my_func(x_f)

plt.plot(x_f,y_f,linestyle="dashed") #用虚线表示函数
plt.scatter(record_x,record_y)  #显示x与y的记录

plt.xlabel("x",size=14)
plt.ylabel("y",size=14)
plt.grid()

plt.show()
