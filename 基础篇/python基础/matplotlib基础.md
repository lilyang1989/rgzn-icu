# Matplotlib简介

matplotlib和Numpy一样，都是python的外部模块，用于绘制图表、显示图像和创造简单动画

引入Matplotlib

```python
import  numpy as np
import matplotlib.pyplot as plt
```



# 本节课程：

* `linspace()`函数
* 绘制图表
* 图表的装饰
* 散点图的显示
* 直方图的显示



## `linspace()`函数

`linspace()`函数可以创造一个Numpy数组，该数组将一个区间等分为50等份，此数组经常用于图表水平轴的值

例：

```python
import  numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,1) #将0-1这个区间等份为50等份
print(x)
print(len(x))
```

结果为：

```python
[0.         0.02040816 0.04081633 0.06122449 0.08163265 0.10204082
 0.12244898 0.14285714 0.16326531 0.18367347 0.20408163 0.2244898
 0.24489796 0.26530612 0.28571429 0.30612245 0.32653061 0.34693878
 0.36734694 0.3877551  0.40816327 0.42857143 0.44897959 0.46938776
 0.48979592 0.51020408 0.53061224 0.55102041 0.57142857 0.59183673
 0.6122449  0.63265306 0.65306122 0.67346939 0.69387755 0.71428571
 0.73469388 0.75510204 0.7755102  0.79591837 0.81632653 0.83673469
 0.85714286 0.87755102 0.89795918 0.91836735 0.93877551 0.95918367
 0.97959184 1.        ]

50
```

## 用pyplot绘制简单图表

用pyplot绘制一条直线，先用上述函数将x的坐标数据作为数组生成，然后用该值乘以2作为y坐标，最后用pyplot绘制出来

```python
import  numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-5,5)
y=x*2
plt.plot(x,y)
plt.show()
```

结果为

![image-20220809005133335](C:\Users\Yang\AppData\Roaming\Typora\typora-user-images\image-20220809005133335.png)

## 图表的装饰

* 轴标签
* 图表标题
* 显示表格
* 图例和线条样式

```python
plt.xlabel("x value",size=14)
#指定轴标签文字大小为14
plt.ylabel("y value",size=14)
#图表标题
plt.title("My graph")
#显示网格
plt.grid()

#指定出图时的图例和线条样式

plt.plot(x,y_1,label="y1")
plt.plot(x,y_2,label="y2",linestyle="dashed")
plt.legend()#展示图例

plt.show()
```

结果为

![image-20220809005834533](C:\Users\Yang\AppData\Roaming\Typora\typora-user-images\image-20220809005834533.png)

## 散点图的显示

可以使用`scatter()`函数来显示散点图

```python
import  numpy as np
import matplotlib.pyplot as plt

x=np.array([1.2,2.4,0.0,1.4,1.5,0.3,0.7])
y=np.array([2.4,1.4,1.0,0.1,1.7,2.0,0.6])

plt.scatter(x,y)#散点图
plt.grid()
plt.show()
```

结果为

![image-20220809010620532](C:\Users\Yang\AppData\Roaming\Typora\typora-user-images\image-20220809010620532.png)

## 直方图

利用`hist()`函数可以绘制直方图，直方图可以统计每个范围的值的出现频率，并用矩形柱进行表示

```python
import  numpy as np
import matplotlib.pyplot as plt

data=np.array([0,1,1,2,2,2,3,3,4,5,6,6,7,7,7,8,8,9])

plt.hist(data,bins=10) #直方图中bins为柱的数量
plt.show()
```

结果为

![image-20220809011449053](C:\Users\Yang\AppData\Roaming\Typora\typora-user-images\image-20220809011449053.png)