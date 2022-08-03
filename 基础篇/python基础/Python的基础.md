# 2.1Python的基础

## 2.1.1变量与赋值

python声明变量且赋值的方式非常简单，只需变量=值即可。

例：

```py
abcd=1234
```

变量名称可以包含数字或下划线，python可以将整数、小数和字符串代入变量。

```python
a=123;
b_123=123.456
hello_world="Hello World!" #给变量hello_world赋值代入字符串“Hello World！”
```

>注意：Python和C++一样，对变量同样是大小写敏感的，例如abcd和ABCD会被识别为不同的变量

## 2.1.2值的显示

 显示值的示例

```python
a=123
print(a)
```

集中显示值的示例

```pyhton
print(123,123.456,"Hello World")
```

## 2.1.3运算符

下面的示例中使用了加减乘幂运算符。

```python
a=3
b=4
c=a+b
print("加法:",c)

d=a-b
print("减法:",d)

e=a*b
print("乘法:",e)


f=a**b  # 幂（a的b次方）
print("幂:",f)
```

除法方面，有小数除法和整数除法。"/"运算符将结果转换为小数，”//“运算符将结果转换为整数。还可以使用%运算符计算除以整数后的余数。

```python
g=a/b  #结果转换为小数
print("除法（小数）:",g)

h=a//b #结果为整数
print("除法（整数）:",h)

i=a%b #余数
print("余数:"，i)
```



## 2.1.4 大数字小数字的表示方法

>较大数字和较小数字位数较多时可以用e来表示。

例：

```python
a=1.2e5  #120000
print(a)

b=1.2e-4  #0.00012
print(b)
```



## 2.1.6 列表

列表允许将多个值合并为一个变量。列表用[]包围住整个值，并用“，”对他们进行分隔。

```python
a= [1,2,3,4]
print(a)
```

输出为：

```python
[1,2,3,4]
```

同样，类比数组，你可以在列表名称后面加上索引来检索列表中的元素。索引从元素的开头起以0、1、2、3、····的顺序依次排序

```python
b=[4,5,6,7]
print(b[2])  #以0开头并且按照0、1、2、3、···的顺序添加索引，并且检索索引为2的元素
```

输出为

```python
6
```

### 为列表添加元素

>通过调用append（）方法将新的元素添加到列表中

```python
c=[1,2,3,4,5]
c.append(6) #将6添加进列表
print(c)
```

结果：

```python
[1,2,3,4,5,6]
```

还可以通过将列表房主另一个列表中来创建双重列表

```python
d=[[1,2,3],[4,5,6]]
print(d)
```

结果：

```python
[[1,2,3],[4,5,6]]
```

### 创建新的列表

例：创建一个新的列表，其中所有元素均以复数条目排列

```python
e=[1,2]
print(e*3) #新列表中将原列表中的元素进行3次重复排列
```

结果为：

```python
[1,2,1,2,1,2]
```

## 2.1.7元组

>与列表一样，元组用于处理多个值，但不能增删改。

元组用（）包围整个值（元素），如果不需要进行更改元素的操作，使用元组比使用列表来的更加方便。

```python
a=(1,2,3,4,5)
b=a[2]
print(b)
```

输出:

```python
3
```



* 只有一个元素的元组：

```python
c=(3,)
print(c)
```

输出为：

```python
(3,)
```

* 将列表或元组的元素统一代入变量

```python
d=[1,2,3]
d_1,d_2,d_3=d
print(d_1,d_2,d_3)

e=(4,5,6)
e_1,e_2,e_3=e
print(e_1,e_2,e_3)
```

输出为

```python
 1 2 3 
 4 5 6  
```

## 2.1.8 流程控制类工具

### if语句

伪代 码如下：

```python
if 条件表达式
	语句块1
else:
    语句块2
```

注意：语句块由行首处的缩进表示，在python中缩进通常由四个半角空格表示，通常按下一次Tab键就能达到该效果

例：

```python
a=5

if a>3:
    print(a+2)
else:
    print(a-2)
```

结果

```python
7
```

### for 语句

​	for语句和列表一起使用时，语法通常如下所示

```python
for 变量 in 列表：
    语法块
```

例：

```python
for a in [4,7,10]:
	print(a+1)
```

```python
5
8
11
```

### range()

​	当for语句和range()一起使用时

```python
for 变量 in range(整数)
 		语法块
```

例：

```python
for a in range(5)
	print(a)
```

输出为：

```python
0
1
2
3
4
```

### 循环中的break、continue语句

break和continue语句和借鉴自C语言，来看以下程序

```python
for letter in 'Python':    
   if letter == 'h':
      break
   print '当前字母 :', letter
```

输出为：

```python
当前字母 : P
当前字母 : y
当前字母 : t
```



continue语句跳出本次循环，而break跳出整个循环。

continue 语句用来告诉Python跳过当前循环的剩余语句，然后继续进行下一轮循环。

```python
var = 10                   
while var > 0:              
   var = var -1
   if var == 5:
      continue
   print '当前变量值 :', var
print "Good bye!"
```

结果

```python
当前变量值 : 9
当前变量值 : 8
当前变量值 : 7
当前变量值 : 6
当前变量值 : 4
当前变量值 : 3
当前变量值 : 2
当前变量值 : 1
当前变量值 : 0
Good bye!
```

### pass语句

pass为空语句，为了保证程序结构的完整性

pass不做**任何事情**，一般用做占位语句。

```python
# 输出 Python 的每个字母
for letter in 'Python':
   if letter == 'h':
      pass
      print '这是 pass 块'
   print '当前字母 :', letter
 
print "Good bye!"
```

## 2.1.9 函数

格式如下所示:

```python
def my_func_1():
    a=2
    b=3
    print(a+b)
my_func_1()  # 调用函数
```

带有参数的函数

```python
def my_func_2(p,q): #p,q为参数
	c=p+q
    return c
  
my_func_2(6,4) #调用函数的同时传入参数
```

如果要返回多个值，需要用到元组写在return后面

```python
def my_func_3(p,q):
    r=p+q
    s=p-q
    return(r,s)#将返回值设置为元组
k,l=my_func_3(5,2)#将元组的值分别代入k,l
```

## 2.1.10 作用域

### 局部变量和全局变量

```python
a=123 #全局变量
def show_number():
    b=456 #局部变量
    print(a,b)
    
show_number()
```

结果

```python
123 456
```

变量的作用域取决于书写变量的位置，但全局变量的规则要复杂些

如果尝试为函数中的全局变量赋值，python会将其视为一个局部变量

```python
a=123 #全局变量

def set_local():
    a=456 #a不同于上述变量，是另一个局部变量
    print("Local",a)
    
set_local()
print("Global:",a) #全局变量的值不变
```

结果

```python
Local:456
Global:123
```

### 作为函数变量的作用域

同样的规则也适用于作为函数参数的变量，在下面的例子中，参数的变量名a与全局变量的名称相同，但在函数内部a是另一个局部变量

```python
a=123

def show_arg(a) #a和上面的变量不同 两者不是一个变量
	print("Local:",a)
    
 show_arg(456) #调用函数打印456
print(a) #打印全局变量a
```

结果

```python
123
456
```

