import numpy as np

c=np.array([[0,1,2],[3,4,5]])
print(c[1,:]) #检查索引为1的行
print()
c[:,1]=np.array([6,7]) #替换索引为1的行
print(c)