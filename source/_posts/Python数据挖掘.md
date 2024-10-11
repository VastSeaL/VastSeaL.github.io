# Matplotlib

##2.1matplotlib之HelloWorld

### 2.1.1什么是matplotlob

专门用于开发2D（3D）图标

使用起来简单

以渐进、交互方式实现数据可视化

数据可视化——帮助理解数据，方便选择更适合的分析方法

js库： D3  	echarts

### matplotlib图像结构

#### matplotlib三层结构

​	1、容器层

​			画板层(Canvas )：位于最底层，用户一般接触不到

​			画布层（FIgure) :plt.figure()	建立在Canvas之上

​			绘图区(AXES)、坐标系(axis)	pil.subplots()

​							x、y轴张成的区域

​					2、辅助显示层

​					3、图像层

​						根据函数绘制图像

## 折线图绘制(plot)与基础绘图功能

### 基本步骤

1、创建画布

2、绘制图像

3、显示图像

```python
#1创建画布
plt.figure(figsize=(20,8),dpi=80)
	### figsize : 画布大小
	### dpi : dot per inch 图像的清晰度,每英寸显示点数
#2绘制图像
plt.plot([1,2,3,4,5],[6,7,8,9,10])

#保存图像(注意保存图片代码应位于plt.show()之前)
plt.savefig("test.png")

#3显示图像
plt.show()
#plt.show()会释放figsure资源
```

### 完善原始折线图（辅助显示层）

```python
# 需求：再添加一个城市的温度变化
# 收集到北京当天温度变化情况，温度在1度到3度。 

# 1、准备数据 x y
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1, 3) for i in x]

# 中文显示问题
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制图像
plt.plot(x, y_shanghai, color="r", linestyle="-.", label="上海")
plt.plot(x, y_beijing, color="b", label="北京")

# 显示图例，这里显示图例的前提是plt.plot时要添加标签lsbel=“”
plt.legend()#legend有自己的参数可以控制图例位置

# 修改x、y刻度
# 准备x的刻度说明  ticks表示刻度
x_label = ["11点{}分".format(i) for i in x]
plt.xticks(x[::5], x_label[::5])
#步长为5，即不让刻度显示过于密集第一处的x[::5]也要写，应该是用来给x_label定位的
plt.yticks(range(0, 40, 5))

# 添加网格显示，其中的alpha是网格的透明程度
plt.grid(linestyle="--", alpha=0.5)

# 添加描述信息
plt.xlabel("时间变化")
plt.ylabel("温度变化")
plt.title("上海、北京11点到12点每分钟的温度变化状况")

# 4、显示图
plt.show()

```

```python
# 需求：再添加一个城市的温度变化
# 收集到北京当天温度变化情况，温度在1度到3度。 

# 1、准备数据 x y
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1, 3) for i in x]

# 中文显示问题
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制图像
plt.plot(x, y_shanghai, color="r", linestyle="-.", label="上海")
plt.plot(x, y_beijing, color="b", label="北京")

# 显示图例，这里显示图例的前提是plt.plot时要添加标签lsbel=“”
plt.legend()#legend有自己的参数可以控制图例位置

# 修改x、y刻度
# 准备x的刻度说明  ticks表示刻度
x_label = ["11点{}分".format(i) for i in x]
plt.xticks(x[::5], x_label[::5])
#步长为5，即不让刻度显示过于密集第一处的x[::5]也要写，应该是用来给x_label定位的
plt.yticks(range(0, 40, 5))

# 添加网格显示，其中的alpha是网格的透明程度
plt.grid(linestyle="--", alpha=0.5)

# 添加描述信息
plt.xlabel("时间变化")
plt.ylabel("温度变化")
plt.title("上海、北京11点到12点每分钟的温度变化状况")

# 4、显示图
plt.show()

```

### 多个坐标系显示-plt.subplots（面向对象的画图方法）

```python
#主要区别：
#figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi=80)

# 需求：再添加一个城市的温度变化
# 收集到北京当天温度变化情况，温度在1度到3度。 

# 1、准备数据 x y
x = range(60)
y_shanghai = [random.uniform(15, 18) for i in x]
y_beijing = [random.uniform(1, 3) for i in x]

# 2、创建画布
# plt.figure(figsize=(20, 8), dpi=80)
figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8), dpi=80)

# 3、绘制图像
axes[0].plot(x, y_shanghai, color="r", linestyle="-.", label="上海")
axes[1].plot(x, y_beijing, color="b", label="北京")

# 显示图例
axes[0].legend()
axes[1].legend()

# 修改x、y刻度
# 准备x的刻度说明
x_label = ["11点{}分".format(i) for i in x]
axes[0].set_xticks(x[::5])
axes[0].set_xticklabels(x_label)
axes[0].set_yticks(range(0, 40, 5))
axes[1].set_xticks(x[::5])
axes[1].set_xticklabels(x_label)
axes[1].set_yticks(range(0, 40, 5))

# 添加网格显示
axes[0].grid(linestyle="--", alpha=0.5)
axes[1].grid(linestyle="--", alpha=0.5)

# 添加描述信息
axes[0].set_xlabel("时间变化")
axes[0].set_ylabel("温度变化")
axes[0].set_title("上海11点到12点每分钟的温度变化状况")
axes[1].set_xlabel("时间变化")
axes[1].set_ylabel("温度变化")
axes[1].set_title("北京11点到12点每分钟的温度变化状况")

# 4、显示图
plt.show()

```

## 常见图形种类及意义

​	折线图plot

​	散点图scatter

​			关系/规律

​	柱状图bar

​			统计/对比

​	直方图histogram

​			分布状况

​	饼图pie

​			占比

### 散点图绘制

```python
# 需求：探究房屋面积和房屋价格的关系

# 1、准备数据
x = [225.98, 247.07, 253.14, 457.85, 241.58, 301.01,  20.67, 288.64,
       163.56, 120.06, 207.83, 342.75, 147.9 ,  53.06, 224.72,  29.51,
        21.61, 483.21, 245.25, 399.25, 343.35]

y = [196.63, 203.88, 210.75, 372.74, 202.41, 247.61,  24.9 , 239.34,
       140.32, 104.15, 176.84, 288.23, 128.79,  49.64, 191.74,  33.1 ,
        30.74, 400.02, 205.35, 330.64, 283.45]
# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制图像
plt.scatter(x, y)

# 4、显示图像
plt.show()

```

## 柱状图绘制

```python
##绘制票房分布直方图

# 1、准备数据
movie_names = ['雷神3：诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴', '降魔传','追捕','七十七天','密战','狂兽','其它']
tickets = [73853,57767,22354,15969,14839,8725,8716,8318,7916,6764,52222]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)




# 3、绘制柱状图
x_ticks = range(len(movie_names))
plt.bar(x_ticks, tickets,width=[0.2 for i in range(x_ticks)],color=['b','r','g','y','c','m','y','k','c','g','b'])
#主要参数x列表，y列表，width列表，color列表


# 修改x刻度
plt.xticks(x_ticks, movie_names)

# 添加标题
plt.title("电影票房收入对比")

# 添加网格显示
plt.grid(linestyle="--", alpha=0.5)

# 4、显示图像
plt.show()
```

```python
# 1、准备数据
movie_name = ['雷神3：诸神黄昏','正义联盟','寻梦环游记']

first_day = [10587.6,10062.5,1275.7]
first_weekend=[36224.9,34479.6,11830]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制柱状图
x_ticks=[i for i in range(3)]
plt.bar(x_ticks, first_day, width=0.2, label="首日票房")
plt.bar([i+0.2 for i in x_ticks], first_weekend, width=0.2, label="首周票房")
##########为何是i+0.2呢，因为前面设置的柱状图宽度width=0.2，为了紧密相连，同时没有重合，因此设置0.2


# 显示图例
plt.legend()

# 修改刻度,即显示坐标轴上的数字或字符，本例即显示电影名字
plt.xticks([i+0.1 for i in x_ticks], movie_name)
###########柱状图宽度为0.2，为了字符名字在中间，因此相对于前面是加了0.1

# 4、显示图像
plt.show()

```

## 直方图

组数：在统计数据时，我们把数据按照不同的范围分成几个组，分成的组的个数称为组数

组距：每一组两个端点的差

```python
# 需求：电影时长分布状况
# 1、准备数据
time = [131,  98, 125, 131, 124, 139, 131, 117, 128, 108, 135, 138, 131, 102, 107, 114, 119, 128, 121, 142, 127, 130, 124, 101, 110, 116, 117, 110, 128, 128, 115,  99, 136, 126, 134,  95, 138, 117, 111,78, 132, 124, 113, 150, 110, 117,  86,  95, 144, 105, 126, 130,126, 130, 126, 116, 123, 106, 112, 138, 123,  86, 101,  99, 136,123, 117, 119, 105, 137, 123, 128, 125, 104, 109, 134, 125, 127,105, 120, 107, 129, 116, 108, 132, 103, 136, 118, 102, 120, 114,105, 115, 132, 145, 119, 121, 112, 139, 125, 138, 109, 132, 134,156, 106, 117, 127, 144, 139, 139, 119, 140,  83, 110, 102,123,107, 143, 115, 136, 118, 139, 123, 112, 118, 125, 109, 119, 133,112, 114, 122, 109, 106, 123, 116, 131, 127, 115, 118, 112, 135,115, 146, 137, 116, 103, 144,  83, 123, 111, 110, 111, 100, 154,136, 100, 118, 119, 133, 134, 106, 129, 126, 110, 111, 109, 141,120, 117, 106, 149, 122, 122, 110, 118, 127, 121, 114, 125, 126,114, 140, 103, 130, 141, 117, 106, 114, 121, 114, 133, 137,  92,121, 112, 146,  97, 137, 105,  98, 117, 112,  81,  97, 139, 113,134, 106, 144, 110, 137, 137, 111, 104, 117, 100, 111, 101, 110,105, 129, 137, 112, 120, 113, 133, 112,  83,  94, 146, 133, 101,131, 116, 111,  84, 137, 115, 122, 106, 144, 109, 123, 116, 111,111, 133, 150]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制直方图
distance = 2#组距
group_num = int((max(time) - min(time)) / distance)#组数=极差/组距

plt.hist(time, bins=group_num, density=True)
##第一个参数是数据，第二个参数是组数，第三个参数是density默认为False
##False显示的是频数，True显示的是频率

# 修改x轴刻度
plt.xticks(range(min(time), max(time) + 2, distance))

# 添加网格
plt.grid(linestyle="--", alpha=0.5)

# 4、显示图像
plt.show()

```

## 饼图

```python
# 1、准备数据
movie_name = ['雷神3：诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴','降魔传','追捕','七十七天','密战','狂兽','其它']

place_count = [60605,54546,45819,28243,13270,9945,7679,6799,6101,4621,20105]

# 2、创建画布
plt.figure(figsize=(20, 8), dpi=80)

# 3、绘制饼图
plt.pie(place_count, labels=movie_name, colors=['b','r','g','y','c','m','y','k','c','g','y'], autopct="%1.2f%%")

# 显示图例
plt.legend()

plt.axis('equal')
##表示横纵轴比相同，即显示为圆形

# 4、显示图像
plt.show()

```

# 3.Numpy

高效的运算工具

  ## 3.1 Numpy介绍 数值计算库

## ndarray优势

​	1）存储风格

​					ndarray -- 相同类型 -- 通用性不强

​					list		--  不同类型 -- 通用性很强

​	2）并行化运算

​					ndarray支持向量化运算

​	3）底层语言

### 3.2.1ndarray属性

|                        |                            |
| ---------------------- | -------------------------- |
| 属性名字               | 属性解释                   |
| ndarray.shape          | 数组维度的元组             |
| ndarray.ndim           | 数组维数                   |
| ndarray.size           | 数组中的元素数量           |
| ndarray.itemsize       | 一个数组元素的长度（字节） |
| ndarray.dtype          | 数组元素的类型             |
| 使用方法 数组名.函数名 |                            |

### 3.2.2 ndarray的形状

 ### 3.2.3 ndarray的类型

```python
a = np.array([[1,2,3],[4,5,6]], dtype = "float32")
a = np.array([[1,2,3],[4,5,6]], dtype = np.float32)
```

## 3.3基本操作

​		adarray.方法()

​		np.函数名()

### 3.3.1生成数组的方法

1. 生成0和1数组

   ```python
   # np.zeros(shape)
   # np.ones(shape)
   zero = np.zeros(shape=(3,4), dtype ='int32')
   ones = np.ones(shape=[3,4], dtype ='int32')		#此时指定属性时，既可以是元组也可以是列表
   ```

2. 从现有数组生成

   ```python
   #np.array()
   #np.copy()
   #np.asarray()
   data1=np.array(score)
   data2=np.copy(score)
   #data1、data2深拷贝
   data3=np.asarray(score)
   #data3浅拷贝
   ```

3. 生成固定范围数组

   ```python
   np.linspace(0,10,100)		#[0,10] 等距离
   
   np.arange(0,10,2)			#[a,b） c是步长
   ```

4. 生成随机数组

   分布状况---直方图

   均匀分布

   - np.random模块	

     1. ```python
        np.random.uniform(low = 0.0, high = 1.0, size = none)
        #功能：从一个均匀分布中[low,high)中随机采样，左闭右开
        #low:采样下界，float类型，默认值为0；
        #high:采样上界，float类型，默认值为1
        #size:输出样本数目，为int或元组类型，例如。size=(m,n,k),则输出mnk个样本，缺省时输出1个值
        #返回值：ndarray类型，其形状和参数size中描述一致
        ```

   正态分布：是一种概率分布。具有两个参数μ和σ的连续型随机变量的分布。μ是服从正太分布的随机变量的均值，σ是此随机变量的标准差，记为N(μ，σ)

```python
np.random.normal(loc=1.75,scale=0.1,size=10000)
```

### 3.3.2数组的索引、切片

```python
atock_change = np.random.normal(loc = 0, scale = 1, size =(8,10))
atock_change
#获取第一支股票的前三个交易日的涨跌幅数据
atock_change[0, 0:3]			#二维数组索引
a1 = np.array([[[1,2,3],[4,5,6]],[[12,3,34],[5,6,7]]])
a1 #(2,2,3)
a1.shape
a1[1,0,2]	#三维数组索引
a1[1,0,2]=100000
#负数对应np中数组从后向前，-1对应最后一个数

```

### 3.3.3形状修改

ndarray.reshape()

ndarray.resize()

ndarray.T()

```python
stock_change.reshape((10, 8)) # 返回新的ndarray, 原始数据没有改变
stock_change.resize((10, 8)) # 没有返回值， 对原始的ndarray进行了修改
stock_change.T # 转置 行变成列，列变成行  返回一个ndarray，原数据未改变


##reshape()是一个函数，因此第一个括号是函数个括号，而第二个括号是因为传入了一个元##组，其实用列表也可

```



### 3.3.4类型改变

ndarray.astype(type)

ndarray序列化到本地

```python
stock_change.astype("int32")
stock_change.tostring() # ndarray序列化到本地？？？？？？
```

### 3.3.5数组的去重

ndarray.unique

```python
temp = np.array([[1, 2, 3, 4],[3, 4, 5, 6]])
np.unique(temp)
set(temp.flatten())##set的操作对象需要时一维的，.flatten()可以压缩为一维的
```

##3.4ndarray运算

### 3.4.1逻辑运算

​	运算符 布尔索引

```python
stock_change = np.random.uniform(low=-1, high=1, size=(5,10))
# 逻辑判断, 如果涨跌幅大于0.5就标记为True 否则为False
stock_change > 0.5     
#返回一个True和False的等大小矩阵

stock_change[stock_change > 0.5] = 1.1  
#将>0.5的全部改为1
```

通用判断函数

np.all(布尔值)

​						只要有一个False就返回False，只有全是True才返回True

np.any(布尔值)

​						只要有一个True就返回True，只有全是False才返回False

```python
#以下两者均只返回一个布尔值

# 判断stock_change[0:2, 0:5]是否全是上涨的
np.all(stock_change[0:2, 0:5] > 0) 
# 判断前5只股票这段期间是否有上涨的
np.any(stock_change[:5, :] > 0)
```

三元运算符

np.where()

```python
# np.where(布尔表达式，True的位置的值，False的位置的值)，类似于三元运算符，不# 过需要利用函数
np.where(temp > 0, 1, 0)
###涉及符合逻辑需要额外的函数logical_and/or
# 大于0.5且小于1
np.where(np.logical_and(temp > 0.5, temp < 1), 1, 0)
# 大于0.5或小于-0.5
np.where(np.logical_or(temp > 0.5, temp < -0.5), 11, 3)
```



### 3.4.2统计运算

统计指标函数

- 主要函数：min max mean median(中位数) var(方差) std(标准差)

- 使用方法：np.函数名(数组名) 或 数组名.方法名

- 同时应当注意 axis的使用。 axis=0表示列 axis=1表示行 axis=-1 表示最后一维度

  ```python
  ###返回值
  stock_change.max()  #将返回最大值
  np.max(stock_change,axis=1)#将返回一个向量，即所有行的最大值
  
  
  ###返回索引
  np.argmax(tem,axis=0)
  np.argmin(tem,axis=0)
  ```

  

返回最大值、最小值所在位置

### 3.4.3数组运算

数组与数的运算

​	正常的运算即可 **加减乘除等**

```python
arr = np.array([[1,2,3,3,2,1],[4,5,6,6,5,4]])
arr+1
arr/2
```

数组与数组的运算

![img](https://img-blog.csdnimg.cn/c87823ff98b74249a11890471cd9462f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5aSp5aSp5YaZ54K55Luj56CB,size_10,color_FFFFFF,t_70,g_se,x_16)



广播机制

执行broadcast的前提在于，两个nadarray执行的是element-wise的运算，Broadcast机制的功能是为了方便不同形状的ndarray(numpy库的核心数据结构)进行数学运算。

当操作两个数组时，numpy会逐个比较它们的shape(构成的元组tuple)，只有在下述情况下，两个数组才能够进行数组与数组的运算。

- 纬度相等
- shape(其中相对应的一个地方为1)

## 3.4.4矩阵运算

matrix,和array的区别是矩阵必须是二维的，但是array可以是多维的

矩阵&二维数组(不满足广播机制不能进行计算)

两种方法存储矩阵

​	1)ndarray 二维数组

​	2)matrix数据结构

```python
#array存储矩阵
data = np.array([[80,86],
                [82,80],
                [85,78],
                [90,90],
                [86,82],
                [82,90],
                [78,80],
                [92,94]])
#matrix存储矩阵
data2=np.mat([[80, 86],
       [82, 80],
       [85, 78],
       [90, 90],
       [86, 82],
       [82, 90],
       [78, 80],
       [92, 94]])
```

- 矩阵乘法运算

  - 形状改变

    （m*n） * （n*m）

  - 运算规则

    矩阵的乘法必须满足运算规则，即(m,n)*(n,l)=(m,l)

    ```python
    #如果是二维数组实现矩阵运算
    np.dot(data,data1)
    np.matmul(data,data1)
    data @ data1
    
    #如果是矩阵进行运算
    data1*data23.5
    ```

## 3.5 合并与分隔

### 3.5.1合并

- numpy.hstack 水平拼接

- numpy.vstack 竖拼接

- numpy.concatenate((a1,a2),axis=0|1) 水平|竖拼接

  ```python
  data1 = np.mat([1,2,3])
  data2 = np.mat([4,5,6])
  np.vstack((data1,data2))
  np.hstack((data1,data2))
  np.concatenate((data1,data2),axis=1)
  ```

### 3.5.2 分隔

## 3.6 IO操作与数据处理

data = np.genfromtxt("text.csv", delimeter = ",")

### 3.6.1numpy读取

### 3.6.2 如何处理缺失值

- 直接删除含有缺失值的样本
- 替补/插补；

# 4 Pandas

## 4.1Pandas介绍

- Pandas=panel+data+analysis
  专门用于数据挖掘的开源Python库
  以Numpy为基础，借力Numpy模块在计算方面性能高的优势
  基于matplotlib，能够简便的画图
  独特的数据结构
  便捷的数据处理能力
  读取文件方便
  封装了Matplotlib、Numpy的画图和计算


### 4.1.3 DataFrame

​	结构：既有行索引又有列索引的二维数组

- 索引：行索引-index，横向索引；列索引-columns，纵向索引

  ```python
  pd.DataFrame(stock_change)
  #添加行索引
  stock = ["股票{}".format(i) for i in range(10)]
  pd.DataFrame(stock_change, index = stock)
  #添加列索引
  date = pd.date_range(start="20100101",periods = 5, freq= "B")
  pd.DataFrame(stock_change, index = stock, columns = date)
  ```

- 值：values，利用values即可直接获得去除索引的数据（数组）

- shape：表明形状 (形状不含索引的行列)

- T：行列转置

```python
#获取局部展示
b.head()#默认展示前5行，可在head()加入数字，展示前几行
b.tail()#默认展示后5行，可在tail()加入数字，展示后几行

#获取索引和值
b.index#获取行索引
#####返回一个类似列表的东西，也可以利用数字继续索引例:a.index[1]
b.columns#获取列索引
b.values#获取数据值（数组，不含索引）
b.shape#获取DataFrame的维度数据
b.T#获取转制后的dataframe

#设置行列索引
# 创建一个符合正态分布的10个股票5天的涨跌幅数据
stock_change = np.random.normal(0, 1, (10, 5))
pd.DataFrame(stock_change)
#设置行列索引
stock = ["股票{}".format(i) for i in range(10)]
date = pd.date_range(start="20200101", periods=5, freq="B")#这个是pandas中设置日期的
# 添加行列索引
data = pd.DataFrame(stock_change, index=stock, columns=date)
```

 DataeFrame索引的设置

- 修改行列索引值

  ​	不能单独修改所以，只能整体修改索引

```python
#不能单独修改行列总某一个索引的值，可以替换整行或整列   例：b.index[2]='股票1'  错误
data.index=新行索引
#重设索引
data.reset_index(drop=False)
#drop参数默认为False，表示将原来的索引替换掉，换新索引为数字递增，原来的索引将变为数据的一部分。True表示，将原来的索引删除，更换为数字递增。如下图
```

- 重设索引
- 设置新索引

```python
# 设置新索引
df = pd.DataFrame({'month': [1, 4, 7, 10],
                    'year': [2012, 2014, 2013, 2014],
                    'sale':[55, 40, 84, 31]})
# 以月份设置新的索引
df.set_index("month", drop=True)
#见下图，即将原本数据中的一列拿出来作为index
new_df = df.set_index(["year", "month"])# 设置多个索引，以年和月份   多个索引其实就是MultiIndex
```

### 4.1.4 MultiIndex

分级或分层索引对象

- Index属性
  - names: levels的名称
  - levels:每个level的元组值

## 4.1.5 Panel

pandas.Panel(data=None,items=None,major_axis=None,minor_axis=None,copy=False,dtype=None)

存储3维数组的Panel结构

- items - axis 0，每个项目对应于内部包含的数据帧(DataFrame)。
- major_axis - axis 1，它是每个数据帧(DataFrame)的索引(行)。
- minor_axis - axis 2，它是每个数据帧(DataFrame)的列。

## 4.1.6 Series

​		带索引的一维数组

```python
# 创建
pd.Series(np.arange(3, 9, 2), index=["a", "b", "c"])
# 或
pd.Series({'red':100, 'blue':200, 'green': 500, 'yellow':1000})

sr = data.iloc[1, :]
sr.index # 索引
sr.values # 值

#####就是从dataframe中抽出一行或一列来观察
```

## 4.2基本数据操作

### 4.2.1 索引操作

```python
data=pd.read_csv("./stock_day/stock_day.csv")#读入文件的前5行表示如下
######利用drop删除某些行列，需要利用axis告知函数是行索引还是列索引
data=data.drop(["ma5","ma10","ma20","v_ma5","v_ma10","v_ma20"], axis=1) # 去掉一些不要的列
data["open"]["2018-02-26"] # 直接索引，但需要遵循先列后行

#####按名字索引利用.loc函数可以不遵循列行先后关系
data.loc["2018-02-26"]["open"] # 按名字索引
data.loc["2018-02-26", "open"]


#####利用.iloc函数可以只利用数字进行索引
data.iloc[1][0] # 数字索引
data.iloc[1,0]


# 组合索引
# 获取行第1天到第4天，['open', 'close', 'high', 'low']这个四个指标的结果
data.ix[:4, ['open', 'close', 'high', 'low']] # 现在不推荐用了
###但仍可利用loc和iloc
data.loc[data.index[0:4], ['open', 'close', 'high', 'low']]
data.iloc[0:4, data.columns.get_indexer(['open', 'close', 'high', 'low'])]
```

	#### 4.2.1.1 直接索引

#### 4.2.1.2 按名字索引

#### 4.2.1.3 按数组索引

### 4.2.2 赋值操作

```python
data.open=100
data['open']=100
###两种方式均可
data.iloc[1,0]=100
###找好索引即可
```

### 4.2.3排序

sort_values （比较values进行排序） sort_index （比较行索引进行排序，不行可以先转置简介对列排序）

内容排序

索引排序

```python
data.sort_values(by="high", ascending=False) # DataFrame内容排序，ascending表示升序还是降序，默认True升序

data.sort_values(by=["high", "p_change"], ascending=False).head() # 多个列内容排序。给出的优先级进行排序

data.sort_index(ascending=True)###对行索引进行排序

#这里是取出了一列 “price_change”列，为serise，用法同上
sr = data["price_change"]
sr.sort_values(ascending=False)
sr.sort_index()
```

## 4.3 DataFrame运算

### 4.3.1算术运算

 

```python
#正常的加减乘除等的运算即可
data["open"] + 3
data["open"].add(3) # open统一加3  
data.sub(100)# 所有统一减100 data - 100
(data["close"]-(data["open"])).head() # close减open
```

### 4.3.2 逻辑运算

逻辑运算 ：< ; > ; | ; & 利用逻辑符号或者函数query

```python
# 例如筛选p_change > 2的日期数据
data[data["p_change"] > 2].head()
# 完成一个多个逻辑判断， 筛选p_change > 2并且low > 15
data[(data["p_change"] > 2) & (data["low"] > 15)].head()
data.query("p_change > 2 & low > 15").head()###等效于上一行代码

###判断# 判断'turnover'列索引中是否有4.19, 2.39，将返回一列布尔值
data["turnover"].isin([4.19, 2.39])##如下图
```

利用布尔值索引，即利用一个布尔数组索引出True的数据

```python
###判断# 判断'turnover'列索引中是否有4.19, 2.39，将返回一列布尔值
data["turnover"].isin([4.19, 2.39])##如下图
data[data["turnover"].isin([4.19, 2.39])]
#这块就将返回turnover列布尔值为true的如下图，也就是筛选出turnover中值为4.19和2.39


###布尔值索引是一个很方便的数据筛选操作，比如：
data[data["turnover"]>0.1]
#也将筛选出turnover列中大于0.1的整体data数据，并不是说只返回turnover相关数据，判断只是返回布尔索引，利用索引的是data数据
```



### 4.3.3 统计运算

```python
data.describe()
#将返回关于列的最值，均值，方差等多种信息
##其实这里很多就和numpy相似了
data.max(axis=0)#返回最值
data.idxmax(axis=0) #返回最值索引
```

**累计统计函数（累加，累乘等）**

- cumsum 计算前1/2/3/…/n个数的和
- cummax 计算前1/2/3/…/n个数的最大值
- cummin 计算前1/2/3/…/n个数的最小值
- cumprod 计算前1/2/3/…/n个数的积

**自定义运算**

 apply(func, axis=0)

 func: 自定义函数

 axis=0: 默认按列运算，axis=1按行运算

```python
data.apply(lambda x: x.max() - x.min())
#这里的lambda x: x.max() - x.min()是lambda表达式，是函数的简单写法也可
def fx(data):
	return	data.max()-data.min()
```

## 4.4Pandas画图

pandas.DataFrame.plot
DataFrame.plot(x=None, y=None, kind=‘line’)

- x: label or position, default None
- y: label, position or list of label, positions, default None
  - Allows plotting of one column versus another
  - kind: str
  - ‘line’: line plot(default)
  - 'bar": vertical bar plot
  - “barh”: horizontal bar plot
  - “hist”: histogram
  - “pie”: pie plot
  - “scatter”: scatter plot

## 4.5 文件读取与存储

### 4.5.1 CSV文件

#### 4.5.1.1 读取CSV文件 read_csv()

```python
#pandas.read_csv(filepath_or_buffer='',usecols=，names=)
#filepath_or_buffer : 文件路径
#usecols : 指定读取的列名、列表形式
#names : 参数
data = pd.read_csv("D:\\python\\study\\day3资料\\02-代码\\stock_day\\stock_day.csv")
```

#### 4.5.1.2 写入CSV文件 to_csv()

```python
DataFrame.to_csv(path_or_buf=None,sep=','columns=None,header=True,index=True,index_label=None,mode='w',encoding=None)
```

- path_or_buf ：string or file handle ， default None

- sep : character, default ‘,’（分隔符）

- columns ：sequence,optional

- mode：'w‘:重写，'a’追加

- index:是否写入 行索引

- header:boolean or list of string,default True,是否写进列索引值

  ```python
  Series.to_csv (path=None,index=True,sep=',',na_rep='',float_format=None,header=False,index_label=None,mode='w',encoding=None,compression=None,date_format=None,decimal='.)
  ```

  

```python
pd.read_csv("./stock_day/stock_day.csv", usecols=["high", "low", "open", "close"]).head() # 读哪些列

data = pd.read_csv("stock_day2.csv", names=["open", "high", "close", "low", "volume", "price_change", "p_change", "ma5", "ma10", "ma20", "v_ma5", "v_ma10", "v_ma20", "turnover"]) # 如果列没有列名，用names传入

data[:10].to_csv("test.csv", columns=["open"]) # 保存open列数据

data[:10].to_csv("test.csv", columns=["open"], index=False, mode="a", header=False) # 保存opend列数据，index=False不要行索引，mode="a"追加模式|mode="w"重写，header=False不要列索引
```

## 4.6Pandas高级处理

### 4.6.1 缺失值处理

如何进行缺失值处理？

- 删除含有缺失值的样本
- 替换/插补数据

**判断NaN是否存在**

- pd.isnull(df) 会返回整个dataframe的布尔框架，难以观察(bool为True代表那个位置是缺失值)
- pd.isnull(df).any() 表示只要有一个True就返回True
- pd.notnull(df)会返回整个dataframe的布尔框架，难以观察(bool为False代表那个位置是缺失值)
- pd.notnull(df).all() 表示只要有一个False就返回False

**删除nan数据**

- df.dropna(inplace=True) 默认按行删除 inplace:True修改原数据，False返回新数据，默认False

**替换nan数据**

- df.fillna(value,inplace=True)
- value替换的值
- inplace:True修改原数据，False返回新数据，默认False

```python
movie["Revenue (Millions)"].fillna(movie["Revenue (Millions)"].mean(), inplace=True)
###这就是先利用其他代码判断出"Revenue (Millions)"有nan数据，然后利用.fillna函数，令value=movie["Revenue (Millions)"].mean()列的均值，然后inplace=True修改原数据

import pandas as pd
import numpy as np
movie = pd.read_csv("./IMDB/IMDB-Movie-Data.csv")
# 1）判断是否存在NaN类型的缺失值
np.any(pd.isnull(movie)) # 返回True，说明数据中存在缺失值
np.all(pd.notnull(movie)) # 返回False，说明数据中存在缺失值
pd.isnull(movie).any()
pd.notnull(movie).all()

# 2）缺失值处理
# 方法1：删除含有缺失值的样本
data1 = movie.dropna()
pd.notnull(data1).all()

# 方法2：替换
# 含有缺失值的字段
# Revenue (Millions)    
# Metascore
movie["Revenue (Millions)"].fillna(movie["Revenue (Millions)"].mean(), inplace=True)
movie["Metascore"].fillna(movie["Metascore"].mean(), inplace=True)
```

**替换非nan的标记数据**

```python
# 读取数据
path = "wisconsin.data"
name = ["Sample code number",  "Normal Nucleoli","Mitoses", "Class"]
data = pd.read_csv(path, names=name)

#这里的非nan标记值缺失值就是利用“?”表示的，因此利用参数to_replace,value=np.nan,将默认标记值替换为nan值，然后再利用签署方法处理nan缺失值
# 1）替换
data_new = data.replace(to_replace="?", value=np.nan)
```

## 4.7 数据离散化

![image-20221031151945710](D:\人工智能\深度学习\graph\image-20221031151642094.png)

- 实现方法：

  1.分组

  - 自动分组 sr = pd.qcut(data,bins)
  - 自定义分组 sr = pd.cut(data,[])

  2.将分组好的结果转换成one-hot编码（哑变量）

  - pd.get_dummies(sr, prefix=)

```python
# 1）准备数据
data = pd.Series([165,174,160,180,159,163,192,184], index=['No1:165', 'No2:174','No3:160', 'No4:180', 'No5:159', 'No6:163', 'No7:192', 'No8:184']) 
# 2）分组
# 自动分组
sr = pd.qcut(data, 3)
sr.value_counts()  # 看每一组有几个数据
# 3）转换成one-hot编码
pd.get_dummies(sr, prefix="height")

# 自定义分组
bins = [150, 165, 180, 195]#这就表示有三组[150,165][165,180][180,195]
sr = pd.cut(data, bins)
# get_dummies
pd.get_dummies(sr, prefix="身高")
```

## 4.8 合并

指合并不同dataframe上的内容数据

- 按方向拼接

  ```python
  pd.concat([data1, data2], axis=1) 
  #axis：0为列索引；1为行索引
  ```

- 按索引拼接

  ```python
  left = pd.DataFrame({'key1': ['K0', 'K0', 'K1', 'K2'],
                          'key2': ['K0', 'K1', 'K0', 'K1'],
                          'A': ['A0', 'A1', 'A2', 'A3'],
                          'B': ['B0', 'B1', 'B2', 'B3']})
  right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                          'key2': ['K0', 'K0', 'K0', 'K0'],
                          'C': ['C0', 'C1', 'C2', 'C3'],
                          'D': ['D0', 'D1', 'D2', 'D3']})
  pd.merge(left, right, how="inner", on=["key1", "key2"])
  pd.merge(left, right, how="left", on=["key1", "key2"])
  pd.merge(left, right, how="outer", on=["key1", "key2"])
  ###这里merge参数解释：
  #left: 需要合并的一个表，合并后在左侧
  #right:需要合并的一个表，合并后在右侧
  #how: 合并方式
  #on: 在哪些索引上进行合并
  ```

  # 4.9 交叉表与透视表

  找到、探索两个变量之间的关系0