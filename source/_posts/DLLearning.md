---
title: DLLearning
date: 2023-12-29 15:33:32
tags:
---







# Python

# Python内置函数

### enumerate()函数

enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。

```python
enumerate(sequence, [start=0])
```

### 参数

- sequence -- 一个序列、迭代器或其他支持迭代对象。
- start -- 下标起始位置的值。

### 返回值

返回 enumerate(枚举) 对象。

# Python面向对象

## **类(class)**:

用来描述具有相同属性和方法的集合。定义了该集合中每个对象所共有的属性和方法。**对象是类的实例。**

###类对象

类对象支持两种操作：属性引用和实例化。

属性引用使用和 Python 中所有的属性引用一样的标准语法：**obj.name**。

类对象创建后，类命名空间中所有的命名都是有效属性名。所以如果类定义是这样:

```python
class MyClass:
    """一个简单的类实例"""
    i = 12345
    def f(self):
        return 'hello world'
 
# 实例化类
x = MyClass()
 
# 访问类的属性和方法
print("MyClass 类的属性 i 为：", x.i)
print("MyClass 类的方法 f 输出为：", x.f())
```

##类属性与方法

**类的私有属性：****__private_attrs**：两个下划线开头，声明该属性为私有，不能在类的外部被使用或直接访问。在类内部的方法中使用时 **self.__private_attrs**。

```python
class JustCounter:
    __secretCount = 0  # 私有变量
    publicCount = 0    # 公开变量
 
    def count(self):
        self.__secretCount += 1
        self.publicCount += 1
        print (self.__secretCount)
 
counter = JustCounter()
counter.count()
counter.count()
print (counter.publicCount)
print (counter.__secretCount)  # 报错，实例不能访问私有变量
```

###**类的方法：**

在类的内部，使用 def 关键字来定义一个方法，与一般函数定义不同，类方法必须包含参数 **self**，且为第一个参数，**self** 代表的是类的实例。

**self** 的名字并不是规定死的，也可以使用 **this**，但是最好还是按照约定使用 **self**。

### 类的私有方法

**__private_method**：两个下划线开头，声明该方法为私有方法，只能在类的内部调用 ，不能在类的外部调用。**self.__private_methods**。

```python
class Site:
    def __init__(self, name, url):
        self.name = name       # public
        self.__url = url   # private
 
    def who(self):
        print('name  : ', self.name)
        print('url : ', self.__url)
 
    def __foo(self):          # 私有方法
        print('这是私有方法')
 
    def foo(self):            # 公共方法
        print('这是公共方法')
        self.__foo()
 
x = Site('菜鸟教程', 'www.runoob.com')
x.who()        # 正常输出
x.foo()        # 正常输出
x.__foo()      # 报错
```

### 类的专有方法

![类的专有方法](类专有方法.png)

## **方法**：

类中定义的函数。

- 类有一个名为  __ init __() 的特殊方法（**构造方法**），该方法在类实例化时会自动调用,

  - _ _ init_ _() 方法可以有参数，参数通过 _ _init__ _() 传递到类的实例化操作上

    ```python
    class Complex:
        def __init__(self, realpart, imagpart):
            self.r = realpart
            self.i = imagpart
    x = Complex(3.0, -4.5)
    print(x.r, x.i)   # 输出结果：3.0-4.5
    ```

  - ### self代表类的实例，而非类

    - 类的方法与普通的函数只有一个特别的区别——它们必须有一个额外的**第一个参数名称**, 按照惯例它的名称是 self.
    - self 代表的是类的实例，代表当前对象的地址，而 self.class 则指向类。
    - self 不是 python 关键字，我们把他换成 runoob 也是可以正常执行的

    ```python
    class Test:
        def prt(self):
            print(self)
            print(self.__class__)
     
    t = Test()
    t.prt()
    ```

- 类的方法：在类的内部，使用 **def** 关键字来定义一个方法，与一般函数定义不同，类方法必须包含参数 self, 且为第一个参数，self 代表的是类的实例。

  ```1
  #类定义
  class people:
      #定义基本属性
      name = ''
      age = 0
      #定义私有属性,私有属性在类外部无法直接进行访问
      __weight = 0
      #定义构造方法
      def __init__(self,n,a,w):
          self.name = n
          self.age = a
          self.__weight = w
      def speak(self):
          print("%s 说: 我 %d 岁。" %(self.name,self.age))
   
  # 实例化类
  p = people('runoob',10,30)
  p.speak()
  ```

  

##**类变量：**

类变量在整个实例化的对象中是公用的。类变量定义在类中且在函数体之外。类变量通常不作为实例变量使用。

##**数据成员：**

类变量或者实例变量用于处理类及其实例对象的相关的数据。

##**方法重写：**

如果从父类继承的方法不能满足子类的需求，可以对其进行改写，这个过程叫方法的覆盖（override），也称为方法的重写。

- ```python
  class Parent:        # 定义父类
     def myMethod(self):
        print ('调用父类方法')
   
  class Child(Parent): # 定义子类
     def myMethod(self):
        print ('调用子类方法')
   
  c = Child()          # 子类实例
  c.myMethod()         # 子类调用重写方法
  super(Child,c).myMethod() #用子类对象调用父类已被覆盖的方法
  ```

  [super() 函数](https://www.runoob.com/python/python-func-super.html)是用于调用父类(超类)的一个方法。

##**局部变量：**

定义在方法中的变量，只作用于当前实例的类。

##**实例变量：**

在类的声明中，属性是用变量来表示的，这种变量就称为实例变量，实例变量就是一个用 self 修饰的变量。

##**继承：**

即一个派生类（derived class）继承基类（base class）的字段和方法。继承也允许把一个派生类的对象作为一个基类对象对待。例如，有这样一个设计：一个Dog类型的对象派生自Animal类，这是模拟"是一个（is-a）"关系（例图，Dog是一个Animal）。

- 派生类

  ```python
  class DerivedClassName(BaseClassName):
      <statement-1>
      .
      .
      .
      <statement-N>
  ```

  子类（派生类 DerivedClassName）会继承父类（基类 BaseClassName）的属性和方法。

  BaseClassName（实例中的基类名）必须与派生类定义在一个作用域内。除了类，还可以用表达式，基类定义在另一个模块中时这一点非常有用:

  ```python
  class DerivedClassName(modname.BaseClassName):
  ```

  ```python
  #类定义
  class people:
      #定义基本属性
      name = ''
      age = 0
      #定义私有属性,私有属性在类外部无法直接进行访问
      __weight = 0
      #定义构造方法
      def __init__(self,n,a,w):
          self.name = n
          self.age = a
          self.__weight = w
      def speak(self):
          print("%s 说: 我 %d 岁。" %(self.name,self.age))
   
  #单继承示例
  class student(people):
      grade = ''
      def __init__(self,n,a,w,g):
          #调用父类的构函
          people.__init__(self,n,a,w)
          self.grade = g
      #覆写父类的方法
      def speak(self):
          print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
   
   
   
  s = student('ken',10,60,3)
  s.speak()
  ```

- 多继承

  ```python
  class DerivedClassName(Base1, Base2, Base3):
      <statement-1>
      .
      .
      .
      <statement-N>
  ```

  需要注意圆括号中父类的顺序，若是父类中有相同的方法名，而在子类使用时未指定，python从左至右搜索 即方法在子类中未找到时，从左到右查找父类中是否包含方法。

  ```python
  #类定义
  class people:
      #定义基本属性
      name = ''
      age = 0
      #定义私有属性,私有属性在类外部无法直接进行访问
      __weight = 0
      #定义构造方法
      def __init__(self,n,a,w):
          self.name = n
          self.age = a
          self.__weight = w
      def speak(self):
          print("%s 说: 我 %d 岁。" %(self.name,self.age))
   
  #单继承示例
  class student(people):
      grade = ''
      def __init__(self,n,a,w,g):
          #调用父类的构函
          people.__init__(self,n,a,w)
          self.grade = g
      #覆写父类的方法
      def speak(self):
          print("%s 说: 我 %d 岁了，我在读 %d 年级"%(self.name,self.age,self.grade))
   
  #另一个类，多重继承之前的准备
  class speaker():
      topic = ''
      name = ''
      def __init__(self,n,t):
          self.name = n
          self.topic = t
      def speak(self):
          print("我叫 %s，我是一个演说家，我演讲的主题是 %s"%(self.name,self.topic))
   
  #多重继承
  class sample(speaker,student):
      a =''
      def __init__(self,n,a,w,g,t):
          student.__init__(self,n,a,w,g)
          speaker.__init__(self,n,t)
   
  test = sample("Tim",25,80,4,"Python")
  test.speak()   #方法名同，默认调用的是在括号中参数位置排前父类的方法
  super(student, test).speak()
  ```

## **实例化：**

创建一个类的实例，类的具体对象。

## **对象：**

通过类定义的数据结构实例。对象包括两个数据成员（类变量和实例变量）和方法。

#线性模型

##MSE（平均平方误差 Mean Square Error）

![image-20221116141550394](D:\人工智能\photo\pytorch md\d2l-en-pytorch.pdf)

穷举法

#梯度下降算法实践

分治：若是凸函数可用，不是话陷入局部最优

 梯度(Gradient): 

梯度下降法也会陷入到局部最优，后来在神经网络中发现用梯度下降算法很难陷入局部最优点

非凸函数： 局部最优

![局部优化](局部优化.png)

鞍点：梯度为0

![鞍点](鞍点1.png)

![梯度下降](梯度下降.png)

指数加权均值：	C~i~是当前损失，C^`^~i~是更新后损失

![指数加权均值](指数加权均值.png)

训练发散：训练集正确训练后都是收敛的，对于训练发散常见原因是学习率取得太大 

![SGD](SGD.png)

## 随机梯度下降(SGD)

![SGD](SGD.png)

## Batch

在梯度下降算法w计算是可以并行的

![Batch](Batch.png)

## Back Propagation 反向传播

![反向传播](反向传播.png)

![image-20221116151255009](D:\人工智能\photo\pytorch md\fanxiangchuanbo.pnj)

## Chain Rule 链式法则

 前馈

Backward

![chain rule](chain rule.png)

![求梯度](求梯度.png)

## Pytorch中前馈和反馈计算

tensor:Pytorch中存储数据数据

​			data		grad

![tensor]tensor.png)

# 用Pytorch实现线性回归

![pytorch实现](pytorch实现.png)

## 广播机制

![广播机制](广播机制.png)

## affine model 仿射模型 

线性单元

![线性单元](线性单元.png)

列数为维度，loss为标量

定义模型时必须继承自nn.Module类	构造函数：__ init __() 初始化构造对象使用的函数 和 forward()函数  前馈过程中必须使用的函数 必须定义   backward无是因为Module对象会自动求导

![definite module](definite module.png)

torch.nn.Linear(,)构造对象

nn: Neural Network

![nn_linear](nn_linear.png)

## 训练过程

![image-20221117144717501](D:\人工智能\photo\pytorch md\训练过程.pnj)

## 代码

```python
import torch
#1、数据准备
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])
#2、模型 design model using class
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel,self).__init__()
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias
        self.linear = torch.nn.Linear(1,1)
    
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred
    
model = LinearModel()
#3、构建损失函数和优化器
criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
#4、训练 training cycle forward, backward, update
for epoch in range(100000):
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())
x_test = torch.tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```

 # 逻辑斯蒂回归   分类(classification)

分类问题中输出的是概率

二分类：只有两个类别的分类问题

## torchvision

torchvision中有很多数据集

参数train表示想要下载的是训练集还是测试集

![minist](minist.png)

##Logistic Function

![logistic](logistic.png)

sigmoid functions

![sigmoid functions](sigmoid functions.png)

## Logistic Regression Model



![logistic Regression Model](logistic Regression Model.png)

## Loss function for Binary Classification

此时，我们输出的不在是一个数值而是一个分布

BCE

![Loss function classcification](Loss function classcification.png)

两个分布间的差异

交叉熵

![BCE](BCE.png)

# 处理多维特征的输入

行——样本(sample)

列——特征(Feature)

并行计算

![mini batch](mini batch.png)

## 构造多层神经网络

![Linear Layer](Linear Layer.png)

# 加载数据集

Dataset——数据集 索引

DataLoader——Mini Batch

## Epoch、Batch-Size、Iterations

Epoch:所有的训练样本进行了一次前向传播和反向传播是1次Epoch

Batch Size : 每次训练所用的样本数量

Iteration:迭代了多少次 

shuffle=True 打乱数据集

![shuffle](shuffle.png)

# 多分类问题

实现输出分类的要求 大于0 和为1

![多分类](分类.png)

![softmaxlayer](softmaxlayer.png)

![softmaxexample](softmaxexample.png)

## NLLLoss

![NLLLoss](NLLLoss.png)

![torchcrossEntropy](torchcrossEntropy.png)

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])  # 归一化,均值和方差

train_dataset = datasets.MNIST(root='../dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)  # -1其实就是自动获取mini_batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 最后一层不做激活，不进行非线性变换


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        # 获得一个批次的数据和标签
        inputs, target = data
        optimizer.zero_grad()
        # 获得模型预测结果(64, 10)
        outputs = model(inputs)
        # 交叉熵代价函数outputs(64,10),target（64）
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)  # dim = 1 列是第0个维度，行是第1个维度
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # 张量之间的比较运算
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

# 卷积神经网路 CNN

图片全连接后 可能会丧失一些原有的图片的空间的特征，比如图片中两点列相邻但是全连接后岔开

卷积神经网络将图像按原始空间结构进行保存

输入张量的维度 与 输出张量的维度

![cnn](cnn.png)

Feature Extraction 特征提取器 		Classification 分类器

## 图像是什么？

RGB——

栅格图像			矢量图像 

![Convolution](Convolution.png)

![convolutioninputchannels](convolutioninputchannels.png)

![Convolution n input channels and M output Channels](Convolution n input channels and M output Channels.png)

![Convolution Layer](Convolution Layer.png)

## padding

![padding](padding.png)

## stride 步长

每次索引的坐标+

可有效降低图像的宽度和高度

## Max Pooling  最大池化层

![MaxPooling](MaxPooling.png)

分成n*n组，找每组的最大值

![Simple Example](Simple Example.png)

减少代码冗余：函数/类

Concatenate：拼接 将张量沿着通道连接

![concatenate](concatenate.png)

What is 1×1 convolution？

信息融合  改变通道数量

![1×1 convolution](1×1 convolution.png) 

![why  is 1×1](why  is 1×1.png)

# 循环神经网络 RNN

DNN：Dense（Deep） 稠密神经网络

RNN：处理具有序列连接的输入数据（例如：金融股市、天气、自然语言处理）

## RNN Cell

本质：线形层，把某个维度映射到另一个维度的空间。 Linear

![image-20221214165453611](D:\人工智能\photo\pytorch md\RNN Cell)

![image-20221214171337356](D:\人工智能\photo\pytorch md\RNN Cell2)

![image-20221214171356832](D:\人工智能\photo\pytorch md\RNN Cell3)

![image-20221214171723190](D:\人工智能\photo\pytorch md\RNN Cell in Pytorch)

![image-20221214174335513](D:\人工智能\photo\pytorch md\RNN Cell in Pytorch 2)

## How to use RNNCell

![image-20221214174455158](D:\人工智能\photo\pytorch md\use RNNCell1)

![image-20221214174611539](D:\人工智能\photo\pytorch md\use RNNCell2)

![image-20221214174814307](D:\人工智能\photo\pytorch md\use RNNCell3)

## How to use RNN

![image-20221214174959457](D:\人工智能\photo\pytorch md\use RNN1)

![image-20221214180055695](D:\人工智能\photo\pytorch md\use RNN2)

![image-20221214180951812](D:\人工智能\photo\pytorch md\use RNN3)

![image-20221214181039389](D:\人工智能\photo\pytorch md\use RNN4)

![image-20221214181656965](D:\人工智能\photo\pytorch md\use RNN5)

![image-20221214181925459](D:\人工智能\photo\pytorch md\batch_first1)

![image-20221214182021109](D:\人工智能\photo\pytorch md\batch_first2)

# 李宏毅深度学习



##Pytorch Tutorial

![image-20230214164320489](D:\人工智能\photo\pytorch md\pytorch turtorial)

###Step1 Load Data

torch.utils.data.Dataset & torch.utils.data.DataLoader

- Dataset:	stores data samples and expected values  将Python定义class将资料一笔笔读进来打包。                 
- Dataloader: groups data in batches, enables multiprocessing 将Dataset中一个个的资料合并成一个个batch，平行化处理

- dataset = MyDataset(file)
- dataloader = Dataloader(dataset, batch_size, shuffle = True)

![image-20230214144609312](D:\人工智能\photo\pytorch md\Dataset)

![image-20230214144900704](D:\人工智能\photo\pytorch md\Dataset2)



#### Tensors

- High-dimensional matrices (arrays)

**Shape of Tensors**

​	Check with .shape() 

![image-20230214145346135](D:\人工智能\photo\pytorch md\Tensor)

##### Creating tensors

![image-20230214150139489](D:\人工智能\photo\pytorch md\creat tensors)



##### Common operations

- Addition

  ​						z = x + y

  ​						z=torch.add(x,y)

- Subtraction

  ​						z = x - y

  ​						z= torch.sub(x,y)

- Power

  ​						y = x.pow(2)

- Summation

  ​						y = x.sum()

- Mean

  ​						y = x.mean()

- Transpose:transpose two specified dimensions

  ​				![image-20230214155039585](D:\人工智能\photo\pytorch md\transpose)		

- Squeeze

![image-20230214155219945](D:\人工智能\photo\pytorch md\squeezw)

- Unsqueeze

![image-20230214155829437](D:\人工智能\photo\pytorch md\unsqueeze)

- Cat

![image-20230214155853582](D:\人工智能\photo\pytorch md\cat)

- Device

![image-20230214161322544](D:\人工智能\photo\pytorch md\device)

##### Gradient Calculation  

```python
x = torch.tensor([[1., 0.], [-1., 1.]], requires_grad=True)
z= x.pow(2).sum()
z.backward()
x.grad
```



###Step 2 Define Neural Network

**torch.nn.Module**

- Linear Layer(Fully-connected Layer)
- Non-Linear Activation Functions  

####Build your own neural network

![image-20230214163500685](D:\人工智能\photo\pytorch md\build network)

![image-20230214163623783](D:\人工智能\photo\pytorch md\build network2)



### Step 3 Loss Function

torch.nn.MSELoss
torch.nn.CrossEntropyLoss etc.

![image-20230214163759146](D:\人工智能\photo\pytorch md\Loss functions)

### Step 4 Optimization Algorithm

![image-20230214164200244](D:\人工智能\photo\pytorch md\optim)

![image-20230214164218386](D:\人工智能\photo\pytorch md\optim2)

### Step 5 Entire Procedure

![image-20230214164603078](D:\人工智能\photo\pytorch md\nn training setup)

![image-20230214164730311](D:\人工智能\photo\pytorch md\nn training loop)

![image-20230214165115010](D:\人工智能\photo\pytorch md\nn Validation loop)

![image-20230214165337476](D:\人工智能\photo\pytorch md\nn testing loop)

![image-20230214170501908](D:\人工智能\photo\pytorch md\notice)

### Save/Load Trained Models

![image-20230214170609228](D:\人工智能\photo\pytorch md\save load models)

## Gradient Decent

### Backpropagation

​	**Chain Rule**

![image-20230215200837078](D:\人工智能\photo\pytorch md\chainrule)

![image-20230215200915833](D:\人工智能\photo\pytorch md\backpropagation)

![image-20230215201437884](D:\人工智能\photo\pytorch md\Backpropagation2)

![image-20230215201710167](D:\人工智能\photo\pytorch md\backpropagation3)



## Regression

[Regression李宏毅]:(https://github.com/Fafa-DL/Lhy_Machine_Learning/blob/main/选修 To Learn More/第一节/Regression.pdf)

### Step 1 Model

![image-20230216104238590](D:\人工智能\photo\pytorch md\regression model)

### Step2 Goodness of Function

![image-20230216104604935](D:\人工智能\photo\pytorch md\regression goodnessof function)

### Step3 Best Function

![image-20230216104653344](D:\人工智能\photo\pytorch md\regression best function)

![image-20230216104725243](D:\人工智能\photo\pytorch md\gradient descent)

Local minima 				Global minima

### Model Selection

![image-20230216104921552](D:\人工智能\photo\pytorch md\regression model selection)

****Overfitting**:   A  more complex model does not always lead to better performance on testing data.**

![image-20230220164830103](D:\人工智能\photo\pytorch md\image-20230220164830103.png)

[回归 模型选择](https://github.com/Fafa-DL/Lhy_Machine_Learning/blob/main/选修 To Learn More/第一节/Regression.pdf)

### **Regularization** 

Redefine Loss function



![image-20230216105114915](D:\人工智能\photo\pytorch md\regression regularization)

**Smoother**：meaning is when the input change, the output change smaller(smooth)

Why we want a smooth function?: If some noises corrupt input x~i~ When testing， a smooth function has less influence.

![image-20230220165823191](D:\人工智能\photo\pytorch md\image-20230220165823191.png)







## Classification

### Gaussian Distribution

![image-20230216142409305](D:\人工智能\photo\pytorch md\gaussian diutution)



![image-20230216142609834](D:\人工智能\photo\pytorch md\probability from class)

![image-20230216142701986](D:\人工智能\photo\pytorch md\Maximum Likehood)

![image-20230216143712066](D:\人工智能\photo\pytorch md\maximum likehood2)

## Logistic Regression

![image-20230216144705245](D:\人工智能\photo\pytorch md\Logistic regree)

![image-20230216144840928](D:\人工智能\photo\pytorch md\logistic regree2)

![image-20230216150359309](D:\人工智能\photo\pytorch md\logistic regree3)

![image-20230216150747837](D:\人工智能\photo\pytorch md\logistic regress4)

![image-20230216152128713](D:\人工智能\photo\pytorch md\logistic regree5)

![image-20230216153003383](D:\人工智能\photo\pytorch md\logistic regree6)

![image-20230216153048539](D:\人工智能\photo\pytorch md\logistic regree7)

![image-20230216153209267](D:\人工智能\photo\pytorch md\image-20230216153209267.png)

 ![image-20230216153429112](D:\人工智能\photo\pytorch md\image-20230216153429112.png)

![image-20230216154717652](D:\人工智能\photo\pytorch md\cross entropy)

![image-20230216155931419](D:\人工智能\photo\pytorch md\image-20230216155931419.png)

Generative model 进行了一定的假设

![image-20230216160817270](D:\人工智能\photo\pytorch md\image-20230216160817270.png)

### Multi-class Classfication

![image-20230216161736917](D:\人工智能\photo\pytorch md\image-20230216161736917.png)

![image-20230216162028665](D:\人工智能\photo\pytorch md\image-20230216162028665.png)

### Limitation of Logistic Regression

Feature Transformation

![image-20230216162441172](D:\人工智能\photo\pytorch md\image-20230216162441172.png)

![image-20230216163421455](D:\人工智能\photo\pytorch md\image-20230216163421455.png)

## General Guidance

![image-20230217162531424](D:\人工智能\photo\pytorch md\image-20230217162531424.png)

### Model Bias



![image-20230217163405067](D:\人工智能\photo\pytorch md\image-20230217163405067.png)

### OPtimization Issue



![image-20230217163545349](D:\人工智能\photo\pytorch md\image-20230217163545349.png)

### Overfitting



![image-20230217163840304](D:\人工智能\photo\pytorch md\image-20230217163840304.png)

![image-20230217164009238](D:\人工智能\photo\pytorch md\image-20230217164009238.png)

![image-20230217164404659](D:\人工智能\photo\pytorch md\image-20230217164404659.png)

**Data augmentation 要根据资料特性合理设置**

![image-20230217164553921](D:\人工智能\photo\pytorch md\image-20230217164553921.png)

![image-20230217164710202](D:\人工智能\photo\pytorch md\image-20230217164710202.png)

![image-20230217164849894](D:\人工智能\photo\pytorch md\image-20230217164849894.png)

![image-20230217165040061](D:\人工智能\photo\pytorch md\image-20230217165040061.png)

![image-20230217165245444](D:\人工智能\photo\pytorch md\image-20230217165245444.png)

**模型选择 有可能恰好模型产生随机全正确**

![image-20230217170146204](D:\人工智能\photo\pytorch md\image-20230217170146204.png)

![image-20230217170342996](D:\人工智能\photo\pytorch md\image-20230217170342996.png)

#### used a validation set, but model still overfitted?

![image-20230222150042220](D:\人工智能\photo\pytorch md\image-20230222150042220.png)

![image-20230222150507570](D:\人工智能\photo\pytorch md\image-20230222150507570.png)

















### Mismatch

![image-20230217170703851](D:\人工智能\photo\pytorch md\image-20230217170703851.png)



## ptimization Fails

![image-20230217174738922](D:\人工智能\photo\pytorch md\image-20230217174738922.png)

### local minima

### saddle point

![image-20230217175230940](D:\人工智能\photo\pytorch md\image-20230217175230940.png)

![image-20230217175445395](D:\人工智能\photo\pytorch md\image-20230217175445395.png)

![image-20230217175935273](D:\人工智能\photo\pytorch md\image-20230217175935273.png)

![image-20230217185855119](D:\人工智能\photo\pytorch md\image-20230217185855119.png)

![image-20230217190539102](D:\人工智能\photo\pytorch md\image-20230217190539102.png)

#### Don't afraid of saddle point

![image-20230217190851402](D:\人工智能\photo\pytorch md\image-20230217190851402.png)

![image-20230217192854428](D:\人工智能\photo\pytorch md\image-20230217192854428.png)



## Batch and Momentum

### Batch

![image-20230217194431309](D:\人工智能\photo\pytorch md\image-20230217194431309.png)

![image-20230217195204374](D:\人工智能\photo\pytorch md\image-20230217195204374.png)

![image-20230217200343404](D:\人工智能\photo\pytorch md\image-20230217200343404.png)

![image-20230217200535811](D:\人工智能\photo\pytorch md\image-20230217200535811.png)

   ![image-20230217200708347](D:\人工智能\photo\pytorch md\image-20230217200708347.png)

·Small batch is better on testing data

![image-20230217201131973](D:\人工智能\photo\pytorch md\image-20230217201131973.png)

![image-20230217201303403](D:\人工智能\photo\pytorch md\image-20230217201303403.png)



### Momentum

![image-20230217201537450](D:\人工智能\photo\pytorch md\image-20230217201537450.png)

![image-20230217202010268](D:\人工智能\photo\pytorch md\image-20230217202010268.png)



## Adptive Learning Rate

![image-20230219103610912](D:\人工智能\photo\pytorch md\image-20230219103610912.png)

![image-20230219104510420](D:\人工智能\photo\pytorch md\image-20230219104510420.png)

**在某一个方向上梯度小希望学习率大一些，在某个方向梯度大一些希望学习率小一些**

![image-20230219104723217](D:\人工智能\photo\pytorch md\image-20230219104723217.png)

###Root Mean Square

![image-20230219105222073](D:\人工智能\photo\pytorch md\image-20230219105222073.png)

![image-20230219105414500](D:\人工智能\photo\pytorch md\image-20230219105414500.png)

 ![image-20230219105807737](D:\人工智能\photo\pytorch md\image-20230219105807737.png)

### RMSProop

![image-20230219110213837](D:\人工智能\photo\pytorch md\image-20230219110213837.png)

![image-20230219110527256](D:\人工智能\photo\pytorch md\image-20230219110527256.png)

#### Adam

![image-20230219110610162](D:\人工智能\photo\pytorch md\image-20230219110610162.png)

![image-20230219113320260](D:\人工智能\photo\pytorch md\image-20230219113320260.png)

### New Optimizers for Deep Learning

[Lhy_Machine_Learning/Optimization.pdf at main · Fafa-DL/Lhy_Machine_Learning (github.com)](https://github.com/Fafa-DL/Lhy_Machine_Learning/blob/main/选修 To Learn More/第二节/Optimization.pdf)

![image-20230219173231485](D:\人工智能\photo\pytorch md\image-20230219173231485.png)

![image-20230219173402903](D:\人工智能\photo\pytorch md\image-20230219173402903.png)

![image-20230219173520988](D:\人工智能\photo\pytorch md\image-20230219173520988.png)





#### SGD

#### SGD with Momentum (SGDM)

![image-20230219174817789](D:\人工智能\photo\pytorch md\image-20230219174817789.png)

#### Adagraad

#### RMSProp

#### Adam

![image-20230219175650535](D:\人工智能\photo\pytorch md\image-20230219175650535.png)

<img src="D:\人工智能\photo\pytorch md\image-20230219175745769.png" alt="image-20230219175745769" style="zoom:50%;" />

![image-20230219182432823](D:\人工智能\photo\pytorch md\image-20230219182432823.png)

尝试解释为什么Adam和SGDM训练不一样：

​		Loss Function比较平坦，训练和测试的的Minimum就会比较接近

##### Simply combine Adam with SGDM？----SWATS

![image-20230219182959481](D:\人工智能\photo\pytorch md\image-20230219182959481.png)

##### Towards Improving Adam

[视频解释 39:00](https://www.bilibili.com/video/BV1VN4y1P7Zj?t=2352.9&p=26)

假设β~1~=0，则未使用m~t~，focous adaptive learning rate对Adam造成的影响。通过v~t~表达式可知v~t~受到梯度的影响会维持1/(1-0.999)

![image-20230219184733050](D:\人工智能\photo\pytorch md\image-20230219184733050.png)

![image-20230219184845319](D:\人工智能\photo\pytorch md\image-20230219184845319.png)

###### AMSGrad

![image-20230219185255264](D:\人工智能\photo\pytorch md\image-20230219185255264.png)

##### Towards Improving SGDM

![image-20230219185732554](D:\人工智能\photo\pytorch md\image-20230219185732554.png)

**Engineering：learning rate很小或很大精度都不会很好，适中**

![image-20230219185906756](D:\人工智能\photo\pytorch md\image-20230219185906756.png)

##### Does Adam need warm-up?

![image-20230220151750842](D:\人工智能\photo\pytorch md\image-20230220151750842.png)

为什么Adam已经Adaptive rate为什么还需要warm up?：上图实际实验说明（横轴为Iteration，纵轴为gradient 的distribution），前几步的估计不准

![image-20230220152751776](D:\人工智能\photo\pytorch md\image-20230220152751776.png)

![image-20230220153041140](D:\人工智能\photo\pytorch md\image-20230220153041140.png)

![image-20230220153440025](D:\人工智能\photo\pytorch md\image-20230220153440025.png)

![image-20230220154709135](D:\人工智能\photo\pytorch md\image-20230220154709135.png)

![image-20230220154902836](D:\人工智能\photo\pytorch md\image-20230220154902836.png)

##### More than momentum

![image-20230220155117699](D:\人工智能\photo\pytorch md\image-20230220155117699.png)



![image-20230220160235068](D:\人工智能\photo\pytorch md\image-20230220160235068.png)

**▽L(θ~t-1~-λm~t-1~)表示预测下一点的梯度时如何**

![image-20230220161633898](D:\人工智能\photo\pytorch md\image-20230220161633898.png)

##### Nadam

![image-20230220162817076](D:\人工智能\photo\pytorch md\image-20230220162817076.png)

![image-20230220163803033](D:\人工智能\photo\pytorch md\image-20230220163803033.png)

#### Something helps optimization

![image-20230220163905897](D:\人工智能\photo\pytorch md\image-20230220163905897.png)



![image-20230220164300905](D:\人工智能\photo\pytorch md\image-20230220164300905.png)

![image-20230220164324808](D:\人工智能\photo\pytorch md\image-20230220164324808.png)



















## Learning Rate Scheduling 

将Learning Rate与时间有关

![image-20230219113538883](D:\人工智能\photo\pytorch md\image-20230219113538883.png)

![image-20230219114058621](D:\人工智能\photo\pytorch md\image-20230219114058621.png)



![image-20230219113906542](D:\人工智能\photo\pytorch md\image-20230219113906542.png)

 

![image-20230219114343052](D:\人工智能\photo\pytorch md\image-20230219114343052.png)

## 再探宝可梦、数码宝贝分类器---浅谈机器学习原理

![image-20230219134611063](D:\人工智能\photo\pytorch md\image-20230219134611063.png)

#### 模型复杂度

![image-20230219140424624](D:\人工智能\photo\pytorch md\image-20230219140424624.png)

![image-20230219140828795](D:\人工智能\photo\pytorch md\image-20230219140828795.png)



#### i.i.d

![image-20230219141400722](D:\人工智能\photo\pytorch md\image-20230219141400722.png)

![image-20230219141507996](D:\人工智能\photo\pytorch md\image-20230219141507996.png)

![image-20230219142322281](D:\人工智能\photo\pytorch md\image-20230219142322281.png)

### What train sample do we want?

**train得到的模型好坏取决于sample时的资料**

![image-20230219143624556](D:\人工智能\photo\pytorch md\image-20230219143624556.png)

L(h^all^, D~all~ )一定会比L(h^train^, D~all~ )小

![image-20230219143741645](D:\人工智能\photo\pytorch md\image-20230219143741645.png)

![image-20230219143939593](D:\人工智能\photo\pytorch md\image-20230219143939593.png)

   ### General

![image-20230219144104393](D:\人工智能\photo\pytorch md\image-20230219144104393.png)

![image-20230219150624632](D:\人工智能\photo\pytorch md\image-20230219150624632.png)

<img src="D:\人工智能\photo\pytorch md\image-20230219150949584.png" alt="image-20230219150949584" style="zoom: 50%;" />

<img src="D:\人工智能\photo\pytorch md\image-20230219151206774.png" alt="image-20230219151206774" style="zoom:50%;" />

<img src="D:\人工智能\photo\pytorch md\image-20230219151432675.png" alt="image-20230219151432675" style="zoom:50%;" />

<img src="D:\人工智能\photo\pytorch md\image-20230219151632461.png" alt="image-20230219151632461" style="zoom:50%;" />

![image-20230219151824419](D:\人工智能\photo\pytorch md\image-20230219151824419.png)

![image-20230219151902001](D:\人工智能\photo\pytorch md\image-20230219151902001.png)

![image-20230219151916700](D:\人工智能\photo\pytorch md\image-20230219151916700.png)

![image-20230219152147020](D:\人工智能\photo\pytorch md\image-20230219152147020.png)

![image-20230219152515395](D:\人工智能\photo\pytorch md\image-20230219152515395.png)

![image-20230219152656072](D:\人工智能\photo\pytorch md\image-20230219152656072.png)

![image-20230219153140826](D:\人工智能\photo\pytorch md\image-20230219153140826.png)

![image-20230219153347413](D:\人工智能\photo\pytorch md\image-20230219153347413.png)



### Why more parameters are easier to overfit?

## 鱼与熊掌可以兼得的机器学习

### Review：Why hidden layer?

可以通过一个hidden layer找出所有可能的function

![image-20230222151535069](D:\人工智能\photo\pytorch md\image-20230222151535069.png)

![image-20230222151614261](D:\人工智能\photo\pytorch md\image-20230222151614261.png)



![image-20230222151643870](D:\人工智能\photo\pytorch md\image-20230222151643870.png)

![image-20230222152253769](D:\人工智能\photo\pytorch md\image-20230222152253769.png)



![image-20230222152423524](D:\人工智能\photo\pytorch md\image-20230222152423524.png)

![image-20230222152447853](D:\人工智能\photo\pytorch md\image-20230222152447853.png)

![image-20230222153444309](D:\人工智能\photo\pytorch md\image-20230222153444309.png)

探讨网络深层的作用

![image-20230222154525682](D:\人工智能\photo\pytorch md\image-20230222154525682.png)

![image-20230222154753511](D:\人工智能\photo\pytorch md\image-20230222154753511.png)



![image-20230222155100307](D:\人工智能\photo\pytorch md\image-20230222155100307.png)

![image-20230222155207551](D:\人工智能\photo\pytorch md\image-20230222155207551.png)



![image-20230222155229862](D:\人工智能\photo\pytorch md\image-20230222155229862.png)







## HW2

## Concolutional Neural Network(CNN)

**Network Architecture designed for Image**

![image-20230221190847727](D:\人工智能\photo\pytorch md\image-20230221190847727.png)

对电脑来说一张图片是什么？

![image-20230221191148146](D:\人工智能\photo\pytorch md\image-20230221191148146.png)

![image-20230222133926254](D:\人工智能\photo\pytorch md\image-20230222133926254.png)

参数过多容易overfitting

![image-20230222134324377](D:\人工智能\photo\pytorch md\image-20230222134324377.png)

### Receptive field

![image-20230222134630110](D:\人工智能\photo\pytorch md\image-20230222134630110.png)



![image-20230222135225522](D:\人工智能\photo\pytorch md\image-20230222135225522.png)

·

![image-20230222135722131](D:\人工智能\photo\pytorch md\image-20230222135722131.png)

![image-20230222135846875](D:\人工智能\photo\pytorch md\image-20230222135846875.png)

parameter sharing

![image-20230222140105874](D:\人工智能\photo\pytorch md\image-20230222140105874.png)

![image-20230222140225472](D:\人工智能\photo\pytorch md\image-20230222140225472.png)



![image-20230222140412998](D:\人工智能\photo\pytorch md\image-20230222140412998.png)

![image-20230222140504306](D:\人工智能\photo\pytorch md\image-20230222140504306.png)

![image-20230222140728856](D:\人工智能\photo\pytorch md\image-20230222140728856.png)

![image-20230222140813030](D:\人工智能\photo\pytorch md\image-20230222140813030.png)

![image-20230222140957944](D:\人工智能\photo\pytorch md\image-20230222140957944.png)

若filter大小一直设置3*3，会使network不能看更大的图吗？

![image-20230222141306342](D:\人工智能\photo\pytorch md\image-20230222141306342.png)

![image-20230222141343778](D:\人工智能\photo\pytorch md\image-20230222141343778.png)



![image-20230222141401887](D:\人工智能\photo\pytorch md\image-20230222141401887.png)



同样的小目标可以出现在不同地方所以不同区域可以共用参数。

### Pooling-Max Pooling

![image-20230222142228736](D:\人工智能\photo\pytorch md\image-20230222142228736.png)

Max Pooling作用：把图片变小

Pooling主要的作用是减少运算量

![image-20230222143015795](D:\人工智能\photo\pytorch md\image-20230222143015795.png)

![image-20230222143054258](D:\人工智能\photo\pytorch md\image-20230222143054258.png)

![image-20230222143303706](D:\人工智能\photo\pytorch md\image-20230222143303706.png) 

![image-20230222143421544](D:\人工智能\photo\pytorch md\image-20230222143421544.png)



![image-20230222143633013](D:\人工智能\photo\pytorch md\image-20230222143633013.png)



## Spatial Transformer Layer

![image-20230222162327297](D:\人工智能\photo\pytorch md\image-20230222162327297.png)



![image-20230222163100727](D:\人工智能\photo\pytorch md\image-20230222163100727.png)

![image-20230222163346290](D:\人工智能\photo\pytorch md\image-20230222163346290.png)

![image-20230222163816076](D:\人工智能\photo\pytorch md\image-20230222163816076.png)

![image-20230222164539375](D:\人工智能\photo\pytorch md\image-20230222164539375.png)

![image-20230222164711179](D:\人工智能\photo\pytorch md\image-20230222164711179.png)

![image-20230222164815484](D:\人工智能\photo\pytorch md\image-20230222164815484.png)

![image-20230222165047469](D:\人工智能\photo\pytorch md\image-20230222165047469.png)



![image-20230222170141809](D:\人工智能\photo\pytorch md\image-20230222170141809.png)

![image-20230222170814869](D:\人工智能\photo\pytorch md\image-20230222170814869.png)

## Self-attention

解决问题：network input is a set of vectors not a vector

![image-20230223161320964](D:\人工智能\photo\pytorch md\image-20230223161320964.png)

例子：文字处理，假设处理的是句子每个句子的长度都不一样，将句子每一个词汇都描绘成向量，则句子是一个Vector Set

如何将词汇表示成向量？---One-hot Encoding，问题假设每个词汇之间没有关系

![image-20230223162805885](D:\人工智能\photo\pytorch md\image-20230223162805885.png)

例子2：声音讯号

![image-20230223163103560](D:\人工智能\photo\pytorch md\image-20230223163103560.png)



![image-20230223163644962](D:\人工智能\photo\pytorch md\image-20230223163644962.png)

![image-20230223164116773](D:\人工智能\photo\pytorch md\image-20230223164116773.png)

![image-20230223164131670](D:\人工智能\photo\pytorch md\image-20230223164131670.png)

### Sequence Labeling

![image-20230223164432190](D:\人工智能\photo\pytorch md\image-20230223164432190.png)



### Self-attention

**How working?**

self-attention会接收一整个sequence资料，input 多少vector就输出多少vector，输出vector考虑一整个sequence得到。

![image-20230223164858274](D:\人工智能\photo\pytorch md\image-20230223164858274.png)

self-attention可以很多次，fully connection network和self-attention可以交替使用，fully connection network处理某一位置资料，self-attention处理整个sequence

![image-20230223165123881](D:\人工智能\photo\pytorch md\image-20230223165123881.png)

![image-20230223165253237](D:\人工智能\photo\pytorch md\image-20230223165253237.png)

![image-20230223165405133](D:\人工智能\photo\pytorch md\image-20230223165405133.png)



![image-20230223170452941](D:\人工智能\photo\pytorch md\image-20230223170452941.png)



![image-20230223170706124](D:\人工智能\photo\pytorch md\image-20230223170706124.png)

![image-20230223170837944](D:\人工智能\photo\pytorch md\image-20230223170837944.png)

![image-20230223173855690](D:\人工智能\photo\pytorch md\image-20230223173855690.png)

从矩阵乘法解释Self-attention：

![image-20230223180959958](D:\人工智能\photo\pytorch md\image-20230223180959958.png)

![image-20230223181544814](D:\人工智能\photo\pytorch md\image-20230223181544814.png)

![image-20230224163731902](D:\人工智能\photo\pytorch md\image-20230224163731902.png)

![image-20230224164123140](D:\人工智能\photo\pytorch md\image-20230224164123140.png)

#### Multi-head Self-attention

![image-20230224164758954](D:\人工智能\photo\pytorch md\image-20230224164758954.png)

![image-20230224164821859](D:\人工智能\photo\pytorch md\image-20230224164821859.png)

Self attention没有位置信息

![image-20230224165938999](D:\人工智能\photo\pytorch md\image-20230224165938999.png)

![image-20230224170054154](D:\人工智能\photo\pytorch md\image-20230224170054154.png)

语言辨识：输入向量会很大，只看很小范围。

![image-20230224170251429](D:\人工智能\photo\pytorch md\image-20230224170251429.png)

![image-20230224171004645](D:\人工智能\photo\pytorch md\image-20230224171004645.png)

![image-20230224171217528](D:\人工智能\photo\pytorch md\image-20230224171217528.png)

CNN是self-attention的特例

Self-attention与CNN比较，模型复杂，容易过拟合

![image-20230224171522251](D:\人工智能\photo\pytorch md\image-20230224171522251.png)

![image-20230224174602637](D:\人工智能\photo\pytorch md\image-20230224174602637.png)

### 各式各样的Attention

![image-20230227155038921](D:\人工智能\photo\pytorch md\image-20230227155038921.png)

N×N的计算量特别大

![image-20230227155812871](D:\人工智能\photo\pytorch md\image-20230227155812871.png)

当Input的N非常大时，以下的处理才会很有效果。

![image-20230227161036092](D:\人工智能\photo\pytorch md\image-20230227161036092.png)

####Skip Some Calculations

N×N矩阵中有些位置不需要计算

![image-20230227161214393](D:\人工智能\photo\pytorch md\image-20230227161214393.png)

##### Local Attention/Truncated Attention

![image-20230227161430915](D:\人工智能\photo\pytorch md\image-20230227161430915.png)

每次attention只能看见小范围，与CNN相似

##### Stride Attention

![image-20230227161633916](D:\人工智能\photo\pytorch md\image-20230227161633916.png)

##### Global Attention

[讲解 第14分钟](https://www.bilibili.com/video/BV1VN4y1P7Zj?t=871.0&p=51)

![image-20230227162251395](D:\人工智能\photo\pytorch md\image-20230227162251395.png)

![image-20230227162444796](D:\人工智能\photo\pytorch md\image-20230227162444796.png)

**用Multi-head attention**

![image-20230227162554550](D:\人工智能\photo\pytorch md\image-20230227162554550.png)

### Focous on Critical Pats

![image-20230227162752928](D:\人工智能\photo\pytorch md\image-20230227162752928.png)

#### Clustering

相近的vector属于相同的cluster，不相近的属于不同的cluster。

![image-20230227163049571](D:\人工智能\photo\pytorch md\image-20230227163049571.png)

![image-20230227163128433](D:\人工智能\photo\pytorch md\image-20230227163128433.png)

### Learnable Patterns

通过Learned计算哪些地方需要计算

![image-20230227163638822](D:\人工智能\photo\pytorch md\image-20230227163638822.png)



Sinkhorn Sorting Network如何实现加速的？[解释 第28分钟](https://www.bilibili.com/video/BV1VN4y1P7Zj?t=1802.4&p=51)

### Do we need full attention matrix?

[定位 第31分钟](https://www.bilibili.com/video/BV1VN4y1P7Zj?t=1802.4&p=51)

![image-20230227164502837](D:\人工智能\photo\pytorch md\image-20230227164502837.png)

![image-20230227164543048](D:\人工智能\photo\pytorch md\image-20230227164543048.png)

 ![image-20230227164613934](D:\人工智能\photo\pytorch md\image-20230227164613934.png)

![image-20230227164942378](D:\人工智能\photo\pytorch md\image-20230227164942378.png)

处理query根据问题考虑，若是作业2那种会减少label数量

#### Reduce Nember of Keys

![image-20230227171901971](D:\人工智能\photo\pytorch md\image-20230227171901971.png)



![image-20230227172155132](D:\人工智能\photo\pytorch md\image-20230227172155132.png)

![image-20230227172136198](D:\人工智能\photo\pytorch md\image-20230227172136198.png)

![image-20230227172326858](D:\人工智能\photo\pytorch md\image-20230227172326858.png)





![image-20230227172435027](D:\人工智能\photo\pytorch md\image-20230227172435027.png)

![image-20230227172633979](D:\人工智能\photo\pytorch md\image-20230227172633979.png)



![image-20230227173058847](D:\人工智能\photo\pytorch md\image-20230227173058847.png)



![image-20230227173153780](D:\人工智能\photo\pytorch md\image-20230227173153780.png)

![image-20230227173605771](D:\人工智能\photo\pytorch md\image-20230227173605771.png)



![image-20230227174638919](D:\人工智能\photo\pytorch md\image-20230227174638919.png)

![image-20230227175224594](D:\人工智能\photo\pytorch md\image-20230227175224594.png)



![image-20230227175551056](D:\人工智能\photo\pytorch md\image-20230227175551056.png)

![image-20230227175834055](D:\人工智能\photo\pytorch md\image-20230227175834055.png)

![image-20230227175849100](D:\人工智能\photo\pytorch md\image-20230227175849100.png)

![image-20230227180202579](D:\人工智能\photo\pytorch md\image-20230227180202579.png)

![image-20230227180222407](D:\人工智能\photo\pytorch md\image-20230227180222407.png)

![image-20230227180355433](D:\人工智能\photo\pytorch md\image-20230227180355433.png)

![image-20230227180413220](D:\人工智能\photo\pytorch md\image-20230227180413220.png)



#### Synthesizer







## RNN

[RNN PART1](https://www.youtube.com/watch?v=xCGidAeyS4M)

[RNN PART2](https://www.youtube.com/watch?v=rTqmWlnwz_0)

### Example Application

![image-20230824094358392](D:\人工智能\image-20230824094358392.png)

![image-20230824094857191](D:\人工智能\image-20230824094857191.png)

![image-20230824094637898](D:\人工智能\image-20230824094637898.png)

![image-20230824094806665](D:\人工智能\image-20230824094806665.png)

希望神经网络是有记忆的：如输入台北只能输出是目的地而不能分辨此时的台北是出发地还是到达地

![image-20230824095157640](D:\人工智能\image-20230824095157640.png)

###ElmanNetwork



![image-20230824095752464](D:\人工智能\image-20230824095752464.png)

![image-20230824100037972](D:\人工智能\image-20230824100037972.png)

![image-20230824100110111](D:\人工智能\image-20230824100110111.png)

 ![image-20230824100146747](D:\人工智能\image-20230824100146747.png)

### Jordan Network

![image-20230824100319375](D:\人工智能\image-20230824100319375.png)

Jordan Network学习效果可能比较好

### Bidirectional RNN

产生输出时看的学习到的范围比较广

![image-20230824100517783](D:\人工智能\image-20230824100517783.png)

### LSTM

Long Short-term MEMORY

Input Gate：只有打开时才能将值写入Memory Cell，打开关闭可以有NN自己学习

Output Gate：决定外界可不可以将值读出来

Forget Gate：决定何时将Memory Cell忘掉，打开时代表记住，关闭代表遗忘

![image-20230824101358392](D:\人工智能\image-20230824101358392.png)

![image-20230824101956681](D:\人工智能\image-20230824101956681.png)

激活函数通常旋转sigmoid是因为此值在0-1.可以代表打开程度

![image-20230824102122860](D:\人工智能\image-20230824102122860.png)

![image-20230824102428705](D:\人工智能\image-20230824102428705.png)

### Difference between RNN and LSTM

![image-20230824102900519](D:\人工智能\image-20230824102900519.png)

![image-20230824102825524](D:\人工智能\image-20230824102825524.png)

![image-20230824103223033](D:\人工智能\image-20230824103223033.png)

![image-20230824103345925](D:\人工智能\image-20230824103345925.png)

![image-20230824103442370](D:\人工智能\image-20230824103442370.png)

### Multiple-layer LSTM



![image-20230824103522067](D:\人工智能\image-20230824103522067.png)

### Learning Target

结果的cost:每个RNN的output和reference vector的cross entropy和 去minimize

![image-20230824105847788](D:\人工智能\image-20230824105847788.png)

### BPTT

![image-20230824110021184](D:\人工智能\image-20230824110021184.png)

![image-20230824110158861](D:\人工智能\image-20230824110158861.png)

![image-20230824110322220](D:\人工智能\image-20230824110322220.png)

![image-20230824110554974](D:\人工智能\image-20230824110554974.png)

Clipping： 当gradient大于某个threshold时，就不要超过threshold

![image-20230824111209049](D:\人工智能\image-20230824111209049.png)

为什么RNN误差会很崎岖：RNN训练问题，源自在时间和时间转换transition时反复使用，从memory接到neuron的一组weight反复被使用，所以ｗ有变化，则会产生如上图gradient会有时很大有时很小

使用LSTM时候可以避免gradient平坦，因此可以将ｌｅａｒｎｉｎｇ　ｒａｔｅ设的小，如下图

![image-20230824112210517](D:\人工智能\image-20230824112210517.png)

参数多可能会带来Over fitting的情况

![image-20230824112329435](D:\人工智能\image-20230824112329435.png)





## GNN



暂时略过

## Quick Introduction of Batch Normalization

![image-20230226104124832](D:\人工智能\photo\pytorch md\image-20230226104124832.png)

![image-20230226104007327](D:\人工智能\photo\pytorch md\image-20230226104007327.png)

难训练

给feature中不同的dimension，有同样的数值范围。

### Feature Normalization

![image-20230226104721560](D:\人工智能\photo\pytorch md\image-20230226104721560.png)

![image-20230226104935859](D:\人工智能\photo\pytorch md\image-20230226104935859.png)

x正规化后，W作用也可能会使训练困难，feature Normalization可以选择在激活函数之前或之后；选择sigmoid做激活函数推荐对z做feature Normalization。

![image-20230226105226151](D:\人工智能\photo\pytorch md\image-20230226105226151.png)

![image-20230226105254442](D:\人工智能\photo\pytorch md\image-20230226105254442.png)

![image-20230226105619749](D:\人工智能\photo\pytorch md\image-20230226105619749.png)

![image-20230226105830567](D:\人工智能\photo\pytorch md\image-20230226105830567.png)



β、γ使Z均值不为0，β初始值1，γ初始值0.

### Batch Normalization ---Testing

![image-20230226110917675](D:\人工智能\photo\pytorch md\image-20230226110917675.png)

### How does Batch Normalization Help Optimization？-----Internal Covariate Shift？

![image-20230226111349311](D:\人工智能\photo\pytorch md\image-20230226111349311.png)

![image-20230226111523076](D:\人工智能\photo\pytorch md\image-20230226111523076.png)

![image-20230226111542936](D:\人工智能\photo\pytorch md\image-20230226111542936.png)

![image-20230226120807141](D:\人工智能\photo\pytorch md\image-20230226120807141.png)

















































# [PyTorch](https://so.csdn.net/so/search?q=PyTorch&spm=1001.2101.3001.7020)数据集归一化- torchvision.transforms.Normalize()





Pytorch数据归一化

### 图像处理为什么要归一化？

对于网络模型训练等，是为了加速神经网络训练收敛，以及保证程序运行时收敛加快。

数据归一化的概念是一个通用概念，指的是将数据集的原始值转换为新值的行为。新值通常是相对于数据集本身进行编码的，并以某种方式进行缩放。

**特征缩放**

出于这个原因，有时数据归一化的另一个名称是特征缩放。这个术语指的是，在对数据进行归一化时，我们经常会将给定数据集的不同特征转化为相近的范围。

在这种情况下，我们不仅仅是考虑一个值的数据集，还要**考虑一个具有多个特征的元素的数据集，及每个特征的值**。

举例来说，假设我们要处理的是一个人的数据集，我们的数据集中有两个相关的特征，年龄和体重。在这种情况下，我们可以观察到，这两个特征集的大小或尺度是不同的，即体重平均大于年龄。

在使用机器学习算法进行比较或计算时，这种幅度上的差异可能是个问题。因此，这可能是我们希望通过特征缩放将这些特征的值缩放到一些相近尺度的原因之一。
**规范化示例**
当我们对数据集进行归一化时，我们通常会对相对于数据集的每个特定值进行某种形式的信息编码，然后重新缩放数据。考虑下面这个例子：

假设我们有一个正数集合 S 。现在，假设我们从集合s 随机选择一个 x 值并思考：这个 x 值是集合s中最大的数嘛 ？
在这种情况下，答案是我们不知道。我们只是没有足够的信息来回答问题。
但是，现在假设我们被告知 集合 S 通过将每个值除以集合内的最大值进行归一化。通过此标准化过程，已对值最大的信息进行了编码，并对数据进行了重新缩放。
集合中最大的成员是 1，并且数据已按比例缩放到间隔 [0,1]。

## Transformer

Seq2Seq

![image-20230226114520352](D:\人工智能\photo\pytorch md\image-20230226114520352.png)



![image-20230226114631545](D:\人工智能\photo\pytorch md\image-20230226114631545.png)

![image-20230226114703556](D:\人工智能\photo\pytorch md\image-20230226114703556.png)

![image-20230226115110295](D:\人工智能\photo\pytorch md\image-20230226115110295.png)

![image-20230226115738607](D:\人工智能\photo\pytorch md\image-20230226115738607.png)

###Seq2seq

  ![image-20230226115959555](D:\人工智能\photo\pytorch md\image-20230226115959555.png)

### Encoder

![image-20230226120051473](D:\人工智能\photo\pytorch md\image-20230226120051473.png)

![image-20230226120145462](D:\人工智能\photo\pytorch md\image-20230226120145462.png)

![image-20230226121013255](D:\人工智能\photo\pytorch md\image-20230226121013255.png)

![image-20230226121054634](D:\人工智能\photo\pytorch md\image-20230226121054634.png)

### Decoder

Decoder:把Encoder产生的输出都读进去，

BEGIN（special token）：Decoder开始符号，

#### Autoregressive(AT)

![image-20230227142225440](D:\人工智能\photo\pytorch md\image-20230227142225440.png)

![image-20230227142303877](D:\人工智能\photo\pytorch md\image-20230227142303877.png)

Decoder看见的输入其实是前一个时间点自己的输出

![image-20230227142315802](D:\人工智能\photo\pytorch md\image-20230227142315802.png)

![image-20230227142519206](D:\人工智能\photo\pytorch md\image-20230227142519206.png)

![image-20230227142537156](D:\人工智能\photo\pytorch md\image-20230227142537156.png)

![image-20230227142630739](D:\人工智能\photo\pytorch md\image-20230227142630739.png)

![image-20230227142737083](D:\人工智能\photo\pytorch md\image-20230227142737083.png)

**Masked**：产生b~i~时候，不能看比i大的信息

![image-20230227142843045](D:\人工智能\photo\pytorch md\image-20230227142843045.png)

![image-20230227142906032](D:\人工智能\photo\pytorch md\image-20230227142906032.png)

**Why masked?** Consider how does decoder work.

![image-20230227143253075](D:\人工智能\photo\pytorch md\image-20230227143253075.png)

**Adding "Stop Token"**

![image-20230227143441165](D:\人工智能\photo\pytorch md\image-20230227143441165.png)

![image-20230227143456890](D:\人工智能\photo\pytorch md\image-20230227143456890.png)

#### NAT Non-autoregressive

一次把整个句子产生出来

![image-20230227143706330](D:\人工智能\photo\pytorch md\image-20230227143706330.png)

![image-20230227144034589](D:\人工智能\photo\pytorch md\image-20230227144034589.png)

### Encoder 2 Decoder

![image-20230227144248632](D:\人工智能\photo\pytorch md\image-20230227144248632.png)

#### Cross attention

![image-20230227144501296](D:\人工智能\photo\pytorch md\image-20230227144501296.png)

![image-20230227144542973](D:\人工智能\photo\pytorch md\image-20230227144542973.png)

![image-20230227144840761](D:\人工智能\photo\pytorch md\image-20230227144840761.png)

![image-20230227145007530](D:\人工智能\photo\pytorch md\image-20230227145007530.png)



### Training

![image-20230227145101111](D:\人工智能\photo\pytorch md\image-20230227145101111.png)

![image-20230227145328425](D:\人工智能\photo\pytorch md\image-20230227145328425.png)

![image-20230227145525543](D:\人工智能\photo\pytorch md\image-20230227145525543.png)

Decoder输入的时候，给Decoder输入正确的答案------**Teacher Forcing**：using the ground truth as input.

#### Tips

#####Copy Mechanism

一些情况下不需要decoder创造输出出来，可能需要从输入中复制一些出来；例如聊天机器人、摘要提取

![image-20230227145849416](D:\人工智能\photo\pytorch md\image-20230227145849416.png)

![image-20230227145913631](D:\人工智能\photo\pytorch md\image-20230227145913631.png)



######**Pointer Network**

##### Guided Attention

强迫将输入的每个东西都学习

![image-20230227150907748](D:\人工智能\photo\pytorch md\image-20230227150907748.png)

##### Beam Search

![image-20230227151307970](D:\人工智能\photo\pytorch md\image-20230227151307970.png)

![image-20230227151710565](D:\人工智能\photo\pytorch md\image-20230227151710565.png)

Beam search并不是都是结果好的，要根据任务决定，如果任务目的非常明确（语音辨识）Beam search会很有帮助，若需要一些创造（可能会有不止一个答案）随机性可能会更好。

TTS：语音合成

#### Blue score

![image-20230227152337680](D:\人工智能\photo\pytorch md\image-20230227152337680.png)

#### Exposure bias

![image-20230227152453643](D:\人工智能\photo\pytorch md\image-20230227152453643.png)

training是Decoder输入是正确的，但是测试时Decoder输入会有错误，为避免在Ground Truth加入一些错误。

##### Scheduled Sampling

![image-20230227152656466](D:\人工智能\photo\pytorch md\image-20230227152656466.png)

## Generation

## Reinforcement Learning





# 李沐动手深度学习

## Resnet 残差网络

为了提到模型预测的精度，想要提高模型的复杂度如下图左所示，但是学习产生模型偏差。Resnet设计每次更复杂的模型使包含上次模型。

![image-20230825155356180](D:\人工智能\image-20230825155356180.png)

![image-20230825155510200](D:\人工智能\image-20230825155510200.png)

复杂模型包含小模型。

![image-20230825155754620](D:\人工智能\image-20230825155754620.png)

![image-20230825155841487](D:\人工智能\image-20230825155841487.png)

![image-20230825160140398](D:\人工智能\image-20230825160140398.png)