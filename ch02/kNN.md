### 准备：使用Python导入数据
新建一个文件kNN.py，在文件中增加下面的代码：

```python
from numpy import *
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels
    
```

第1行导入numpy，用于科学计算

第2行导入operator，其作用在下一节体现

第4~8行定义createDataSet函数

第5行从内置的list类型参数创建了一个numpy.array类型的对象，
对Python不熟悉的初学者容易把这一行写错，注意圆括号里的内容是
```
[
	[1.0, 1.1],
	[1.0, 1.0], 
	[0.0, 0.1]
]
```
这是一个list，用方括号定义，这个list中的每个元素又是list类型的对象

第8行是空行，不可缺少

### 实施kNN分类算法

在kNN.py文件中增加以下内容：
```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5
    sortedDistIndicis = distances.argsort()
    classCount = {}
    for i in range(k):
        voteILabel = labels[sortedDistIndicis[i]]
        classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1),\
                                reverse=True)
    return sortedClassCount[0][0]

```

这个代码片段定义了classify0函数，用于对输入参数inX进行分类，

classify0函数的四个参数：
inX是个一维数组，里面的数据是待分类对象的各个属性值，
dataSet是个二维数组，里面的数据是所有已有对象的各个属性值，
inX的数组长度应该和dataSet的列数（shape[1]）相同，
labels是个list，里面的数据是所有已有对象的标签，
labels的长度应该和dataSet的行数（shape[0]）相同，
k是int类型的数，代表需要寻找的最邻近的对象的个数

第3行中的tile，就像贴瓷砖一样，把inX沿两个方向平铺，
平铺后的array和dataSet的维度一致，
平铺后的array和dataSet做差，得到diffMat，
diffMat就是待分类对象和每个已有对象的差异

第4~6行从差异中计算距离，
注意第5行的array.sum函数，参数axis = 1，代表二维数组行内相加，
得到的结果长度与dataSet行数相同，
如果axis = 0，则是二维数组行与行相加，得到的结果长度与dataSet列数相同

第7行对距离进行排序，
得到的sortedDistIndicis是排序后的距离在原来distances中的序号，
例如如果distances是[8, 7, 3, 5]，则sortedDistIndicis为[2, 3, 1, 0]，
最小的数是原来的第2个数，即为3，第二小的数是原来的第3个数，即为5，
第三小的数是原来的第1个数，即为7，最大的数是原来的第0个数，即为8

第8行classCount，一个空的dict对象

第9~11行把距离最近的k个对象的标签找出来，
dict.get(key, myVal)函数，如果dict中有这个key，则返回字典中key对应的value，
如果dict中没有这个key，返回myVal，注意此时字典中并没有多出来一个key，
例如
```python
d = {'a':1, 'b':2, 'c':3}
```
在执行
```python
myVal = d.get('e', 100)
```
后，myVal 为 100，d 还是原来的 {'a':1, 'b':2, 'c':3}
第11行之所以用dict.get函数，而不用dict[key]，
是因为当dict中不存在key时，调用dict[key]会出错

第12行对dict进行排序，注意这里把原书中的iteritems改为了items，
在Python3中iteritems已废弃，
sorted函数的参数key是个函数对象，这里用operator.itemgetter(1),
表示依照每个对象的第1个属性进行排序，
在这里就是依照统计出来的数字排序，而不是依照标签来排序（第0个属性是标签）

第14行返回最高票的标签
