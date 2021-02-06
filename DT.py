#导入数据
from sklearn.datasets import load_iris
iris = load_iris()

#指定训练集合测试集
test_idx = [0, 50, 100]

import numpy as np
from sklearn import tree
#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

#训练模型
clf = tree.DecisionTreeClassifier(max_depth=3,
                                  criterion='entropy')   # "gini' or 'entropy'
clf.fit(train_data, train_target)

#打印测试结果
print(test_target)
print(clf.predict(test_data))

#可视化,将结果保存至pdf文件
import graphviz
#使用export.graphviz函数，其支持许多参数，包括每个结点的颜色，用来划分
#结点的变量名称和类的名称。如果使用jupyter notebook也可以自动嵌入输出。
dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")