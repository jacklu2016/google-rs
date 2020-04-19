# -*- coding:utf-8 -*-

from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

# define converts(字典)
def country_label(s):
    it = {'United States': 0, 'Japan': 1, 'Italy': 2, 'China': 3, 'South Korea': 4}
    return it[s]


# 1.读取数据集
path = 'owid-covid-data.csv'
data = pd.read_csv(path,  delimiter=',')
# 取2020年4月1日后的数据
data = data[data['location'].isin(['United States', 'Japan', 'Italy', 'China', 'South Korea'])]
data = data.iloc[:,1:6]
data = data[data['date'] >= '2020-04-01']
data['location_code'] = [country_label(location) for location in data['location']]
data = data[['date','total_cases', 'total_deaths', 'location_code']]
print(data)
# 2.划分数据与标签
#x  # x为数据，y为标签
train_data_full = data[data['date'] <= '2020-04-17']
test_data_full = data[data['date'] > '2020-04-17']
train_data = train_data_full.iloc[:, [1, 2]]
train_label = train_data_full.iloc[:, 3]
test_data = test_data_full.iloc[:, [1, 2]]
test_label = test_data_full.iloc[:, 3]
print(test_data.head())
print(test_data.shape)
# print(train_data.shape)

# 3.训练svm分类器
classifier = svm.SVC(C=0.1, kernel='linear', gamma=0.1)  # ovr:一对多策略
classifier.fit(train_data, train_label)
parameters = {'kernel':['linear','rbf','sigmoid','poly'],'C':np.linspace(0.1,20,50),'gamma':np.linspace(0.1,20,20)}
svc = svm.SVC()
#model = GridSearchCV(svc, parameters, cv=5, scoring='accuracy')
#model.fit(train_data, train_label)
#print(model.best_params_)
#print('accuracy: %d' % model.score(test_data, test_label))


# 调用accuracy_score方法计算准确率
tra_label = classifier.predict(train_data)  # 训练集的预测标签
tes_label = classifier.predict(test_data)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))

# 查看决策函数
print('train_decision_function:\n', classifier.decision_function(train_data))  # (90,3)
print('predict_result:\n', classifier.predict(train_data))

# 5.绘制图形
# 确定坐标轴范围
x = np.array(train_data)
y = np.array(train_label)
test_data = np.array(test_data)
test_label = np.array(test_label)
x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# 设置颜色
cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF', '#DC143C', '#9932CC'])
cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])

grid_hat = classifier.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
plt.scatter(x[:, 0], x[:, 1], c=y, s=30, cmap=cm_dark)  # 样本
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label, s=30, edgecolors='k', zorder=2,
            cmap=cm_dark)  # 圈中测试集样本点
plt.xlabel('感染总数', fontsize=13)
plt.ylabel('死亡总数', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title('疫情国家预测')
plt.show()