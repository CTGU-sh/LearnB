#数据加载
import itertools

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
mpl.rcParams['font.serif'] = ['KaiTi']
mpl.rcParams['font.size'] =8

pd.set_option('display.max_rows', 10)
# pd.set_option('display.max_columns', 10)

data=pd.read_csv('creditcard.csv')
print(data)
print(data.shape)
print(data.describe())

def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix"', cmap = plt.cm.Blues) :
    plt.figure()
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_precision_recall():
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2, color = 'b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('准确率')
    plt.title('准确率-召回率 曲线')
    plt.show();

from sklearn.preprocessing import StandardScaler
#对amount进行数据规范化
data['Amount_Norm']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
y=np.array(data.Class.tolist())
data=data.drop(['Time','Class','Amount'],axis=1)
X=data
print(X)
print(y)

#数据集切分
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.1)
#逻辑回归
clf=LogisticRegression()
clf.fit(train_x,train_y)
predict_y=clf.predict(test_x)
#计算混淆矩阵并显示出来
from sklearn.metrics import confusion_matrix, precision_recall_curve

cm=confusion_matrix(test_y,predict_y)
plot_confusion_matrix(cm,classes=[0,1],title="Confusion matrix",cmap=plt.cm.Blues)

#计算样本的置信分数
y_score=clf.decision_function(test_x)
#计算准确率 召回率 阈值 可视化
precision,recall,thresholds=precision_recall_curve(test_y,y_score)
plot_precision_recall()

#显示模型中特征的重要性
coeffs = clf.coef_
df_co = pd.DataFrame(np.transpose(abs(coeffs)), columns=["coef_"])
# 下标设置为Feature Name
df_co.index = data.columns
df_co.sort_values("coef_", ascending=True, inplace=True)
df_co.coef_.plot(kind="barh")
plt.title("Feature Importance")
plt.show()
