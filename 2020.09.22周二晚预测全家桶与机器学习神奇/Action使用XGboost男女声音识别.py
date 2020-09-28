import pandas as pd
import numpy as np
import xgboost as xgb

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 10000)
#1读取数据与数据探索，
data=pd.read_csv("voice.csv")
print("数据的维度:{}".format(data.shape))
# print("数据如下:\n{}".format(data.head(10)))
# print("数据描述:\n{}".format(data.describe()))
# print("数据信息:")
# data.info()
#经过上面4中数据的探索，发现数据比较完整，除了标签lable之外，其余的都是float类型，大部分在0-1之间。

#将lable转化为数字，1代表男，0代表女
from sklearn import preprocessing
mylabeEncoder=preprocessing.LabelEncoder()
data["label"]=mylabeEncoder.fit_transform(data["label"])
# print(data)  经过这一步之后，dataFrame里面的标签就male和female就已经变成了数字1和0了

#3划分数据集
from sklearn.model_selection import train_test_split
# 先筛选特征值和目标值
x=data.iloc[:,1:-1]#data里面所有的行，第一列到倒数第二列
y=data["label"]#data里面class那一列
x_train,x_test,y_train,y_test=train_test_split(x,y)#这样就划分好了数据集

#4数据集标准化
from sklearn.preprocessing import StandardScaler
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)
#5模型训练
estimator=xgb.XGBClassifier()
estimator.fit(x_train,y_train)
print(estimator)

# #6模型评估
print("先用socre的方法看看模型的分数：{}".format(estimator.score(x_test, y_test)))
#使用accuracy_score评估
from sklearn.metrics import accuracy_score
myscore=accuracy_score(y_test,estimator.predict(x_test))
print("accuracy_score:{}".format(myscore))
#做一下网格验证
from sklearn.model_selection import GridSearchCV
# #参数准备
# myparam_dict={"max_depth":[3,4,5,6,7,8,9,10],
#               "n_estimators":[i for i in range(100,10001)],
#               "min_child_weight":[i for i in range(301)],
#               "subsample":[i/(10) for i in range(2,12,1)],
#                # "colsample_bytree":[i/(10) for i in range(2,12,1)]
#               }
# print(myparam_dict)
# estimator2=GridSearchCV(estimator,param_grid=myparam_dict,cv=10)
# estimator2.fit(x_train,y_train)
# print("最佳预估器:{}".format(estimator2.best_estimator_))
# print("最佳分数:{}".format(estimator2.best_score_))
# print("最佳参数:{}".format(estimator2.best_params_))
#做这个网格验证电脑卡死了，先中止吧！