import pandas as pd
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_rows', 10)

#数据加载
data=pd.read_csv('UCI_Credit_Card.csv')
print(data)
print(data.shape)
print(data.describe())

next_month=data['default.payment.next.month'].value_counts()
print(next_month)
df=pd.DataFrame({'default.payment.next.month':next_month.index,'values':next_month.values})
print(df)
#可视化一下这个df
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.title("Credit Default")
# sns.barplot(x='default.payment.next.month',y='values',data=df)
# plt.show()

#把data里面那个ID给去掉吧
data.drop(['ID'],inplace=True,axis=1)
target=data['default.payment.next.month'].values
columns=data.columns.tolist()
print(columns)
columns.remove('default.payment.next.month')
features=data[columns].values

#数据划分
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(features,target,test_size=0.3)
#构造分类器
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

#构建各种分类器
classifiers=[
    SVC(),DecisionTreeClassifier(),RandomForestClassifier(),KNeighborsClassifier()
]
#分类器的名称
classifier_names=[
    'svc','decisiontreeclassifier','randomforestclassifier','kneighborsclassifier'
]

classifier_param_grid = [
            {'svc__C':[1], 'svc__gamma':[0.01,0.02,0.1]},
            {'decisiontreeclassifier__max_depth':[6,9,11]},
            {'randomforestclassifier__n_estimators':[3,5,6]} ,
            {'kneighborsclassifier__n_neighbors':[4,6,8]},
]
#对具体的分类器进行GridSearchCV参数调优
def GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy'):
    gridsearch=GridSearchCV(estimator=pipeline,param_grid=model_param_grid,scoring=score)
    #寻找最优的参数和最优的准确分数
    search=gridsearch.fit(train_x,train_y)
    print("GridSearchCV的最优参数:{}".format(search.best_params_))
    print("GridSearchCV的最优分数:{}".format(search.best_score_))
    predict_y=gridsearch.predict(test_x)
    print("准确率:{}".format(accuracy_score(test_y,predict_y)))

    responce={}
    responce['predict_y']=predict_y
    responce['accuracy_score']=accuracy_score(test_y,predict_y)
    return responce

for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):
    pipeline = Pipeline([
            ('scaler', StandardScaler()),
            (model_name, model)
    ])
    result = GridSearchCV_work(pipeline, train_x, train_y, test_x, test_y, model_param_grid , score = 'accuracy')
