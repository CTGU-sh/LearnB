from surprise import KNNWithMeans
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import KFold
import time
startTime=time.time()
# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('./ratings.csv', reader=reader)
trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
algo = KNNWithMeans(k=50, sim_options={'user_based': False, 'verbose': 'True'})

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)
    # 计算MAE
    accuracy.mae(predictions, verbose=True)
# algo.fit(trainset)

uid = str(196)
iid = str(302)

pred = algo.predict(uid, iid)
print(pred)
endTime=time.time()
print("程序运行的时间:{}".format(endTime-startTime))