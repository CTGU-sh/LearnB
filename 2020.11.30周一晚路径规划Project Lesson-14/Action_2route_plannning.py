import requests
import pandas as pd
import re

#数据加载
data=pd.read_csv('./subway.csv')
print(data)

#保存图中两点之间的距离
from collections import defaultdict
graph=defaultdict(dict)

def compute_distance(longitude1, latitude1, longitude2, latitude2):
    header={'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    request_url = 'http://restapi.amap.com/v3/distance?key=1396f4626d2408b9e2d42f89ba616b76&origins='+ \
    str(longitude1)+','+str(latitude1)+'&destination='+ \
    str(longitude2) +','+str(latitude2)+'&type=1'
    data = requests.get(request_url, headers=header)
    data.encoding = 'utf-8'
    data = data.text
#     print(data)
    pattern = 'distance":"(.*?)","duration":"(.*?)"' # 注意是.*?不是.?*
    result = re.findall(pattern, data)
#     print(result)
    return result[0][0]
print(compute_distance(116.337581,39.993138,116.339941,39.976228))

# 保存途中两点之间的距离
for i in range(data.shape[0]):
    site1 = data.iloc[i]['site']
    if i < data.shape[0]-1:
        site2 = data.iloc[i+1]['site']
        # 如果是同一条线路
        if site1 == site2:
            longitude1, latitude1 = data.iloc[i]['longitude'], data.iloc[i]['latitude']
            longitude2, latitude2 = data.iloc[i+1]['longitude'], data.iloc[i+1]['latitude']
            name1 = data.iloc[i]['name']
            name2 = data.iloc[i+1]['name']
            distance = compute_distance(longitude1, latitude1,longitude2, latitude2)
            graph[name1][name2] = distance
            graph[name2][name1] = distance
            print(name1, name2, distance)

import pickle
output=open('graph.pkl','wb')
pickle.dump(graph,output)
