import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def get_page_content(request_url):
    header = {'user-agent':'Mozilla/5.0(Windows NT 10.0;WOW64)AppleWebKit/537.36(KHTML,like Gecko)Chrome/74.0.3729.131 Safari/537.36'}
    html=requests.get(request_url,headers=header,timeout=10)
    content=html.text
    #print(content)
    #通过content创建BS对象,html.parser是BS自带的HTML解析器
    soup=BeautifulSoup(content,'html.parser',from_encoding='utf-8')
    return soup

request_url='https://ditie.mapbar.com/beijing_line/'
soup=get_page_content(request_url)
subways=soup.find_all('div',class_='station')
df=pd.DataFrame(columns=['name','site'])
for subway in subways:
    #得到线路名称
    route_name=subway.find('strong',class_='bolder').text
    #print('route_name=',route_name)
    #找到该线路中，每一站的名称
    routes=subway.find('ul')
    routes=routes.find_all('a')
    for route in routes:
        #name 地铁站名 site线路名
        temp={'name':route.text,'site':route_name}
        df=df.append(temp,ignore_index=True)
        # print(route.text)
df['city']='北京'
# df.to_excel('./subway.xlsx',index=False)

#添加经度longtitude，纬度latitude
def get_location(keyword, city):
    header = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36'}
    request_url = 'http://restapi.amap.com/v3/place/text?key=1396f4626d2408b9e2d42f89ba616b76&keywords=' + \
                  keyword + '&types=&city=' + \
                  city + '&children=1&offset=1&page=1&extensions=all'
    data = requests.get(request_url, headers=header)
    data.encoding = 'utf-8'
    data = data.text
    #print(data)
    pattern = 'location":"(.*?),(.*?)"'
    result = re.findall(pattern, data)#获取经纬度
    #     print(result)
    #     print(result[0][0], result[0][1])
    # 因为石门站有问题,可能高德地图没有石门站
    try:
        return result[0][0], result[0][1]
    except:
        return get_location(keyword.replace('站', ''), city)

df['longitude'], df['latitude'] = None, None
for index,row in df.iterrows():
#     print(row['name'],row['city'])
    longitude, latitude = get_location(row['name'],row['city'])
    df.iloc[index]['longitude'] = longitude
    df.iloc[index]['latitude'] = latitude
df.to_csv('./subway.csv',index=False)