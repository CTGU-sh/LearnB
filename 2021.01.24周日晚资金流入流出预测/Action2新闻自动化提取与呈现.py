import requests
from bs4 import BeautifulSoup
import jieba
from wordcloud import WordCloud

#目标的url
# url='https://3w.huanqiu.com/a/c36dc8/3xqGPRBcUE6?agt=8'
url='https://m.3dmgame.com/ol/gl/73729.html'
html=requests.get(url,timeout=10)
content=html.content
# print(content)

#通过content创建一个BeautifulSoup对象
soup=BeautifulSoup(content,'html.parser',from_encoding='utf-8')
text=soup.get_text()
print(text)

import jieba.posseg as pseg
#获取人物 地点
words=pseg.lcut(text)
#人物集合
news_person={word for word,flag in words if flag=='nr'}
news_place={word for word,flag in words if flag=='ns'}
print('新闻中的任务有：',news_person)
print('新闻中的地点有：',news_place)

#提取中文和相关的标点符号
import re
text=re.sub('[^[\u4e00-\u9fa5]。，！：、]{3,}','',text)
print(text)


#去掉停用词
def remove_stop_words(f):
    stop_words=['时候','没有']
    for stop_words in stop_words:
        f=f.replace(stop_words,'')
    return f

def create_word_cloud(f):
    f = remove_stop_words(f)
    seg_list=jieba.lcut(f)
    cut_text =' '.join(seg_list)
    wc = WordCloud(
        max_words=100,
        width=2000,
        height=1200,
        font_path='msyh.ttf'
    )
    wordcloud = wc.generate(cut_text)
    wordcloud.to_file("wordcloud.jpg")

create_word_cloud(text)

from textrank4zh import TextRank4Keyword, TextRank4Sentence
# 输出关键词，设置文本小写，窗口为2
tr4w = TextRank4Keyword()
tr4w.analyze(text=text, lower=True, window=3)
print('关键词：')
for item in tr4w.get_keywords(20, word_min_len=2):
    print(item.word, item.weight)

# 输出重要的句子
tr4s = TextRank4Sentence()
tr4s.analyze(text=text, lower=True, source = 'all_filters')
print('摘要：')
# 重要性较高的三个句子
for item in tr4s.get_key_sentences(num=3):
	# index是语句在文本中位置，weight表示权重
    print(item.index, item.weight, item.sentence)