# -*- coding: utf-8 -*-
"""
Author : 张程东
File : main_chinease.py 
"""
import os
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import jieba

#打开文本文件
text=open('xyj.txt',encoding='utf-8').read()
# print(text)
#进行分词
data=jieba.cut(text)
txt=" ".join(data)
wc=WordCloud(font_path="Hiragino.ttf",mode='RGBA').generate(txt)
plt.imshow(wc)
plt.show()

