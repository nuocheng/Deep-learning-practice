"""
Author : 张程东
File : main_english.py
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text=open("constitution.txt").read()
#生成wordcloud对象
wc=WordCloud().generate(text)

#显示词云
plt.imshow(wc,interpolation='bilinear')
plt.axis('off')
plt.show()

#保存文件
wc.to_file("./img/constitution.png")