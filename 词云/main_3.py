"""
Author : 张程东
File : main_3.py 
"""
import jieba
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from wordcloud import WordCloud

#使用蒙版进行图云设计

#加载文本数据
text=open("xyj.txt",encoding='utf-8').read()
#数据加载
data=" ".join(jieba.cut(text))

#生成对象
mask=np.array(Image.open("black_mask.png"))
wc=WordCloud(mask=mask,font_path="Hiragino.ttf",mode='RGBA',background_color=None).generate(text)

#显示词云
plt.imshow(wc)
plt.axis("off")
plt.show()

wc.to_file("./img/mengceng.png")


