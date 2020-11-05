"""
Author : 张程东
File : main_4.py 
"""
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import jieba
from wordcloud import WordCloud,ImageColorGenerator

#读取文件
text=open("xyj.txt",encoding='utf-8').read()

#实现分词
data=" ".join(jieba.cut(text))

#加载蒙层
mask=np.array(Image.open("color_mask.png"))
wc=WordCloud(mask=mask,font_path="Hiragino.ttf",background_color=None,mode='RGBA').generate(data)

#提取颜色
image_colors=ImageColorGenerator(mask)
wc.recolor(color_func=image_colors)

plt.imshow(wc)
plt.axis("off")
plt.show()
wc.to_file("./img/mengban2.png")