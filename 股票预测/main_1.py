import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
#获取数据
data=pd.read_csv("./SH600519.csv")
print(data.shape)
#设置训练集，设置测试集
data_trains=data.iloc[:data.shape[0]-300,2:3].values
data_tests=data.iloc[data.shape[0]-300:,2:3].values
#数据归一化
sc=MinMaxScaler(feature_range=(0,1))
data_train=sc.fit_transform(data_trains)
data_test=sc.transform(data_tests)

#创建时间集
x_train=[]
y_train=[]
x_test=[]
y_test=[]
for i in range(60,data_train.shape[0]):
    x_train.append(data_train[i-60:i,:])
    y_train.append(data_train[i,:])

for i in range(60,data_test.shape[0]):
    x_test.append(data_test[i-60:i,:])
    y_test.append(data_test[i,:])
#数据类型转换
x_train=np.array(x_train)
y_train=np.array(y_train)

x_test=np.array(x_test)
y_test=np.array(y_test)

#打乱数据集
np.random.seed(7)
np.random.shuffle(x_train)
np.random.seed(7)
np.random.shuffle(y_train)

#预测数据进行格式转换
x_train=np.reshape(x_train,(x_train.shape[0],60,1))
x_test=np.reshape(x_test,(x_test.shape[0],60,1))


#创建数据类型
model=tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(60,return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.SimpleRNN(60),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

ckpt_path='./checkpoint/mode.ckpt'
if os.path.exists(ckpt_path+".index"):
    model.load_weights(ckpt_path)

call_backs=tf.keras.callbacks.ModelCheckpoint(
    filepath=ckpt_path,
    save_weights_only=True,
    save_best_only=True
)


model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_squared_error'
)

history=model.fit(x_train,y_train,epochs=50,batch_size=64,validation_data=(x_test,y_test),validation_freq=1,callbacks=[call_backs])
pre_data=model.predict(x_test)

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False
p=plt.figure()
p.add_subplot(2,1,1)
plt.plot(history.history['loss'],color='red')
plt.plot(history.history['val_loss'],color='pink')
plt.title("损失函数")
plt.legend(['训练损失函数','测试损失函数'])


p.add_subplot(2,1,2)
plt.plot(y_test,color='green')
plt.plot(pre_data,color='blue')
plt.title("真实值与预测值")
plt.legend(['真实数据','预测数据'])
plt.show()

file = open('./weights.txt', 'w')  # 参数提取
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

