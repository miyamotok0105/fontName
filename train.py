# coding: utf-8
import os
import glob
import numpy as np
from PIL import Image
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

path = "./font"
files = os.listdir(path)
files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]

x_train = []
y_train = []
y_list = []

for d in files_dir:
  y_list.append(len(y_list)-1)
  filelist = glob.glob('font/%s/*.png'%(d))
  for f in filelist:
    x_train.append(np.array(Image.open(f).resize((100, 100)))/255.)
    y_train.append(len(y_list)-1)

y_train = np.array(y_train)
x_train = np.array(x_train)
y_train = keras.utils.to_categorical(y_train, num_classes=9)

print(x_train.shape)
print(y_train.shape)


# x = np.array([np.array(Image.open(fname).resize((256, 256))) for fname in filelist])

# # 疑似データ生成
# x_train = np.random.random((100, 100, 100, 3))
# y_train = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)
# x_test = np.random.random((20, 100, 100, 3))
# y_test = keras.utils.to_categorical(np.random.randint(10, size=(20, 1)), num_classes=10)

# print(x_train.shape)
# print(y_train.shape)

model = Sequential()
# 入力: サイズが100x100で3チャンネルをもつ画像 -> (100, 100, 3) のテンソル
# それぞれのlayerで3x3の畳み込み処理を適用している
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=100)
model.save('font_model.h5')
# score = model.evaluate(x_test, y_test, batch_size=32)



# import coremltools
# coreml_model = coremltools.converters.keras.convert(model)
# coreml_model.save("keras_model.mlmodel")





