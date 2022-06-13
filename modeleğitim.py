import tensorflow as tf
import numpy as np
import cv2

# tensorflow keras mnist adlı dataseti yüklendi.(el yazısı rakamlar)
(trainx,trainy),(testx,testy)=tf.keras.datasets.mnist.load_data()
trainx=tf.keras.utils.normalize(trainx,axis=1)
testx=tf.keras.utils.normalize(testx,axis=1)

#egitilecek olan model tanımlandı.
model=tf.keras.Sequential()
model.add(tf.keras.layers.Flatten()) # katman icindeki veriler düzlestirildi.
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu)) #128 norondan oluşan işlem katmanı olusturuldu.
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax)) #sonuc katmanı olusturuldu.10 noronluk.

#Model Derlendi.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=[tf.keras.metrics.sparse_categorical_accuracy])
model.fit(x=trainx,y=trainy,epochs=10) #alıştırma islemi 10 asamalı
model.save("./model/model.h5") #modelin kaydedilecegi konum.
model.save_weights("./model/model_weights.h5") #model agırlıgı kaydedildi.

testKayip,testHassasiyet=model.evaluate(x=testx,y=testy) #degerlendirme yapıldı.

print("Test Hassasiyeti:",testHassasiyet) #test hassasiyeti terminale yazıldı.


