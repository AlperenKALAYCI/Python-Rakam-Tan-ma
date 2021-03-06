import tensorflow as tf
import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i] #olay tanımlandı.
print(events)

img = np.zeros((512,512,3), np.uint8)  #512 satır ve 512 sutundan olusan dizi olusturup diziyi float64 tipinden uint8 tipine cevirdi.

drawing = False #eger mouse dogru basildiysa
ix,iy= -1,-1

def draw_circle(event,x,y,flags,param):
     global ix, iy, drawing, mode

     if event == cv2.EVENT_LBUTTONDOWN:
         drawing = True
         ix, iy = x, y

     elif event == cv2.EVENT_MOUSEMOVE:

         if drawing == True:
             cv2.circle(img, (x,y), 3, (255,255,255), 20)

     elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


cv2.namedWindow('Ciziniz',cv2.WINDOW_NORMAL)
cv2.resizeWindow("Ciziniz",600,600)
cv2.setMouseCallback('Ciziniz',draw_circle),

while(1):
    cv2.imshow('Ciziniz',img)
    if cv2.waitKey(20) & 0xFF == 13: #son sekiz bit hariC diğer bitler 0 olacaktır. Son sekiz bit ise aynen yeni degere aktarılacaktır.
        break

img=cv2.resize(img,(28,28))
img=[[[i[0]*255*255+i[1]*255+i[2]] for i in part] for part in img]
img = np.array(img)
print(img.shape)
img = np.divide(img, np.max(img)) #img dizisindeki değerleri  img dizisinin max değerine böler ve sonucları img dizisine döndürür.
print(img.max())
img=np.around(img, decimals=2) #img dizisndeki küsüratlı sayıları yuvarlar.Virgülden sonra 2 basamak olacak şekilde
resim = np.multiply(img,255.) # img dizisindeki degerleri 255 ile d carpar ve sonucları yeni bir dizide döndürür.
resim=resim.reshape(28,28) #resim degiskeni yeniden boyutlandırıldı.

img=np.expand_dims(img,axis=0) #img dizisinin genisletilmis boyutları elde edilip img değiskenine atandı.
print(img.shape)

#egitilen model dosyası ve model agırlık dosyası yüklendi.
model=tf.keras.models.load_model("./model/model.h5")
model.load_weights("./model/model_weights.h5")
try:
    res=model.predict(x=img)
except Exception as e:
    print(e)

labels=[0,1,2,3,4,5,6,7,8,9] #tanınacak rakam dizisi tanımlandı.
res=np.around(res,decimals=2)
print(res)
print(np.argmax(res))
print(labels[np.argmax(res)])

cv2.destroyAllWindows() #cizim penceresi kapandı.