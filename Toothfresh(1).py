#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import colorsys
import numpy as np
import cv2

cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
ret,frame = cap.read() # return a single frame in variable `frame`

while(True):
    cv2.imshow('img1',frame) #display the captured image
    if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y' 
        cv2.imwrite('Desktop/praveen/c1.png',frame)
        cv2.destroyAllWindows()
        break
cap.release()
def crop(image_path, coords,saved_location):
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(coords)
        cropped_image.save(saved_location)
        return cropped_image
def comparenew(x,y,z):
    if x in range(190,197):
        if y in range(27,37):
            print("C4")
        if y in range(37,48):
            print("A4")
    if x in range(197,204):
        if y in range(20,29):
            print("A2")
        if y in range(29,38):
            print("D3")
    if x in range(204,208):
        if y in range(31,39):
            print("C3")
        if y in range(39,46):
            print("B4")
    if x in range(208,211):
        if y in range(28,32):
            print("C2")
        if y in range(32,36):
            print("A3")
        if y in range(36,41):
            print("B3")
    if x in range(211,213):
        print("D4")
    if x in range(213,216):
        print("B2")
    if x in range(216,220):
        if y in range(24,30):
            print("D2")
        if y in range(30,35):
            print("A3.5")
    if x in range(220,224):
        if y in range(22,26):
            print("A1")
        if y in range(26,32):
            print("C1")
    if x in range(224,230):
        print("B1")        
img=crop('Desktop/Praveen/c1.png',(20,30,60,90),'Desktop/Praveen/click.png')
#a,b=img.sizee
#print(a,b)
img.show()
class dominantcolors:
    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    def __init__(self, image, clusters=3):
        self.CLUSTERS = clusters
        self.IMAGE = image
    def dominantColors(self):
        img = cv2.imread(self.IMAGE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        self.IMAGE = img
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        self.COLORS = kmeans.cluster_centers_
        self.LABELS = kmeans.labels_
        return self.COLORS.astype(int)
img = 'Desktop/Praveen/click.png'
clusters =17
dc = dominantcolors(img, clusters)
colors = dc.dominantColors()
x,y,z=tuple(np.sum(colors/17,axis=0))
X,Y,Z = colorsys.rgb_to_yiq(x,y,z)
print(X,Y,Z)
comparenew(round(X),round(Y),round(Z))


# In[ ]:




