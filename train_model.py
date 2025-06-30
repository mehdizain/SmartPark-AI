import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

path="/content/drive/MyDrive/AI/ParkingAI/dataset"
categories=['empty','not_empty']

features=[]
labels=[]

for category_index,category in enumerate(categories):
  folder=os.path.join(path,category)
  for file in os.listdir(folder):
    img_path=os.path.join(folder,file)
    img=imread(img_path)
    img=resize(img,(64,64))
    gray=rgb2gray(img)
    hog_feature=hog(gray)
    features.append(hog_feature)
    labels.append(category_index)

X=np.array(features)
y=np.array(labels)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y)

model=SVC(C=10,gamma=0.001)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

pickle.dump(model,open('/content/drive/MyDrive/AI/ParkingAI/model_vehicle.pkl','wb'))
