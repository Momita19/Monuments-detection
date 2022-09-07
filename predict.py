import cv2
import numpy as np
import joblib

img1 = cv2.imread('C:/Users/User/Desktop/Momita/SIH/Monument-Recognition/Dataset/Predict/a.JPG')
# load the model from disk
filename = 'Monument-recognition.sav'
model = joblib.load(open(filename,"rb"))
#result = model.score(X_test, Y_test)
print(model.history['acc'])


print(model.history.keys())