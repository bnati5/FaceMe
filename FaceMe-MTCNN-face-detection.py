#!/usr/bin/env python
# coding: utf-8

# ### Face Detection

# In[6]:


# Import libraries
import time
import pickle
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN 
from tensorflow.keras.models import load_model


# In[25]:


#Load trained model
model = load_model('data/best-facemask-inceptionV3-model.h5')


# In[18]:


#load labels
with open('data/labels.pkl', 'rb') as pf:
    labels = pickle.load(pf)
labels


# In[26]:


# Create detector with default weights
detector = MTCNN(scale_factor= 0.1)

img_size = (150, 150) # set image size
colors = {0: (0, 0, 255), 1: (0, 255, 0), 2: (0, 255, 255)} # set colors

# Video from webcam
cap = cv2.VideoCapture(0)


start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read() 
    
    if not ret:
        break
    
    frame_count += 1     # for fps
    frame = cv2.flip(frame, 1)       # Mirror the image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    for face in faces:
        try:
            x, y, w, h = face['box'] 
            roi =  rgb[y : y+h, x : x+w]   #get region of intrest.
            data = cv2.resize(roi, img_size)/ 255.    #resize imge and flatten
            data = data.reshape((1,) + data.shape)    #reshape the data to fit model
            scores = model.predict(data)              #predict
            target = np.argmax(scores, axis=1)[0]
            
            # Draw bounding boxes
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x+w, y+h), color=colors[target], thickness=2)
            text = "{}: {:.2f}".format(labels[target], scores[0][target])
            cv2.putText(frame, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
        except Exception as e:
            print(e)
            print(roi.shape)

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(img=frame, text='FPS : ' + str(round(fps, 2)), org=(10, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=1)

    # Show the frame
    cv2.imshow('Face Mask Detection', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

