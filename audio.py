from google.colab import files
uploaded = files.upload()

import zipfile
with zipfile.ZipFile("codecrafters.zip",'r') as zip_ref:
  zip_ref.extractall()

!pip install librosa numpy scikit-learn tensorflow matplotlib --quiet

import os
print(os.listdir("codecrafters/datasets/audio"))

audio_folders = "codecrafters/datasets/audio"

import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#load audio file and link
def load_audio(dataset_path):
  data=[]
  labels=[]
  for label,folder in enumerate(['real','fake']): #real-1,fake=0
    path=os.path.join(dataset_path,folder)
    for file in os.listdir(path):
      if file.endswith(".wav"):
        signal,sr=librosa.load(os.path.join(path,file),sr=22050)
        data.append(signal)
        labels.append(label)
  return data, labels
data,labels=load_audio(audio_folders)

#extract mfcc features
def extract_features(data):
  features=[]
  for signal in data:
    mfcc=librosa.feature.mfcc(y=signal,sr=22050,n_mfcc=40)
    mfcc_scaled=np.mean(mfcc.T,axis=0)
    features.append(mfcc_scaled)
  return np.array(features)
X=extract_features(data)
y=np.array(labels)

import os
audio_folders="/content/codecrafters/datasets/audio"
for folder in ['real','fake']:
  path=os.path.join(audio_folders,folder)
  print(folder,"->",len(os.listdir(path)),"files")

#split the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#build neural network
model=Sequential()
model.add(Dense(128,input_shape=(X_train.shape[1],),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#train the model
model.fit(X_train,y_train,epochs=50,batch_size=16,validation_data=(X_test,y_test))

#evaluate accuracy
loss,accuracy=model.evaluate(X_test,y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

model.save("audio_classification_model.h5")
