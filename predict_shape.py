import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from random import randint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

''' This task comes from kaggle.com, you can find there datasets 

source : https://www.kaggle.com/smeschke/four-shapes

'''

PATH = r'C:\Users\adria\OneDrive\Pulpit\kaggle\shapes\\'
IMG_SIZE = 64
shapes = ['circle', 'square', 'triangle', 'star']
labels = []
dataset = []

for shape in shapes:
    print("Getting data for: ", shape)
        
    for path in os.listdir(PATH + shape):
        image = cv2.imread(PATH + shape + '/' + path)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.reshape(12288)
        dataset.append(image)
        labels.append(shapes.index(shape))
        
  
        
index = np.random.randint(0, len(dataset) - 1, size= 20)
plt.figure(figsize=(5,7))

''' Show random shapes '''
for i, ind in enumerate(index, 1):
    img = dataset[ind].reshape((64, 64, 3))
    lab = shapes[labels[ind]]
    plt.subplot(4, 5, i)
    plt.title(lab)
    plt.axis('off')
    plt.imshow(img)
    

'''Converting data to np.array''' 
   
X = np.array(dataset)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

p = Perceptron(shuffle=True, eta0=1.0, max_iter=100)
p.fit(X_train, y_train)
y_predict = p.predict(X_test)

score = p.score(X_test, y_test)
print(score)

bad_results = [(a,b,c) for (a,b,c) in zip(X_test[y_test != y_predict],  
                                         y_test[y_test != y_predict], 
                                         y_predict[y_test != y_predict])]

len(bad_results)

'''Draw bad results'''
i=1
for x_test, y_test, y_pred in bad_results:
    img = x_test.reshape((64, 64, 3))
    label_test = shapes[y_test]
    label_pred = shapes[y_pred]
    plt.figure(figsize=(20,20))
    plt.subplot(len(bad_results), 1, i)
    plt.title(label_test +' - '+ label_pred)
    plt.axis('off')
    plt.imshow(img)
    i+=1
    
    
    
    
'''Classsify random shape'''
idx = randint(0,y_predict.size)
plt.title(shapes[y_predict[idx]])
plt.imshow(X_test[idx].reshape((64,64,3)))