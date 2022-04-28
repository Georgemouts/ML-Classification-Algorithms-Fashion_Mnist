# TensorFlow and tf.keras imports

import tensorflow
example_model = tensorflow.keras.Sequential()
BatchNormalization = tensorflow.keras.layers.BatchNormalization
Conv2D = tensorflow.keras.layers.Conv2D
MaxPooling2D = tensorflow.keras.layers.MaxPooling2D
Activation = tensorflow.keras.layers.Activation
Flatten = tensorflow.keras.layers.Flatten
Dropout = tensorflow.keras.layers.Dropout
Dense = tensorflow.keras.layers.Dense
K = tensorflow.keras.backend

from sklearn import neighbors,metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#libraries for metrics 

import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


import keras
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical



#------NEAREST NEIGHBOR------

def Nearest_Neighbor():
    fashion_mnist = tensorflow.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = np.reshape(train_images, (60000, 28 * 28))
    test_images = np.reshape(test_images, (10000, 28 * 28))
        
        #clothing categories
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    idx =(train_labels ==0 )| (train_labels ==1) |(train_labels ==2 )|(train_labels ==3) |(train_labels==4)|(train_labels==5) | (train_labels==6) |(train_labels==7) |(train_labels==8) |(train_labels ==9)
    X = train_images[idx]
    Y = train_labels[idx]

    print(len(X),len(Y))
    #Fit classifier
    knn = neighbors.KNeighborsClassifier(n_neighbors= 20).fit(X,Y)
        
        
    idx =(test_labels ==0 )| (test_labels ==1) |(test_labels ==2 )|(test_labels ==3) |(test_labels==4)|(test_labels==5) | (test_labels==6) |(test_labels==7) |(test_labels==8) |(test_labels ==9)
        
    x_test= test_images[idx]
    y_test = test_labels[idx]
    print(len(x_test),len(y_test))
        
    prediction = knn.predict(x_test)
    print(prediction)

    print("-------|Nearest Neighbor RESULTS|-------")
    # accuracy: (tp + tn) / (p + n)
    print("Accuracy =",accuracy_score(y_test,prediction))

    # precision tp / (tp + fp)
    print("Precision =",precision_score(y_test,prediction,average='macro'))

    # recall: tp / (tp + fn)
    print('Recall= ',recall_score(y_test,prediction,average='macro'))

    # f1: 2 tp / (2 tp + fp + fn)
    print('F1 score=',f1_score(y_test,prediction,average='macro'))



#-------------------------
#------SVM------
def SVM():
    fashion_mnist = tensorflow.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    train_images = np.reshape(train_images, (60000, 28 * 28))
    test_images = np.reshape(test_images, (10000, 28 * 28))
        
        #clothing categories
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']   

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    idx =(train_labels ==0 )| (train_labels ==1) |(train_labels ==2 )|(train_labels ==3) |(train_labels==4)|(train_labels==5) | (train_labels==6) |(train_labels==7) |(train_labels==8) |(train_labels ==9)
    x_train = train_images[idx]
    y_train = train_labels[idx]

    # Initialize C-Support Vector classifier
    SVM = SVC(kernel="rbf", C = 1.0,gamma="auto")
    # Fit classifier
    SVM.fit(x_train,y_train)


        
    idx =(test_labels ==0 )| (test_labels ==1) |(test_labels ==2 )|(test_labels ==3) |(test_labels==4)|(test_labels==5) | (test_labels==6) |(test_labels==7) |(test_labels==8) |(test_labels ==9)
        
    x_test= test_images[idx]
    y_test = test_labels[idx]
    print(len(x_test),len(y_test))
        
    prediction = SVM.predict(x_test)
    print(prediction)

    # accuracy: (tp + tn) / (p + n)
    print("-------|SVM RESULTS|-------")
    print("Accuracy =",accuracy_score(y_test,prediction))

    # precision tp / (tp + fp)
    print("Precision =",precision_score(y_test,prediction,average='macro'))

    # recall: tp / (tp + fn)
    print('Recall= ',recall_score(y_test,prediction,average='macro'))

    # f1: 2 tp / (2 tp + fp + fn)
    print('F1 score=',f1_score(y_test,prediction,average='macro'))


#---------------------------
#------Neural Networks------
class NN():
    def load():
        fashion_mnist = tensorflow.keras.datasets.fashion_mnist
        (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
        # reshape dataset to have a single channel
        trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
        testX = testX.reshape((testX.shape[0], 28, 28, 1))
        # one hot encode target values
        trainY = to_categorical(trainY)
        testY = to_categorical(testY)
       
        # convert from integers to floats
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        # normalize to range 0-1
        train_norm = train_norm / 255.0
        test_norm = test_norm / 255.0


    def define_model():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(lr=0.01, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def evaluate_model(dataX, dataY, n_folds=5):
        scores, histories = list(), list()
        # prepare cross validation
        kfold = KFold(n_folds, shuffle=True, random_state=1)
        # enumerate splits
        for train_ix, test_ix in kfold.split(dataX):
            # define model
            model = define_model()
            # select rows for train and test
            trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
            # fit model
            history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
            # evaluate model
            _, acc = model.evaluate(testX, testY, verbose=0)
            print('> %.3f' % (acc * 100.0))
            # append scores
            scores.append(acc)
            histories.append(history)
        return scores, histories

    def summarize_performance(scores):
        # print summary
        print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
        # box and whisker plots of results
        pyplot.boxplot(scores)
        pyplot.show()
        
def mainNN():
    nn =NN()
    trainX, trainY, testX, testY = nn.load()
    scores, histories = nn.evaluate_model(trainX, trainY)
    nn.summarize_performance(scores)


#---------------------------


Nearest_Neighbor()
SVM()
mainNN()
