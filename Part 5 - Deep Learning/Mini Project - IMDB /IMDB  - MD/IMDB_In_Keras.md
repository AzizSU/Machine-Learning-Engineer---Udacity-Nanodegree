
# Analyzing IMDB Data in Keras


```python
# Imports
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(42)
```

    Using TensorFlow backend.


## 1. Loading the data
This dataset comes preloaded with Keras, so one simple command will get us training and testing data. There is a parameter for how many words we want to look at. We've set it at 1000, but feel free to experiment.


```python
# Loading the data (it's preloaded in Keras)
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

print(x_train.shape)
print(x_test.shape)
```

    Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz
    17465344/17464789 [==============================] - 0s 0us/step
    (25000,)
    (25000,)


## 2. Examining the data
Notice that the data has been already pre-processed, where all the words have numbers, and the reviews come in as a vector with the words that the review contains. For example, if the word 'the' is the first one in our dictionary, and a review contains the word 'the', then there is a 1 in the corresponding vector.

The output comes as a vector of 1's and 0's, where 1 is a positive sentiment for the review, and 0 is negative.


```python
print(x_train[0])
print(y_train[0])
```

    [1, 11, 2, 11, 4, 2, 745, 2, 299, 2, 590, 2, 2, 37, 47, 27, 2, 2, 2, 19, 6, 2, 15, 2, 2, 17, 2, 723, 2, 2, 757, 46, 4, 232, 2, 39, 107, 2, 11, 4, 2, 198, 24, 4, 2, 133, 4, 107, 7, 98, 413, 2, 2, 11, 35, 781, 8, 169, 4, 2, 5, 259, 334, 2, 8, 4, 2, 10, 10, 17, 16, 2, 46, 34, 101, 612, 7, 84, 18, 49, 282, 167, 2, 2, 122, 24, 2, 8, 177, 4, 392, 531, 19, 259, 15, 934, 40, 507, 39, 2, 260, 77, 8, 162, 2, 121, 4, 65, 304, 273, 13, 70, 2, 2, 8, 15, 745, 2, 5, 27, 322, 2, 2, 2, 70, 30, 2, 88, 17, 6, 2, 2, 29, 100, 30, 2, 50, 21, 18, 148, 15, 26, 2, 12, 152, 157, 10, 10, 21, 19, 2, 46, 50, 5, 4, 2, 112, 828, 6, 2, 4, 162, 2, 2, 517, 6, 2, 7, 4, 2, 2, 4, 351, 232, 385, 125, 6, 2, 39, 2, 5, 29, 69, 2, 2, 6, 162, 2, 2, 232, 256, 34, 718, 2, 2, 8, 6, 226, 762, 7, 2, 2, 5, 517, 2, 6, 2, 7, 4, 351, 232, 37, 9, 2, 8, 123, 2, 2, 2, 188, 2, 857, 11, 4, 86, 22, 121, 29, 2, 2, 10, 10, 2, 61, 514, 11, 14, 22, 9, 2, 2, 14, 575, 208, 159, 2, 16, 2, 5, 187, 15, 58, 29, 93, 6, 2, 7, 395, 62, 30, 2, 493, 37, 26, 66, 2, 29, 299, 4, 172, 243, 7, 217, 11, 4, 2, 2, 22, 4, 2, 2, 13, 70, 243, 7, 2, 19, 2, 11, 15, 236, 2, 136, 121, 29, 5, 2, 26, 112, 2, 180, 34, 2, 2, 5, 320, 4, 162, 2, 568, 319, 4, 2, 2, 2, 269, 8, 401, 56, 19, 2, 16, 142, 334, 88, 146, 243, 7, 11, 2, 2, 150, 11, 4, 2, 2, 10, 10, 2, 828, 4, 206, 170, 33, 6, 52, 2, 225, 55, 117, 180, 58, 11, 14, 22, 48, 50, 16, 101, 329, 12, 62, 30, 35, 2, 2, 22, 2, 11, 4, 2, 2, 35, 735, 18, 118, 204, 881, 15, 291, 10, 10, 2, 82, 93, 52, 361, 7, 4, 162, 2, 2, 5, 4, 785, 2, 49, 7, 4, 172, 2, 7, 665, 26, 303, 343, 11, 23, 4, 2, 11, 192, 2, 11, 4, 2, 9, 44, 84, 24, 2, 54, 36, 66, 144, 11, 68, 205, 118, 602, 55, 729, 174, 8, 23, 4, 2, 10, 10, 2, 11, 4, 2, 127, 316, 2, 37, 16, 2, 19, 12, 150, 138, 426, 2, 2, 79, 49, 542, 162, 2, 2, 84, 11, 4, 392, 555]
    1


## 3. One-hot encoding the output
Here, we'll turn the input vectors into (0,1)-vectors. For example, if the pre-processed vector contains the number 14, then in the processed vector, the 14th entry will be 1.


```python
# One-hot encoding the output into vector mode, each of length 1000
tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
print(x_train[0])
```

    [ 0.  1.  1.  0.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
      1.  1.  0.  1.  1.  1.  1.  0.  1.  1.  0.  1.  1.  0.  0.  1.  1.  1.
      1.  1.  0.  1.  1.  0.  0.  0.  1.  0.  1.  1.  1.  1.  1.  0.  1.  0.
      1.  1.  1.  0.  1.  0.  0.  1.  1.  0.  0.  1.  1.  0.  1.  1.  1.  0.
      0.  0.  0.  0.  0.  1.  0.  1.  0.  0.  1.  0.  1.  0.  1.  0.  1.  0.
      0.  0.  0.  1.  0.  0.  0.  0.  1.  0.  1.  1.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  1.  0.  0.  0.  0.  1.  1.  0.  0.  1.  1.  1.  0.  1.
      0.  1.  0.  0.  0.  0.  0.  1.  0.  0.  1.  0.  1.  0.  0.  0.  1.  0.
      1.  0.  1.  0.  1.  0.  1.  0.  1.  0.  0.  0.  0.  1.  0.  1.  0.  0.
      1.  0.  0.  0.  0.  1.  0.  1.  1.  0.  1.  0.  1.  0.  0.  1.  0.  0.
      1.  0.  0.  0.  0.  0.  0.  1.  1.  0.  0.  0.  1.  0.  0.  0.  0.  0.
      1.  0.  0.  0.  0.  0.  1.  1.  1.  0.  1.  0.  0.  0.  0.  0.  0.  0.
      0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  1.  0.  0.  0.  0.  0.  1.  0.
      0.  0.  1.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  1.  0.  0.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
      0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  1.  1.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  1.  1.  0.  1.  0.
      0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.
      0.  1.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  1.  0.  0.  1.
      0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  1.  0.  0.  1.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.
      0.  0.  0.  1.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  1.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  1.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
      0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]


And we'll also one-hot encode the output.


```python
# One-hot encoding the output
num_classes = 2
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape)
print(y_test.shape)
```

    (25000, 2)
    (25000, 2)


## 4. Building the  model architecture
Build a model here using sequential. Feel free to experiment with different layers and sizes! Also, experiment adding dropout to reduce overfitting.


```python
#Build the model architecture
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=1000))
model.add(Dropout(0.6))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

#Compile the model using a loss function and an optimizer.
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 512)               512512    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 2)                 1026      
    =================================================================
    Total params: 513,538
    Trainable params: 513,538
    Non-trainable params: 0
    _________________________________________________________________


## 5. Training the model
Run the model here. Experiment with different batch_size, and number of epochs!


```python
#Run the model with 10 epochs 
hist = model.fit(x_train, y_train,
          batch_size=32,
          epochs=10,
          validation_data=(x_test, y_test), 
          verbose=2)

```

    Train on 25000 samples, validate on 25000 samples
    Epoch 1/10
     - 16s - loss: 0.4309 - acc: 0.8135 - val_loss: 0.3338 - val_acc: 0.8612
    Epoch 2/10
     - 15s - loss: 0.4738 - acc: 0.8271 - val_loss: 1.6977 - val_acc: 0.6022
    Epoch 3/10
     - 15s - loss: 0.7063 - acc: 0.8123 - val_loss: 1.6169 - val_acc: 0.7068
    Epoch 4/10
     - 15s - loss: 1.0193 - acc: 0.8067 - val_loss: 0.6537 - val_acc: 0.8613
    Epoch 5/10
     - 14s - loss: 1.2825 - acc: 0.8181 - val_loss: 2.3876 - val_acc: 0.7439
    Epoch 6/10
     - 14s - loss: 1.5202 - acc: 0.8255 - val_loss: 4.7938 - val_acc: 0.6000
    Epoch 7/10
     - 15s - loss: 1.6527 - acc: 0.8305 - val_loss: 1.4180 - val_acc: 0.8538
    Epoch 8/10
     - 14s - loss: 1.7646 - acc: 0.8392 - val_loss: 2.0246 - val_acc: 0.8284
    Epoch 9/10
     - 14s - loss: 1.9489 - acc: 0.8360 - val_loss: 1.6787 - val_acc: 0.8579
    Epoch 10/10
     - 15s - loss: 1.9497 - acc: 0.8418 - val_loss: 2.6091 - val_acc: 0.8014



```python
#Run the model with 8 epochs and 20 batchs
hist = model.fit(x_train, y_train,
          batch_size=32,
          epochs=8,
          validation_data=(x_test, y_test), 
          verbose=2)
```

    Train on 25000 samples, validate on 25000 samples
    Epoch 1/8
     - 14s - loss: 2.0425 - acc: 0.8452 - val_loss: 2.1491 - val_acc: 0.8369
    Epoch 2/8
     - 14s - loss: 2.0424 - acc: 0.8484 - val_loss: 1.9574 - val_acc: 0.8534
    Epoch 3/8
     - 14s - loss: 2.1978 - acc: 0.8415 - val_loss: 1.9756 - val_acc: 0.8564
    Epoch 4/8
     - 14s - loss: 2.1762 - acc: 0.8450 - val_loss: 2.0947 - val_acc: 0.8520
    Epoch 5/8
     - 14s - loss: 2.1914 - acc: 0.8472 - val_loss: 2.4111 - val_acc: 0.8318
    Epoch 6/8
     - 14s - loss: 2.2300 - acc: 0.8461 - val_loss: 2.0220 - val_acc: 0.8586
    Epoch 7/8
     - 14s - loss: 2.2164 - acc: 0.8477 - val_loss: 2.0301 - val_acc: 0.8584
    Epoch 8/8
     - 14s - loss: 2.3153 - acc: 0.8423 - val_loss: 2.1691 - val_acc: 0.8512


## 6. Evaluating the model
This will give you the accuracy of the model, as evaluated on the testing set. Can you get something over 85%?


```python
score = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: ", score[1])
```

    Accuracy:  0.8512

