# -*- coding: utf-8 -*-
"""
@author: QAISAR
"""

import numpy as np
import pandas as pd

df=pd.read_csv('message_spam.csv',encoding='latin-1')
df.head()

df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)

x=df.v2
y=df.v1

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

le=LabelEncoder()
y=le.fit_transform(y)
y= y.reshape(-1,1)


x,y=shuffle(x,y,random_state=20)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=2)


from keras.models import Model
from keras.layers import LSTM, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV


maximum_word=1000
maximum_length=150


token=Tokenizer(num_words=maximum_word)
token.fit_on_texts(x_train)
sequences=token.texts_to_sequences(x_train)
sequence_matrix=sequence.pad_sequences(sequences,maxlen=maximum_length)

print(sequences)

sequence_matrix.shape

def create_model():
    
    inputs = Input(name='inputs',shape=[maximum_length])
    layer = Embedding(maximum_word,50,input_length=maximum_length)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(units=256,activation='relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(units=1,activation='sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#    model.summary()

    return model

model=KerasClassifier(build_fn=create_model)
#tuning batch size and epochs
#others like optimizer algo, learning rate, momentum, no of lstm_units can be tuned
epochs=[2,4,6]
batch_size=[32,64,128]
param_grid=dict(epochs=epochs,batch_size=batch_size)

## fold cross validation
grid=GridSearchCV(estimator=model,param_grid=param_grid,scoring='accuracy')

grid.fit(sequence_matrix,y_train)

print(grid.best_score_,grid.best_params_)#returns the best cross validation score and parameters
# gave the result 0.9844460397224216 {'batch_size': 128, 'epochs': 4}



#train the model with the best epoch and and batch size

inputs = Input(name='inputs',shape=[maximum_length])
layer = Embedding(maximum_word,50,input_length=maximum_length)(inputs)
layer = LSTM(64)(layer)
layer = Dense(units=256,activation='relu')(layer)
layer = Dropout(0.5)(layer)
layer = Dense(units=1,activation='sigmoid')(layer)
model = Model(inputs=inputs,outputs=layer)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#run the model with batch_size:128 and epochs 4
model.fit(sequence_matrix,y_train,batch_size=128,epochs=4)

#test performance on test set
test_sequences = token.texts_to_sequences(x_test)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=maximum_length)

loss_and_accuracy=model.evaluate(test_sequences_matrix,y_test)

print(loss_and_accuracy)
