# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 23:00:44 2019

@author: sanke
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 22:06:50 2019

@author: sanke
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:56:05 2019

@author: sanke
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import os
import re
import matplotlib
import seaborn as sns
df=pd.read_csv('test1.csv')
df=df[['headline','is_sarcastic']]

data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)
data=data[['headline','is_sarcastic']]
df=df.append(data)
df['headline'] = df['headline'].apply(lambda x: x.lower())
df['headline'] = df['headline'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
    
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(df['headline'].values)
X = tokenizer.texts_to_sequences(df['headline'].values)
X = pad_sequences(X)
Y = pd.get_dummies(df['is_sarcastic']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.01, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)
embed_dim = 128
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
batch_size = 32
history = model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size, verbose = 2)
dt=pd.read_csv('test.csv')

yfinal=np.zeros((1975, 2))
yfinal[:,0]=dt.iloc[:,0].values
datest=pd.read_csv('test.csv')
datest=datest[['text']]

datest['text'] = datest['text'].apply(lambda x: x.lower())
datest['text'] = datest['text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
yt=[]
pre=datest.iloc[:,0:1].values
for i in range(0,len(pre)):
  headline=pre[i]
  headline = tokenizer.texts_to_sequences(headline)
  headline = pad_sequences(headline, maxlen=45, dtype='int32', value=0)

  sentiment = model.predict(headline,batch_size=1,verbose = 2)[0]
  if(np.argmax(sentiment) == 0):
    yt.append(0)
  elif (np.argmax(sentiment) == 1):
    yt.append(1)

yfinal[:,1]=yt 
