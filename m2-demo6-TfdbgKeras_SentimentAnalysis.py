# Original code at:
# https://medium.com/chakki/debug-keras-models-with-tensorflow-debugger-2b68b8e38370

from __future__ import absolute_import, division, print_function

import keras.backend as K

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb

from tensorflow.python import debug as tf_debug

sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess) #Uncomment to enable debugging

K.set_session(sess)

max_features = 20000 #most frequent words in the dataset
maxlen = 80 #max 80 words
batch_size = 32
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(x_train, y_train), (x_test, y_test)  = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old


x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

model = Sequential()

model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=15, validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
