import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Reshape,BatchNormalization,Dropout
from tensorflow.python.keras import backend as K
import numpy as np

#from keras import backend as K
import tensorflow.contrib.keras as K

class LivenessNet:
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        model.add(Conv2D(16, (3, 3), padding="same",input_shape=inputShape))
        	
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(16, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))    
        model.add(Dropout(0.25))
       # model.add(Flatten())
        a,b,c,d = model.output_shape
        a = b*c*d

        model.add(K.layers.Permute([1, 2, 3]))  # Indicate NHWC data layout
        model.add(K.layers.Reshape((a,)))
       
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
 
		# softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
 
		# return the constructed network architecture
        return model
    
            
            
