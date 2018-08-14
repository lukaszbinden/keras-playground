import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D



if __name__ == '__main__':

    # create simple CNN
    model = Sequential([
        Dense(16, activation='relu', input_shape=(20, 20, 1)),
        Dense(32, activation='relu'),
        Dense(2, activation='softmax')
        ])

    model.summary()
