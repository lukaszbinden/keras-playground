import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D



if __name__ == '__main__':

    # create simple CNN
    model = Sequential([
        Dense(16, activation='relu', input_shape=(20,20,3)),
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid'),
        Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'),
        Flatten(),
        Dense(2, activation='softmax')
        ])

    model.summary()
