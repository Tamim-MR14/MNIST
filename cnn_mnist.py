#links
#https://elitedatascience.com/keras-tutorial-deep-learning-in-python
#https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py


#dropout links
# https://github.com/fchollet/keras/issues/3305
# https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning-And-why-is-it-claimed-to-be-an-effective-trick-to-improve-your-network
# inverted dropuout is used in keras which is equivalent to the dropout..



import keras
from keras.datasets import mnist
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# a training set of 60,000 examples, and a test set of 10,000 examples
print (X_train.shape) #(60000, 28, 28)
#plot the first image :D

plt.imshow(X_train[0])

# for tensor flow , channel comes last
# for theano, channel comes second
#If your image batch is of N images
# of HxW size with C channels, theano uses the NCHW ordering while tensorflow uses the NHWC ordering.

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print (X_train.shape, X_test.shape)
# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

num_classes = 10

# difference between conv2d and Convolution2d
# ??
# error explanation
# https://stackoverflow.com/questions/41651628/negative-dimension-size-caused-by-subtracting-3-from-1-for-conv2d

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
#print(model.output_shape)
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit(X_train, Y_train,
          batch_size=32, nb_epoch=2, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=0)

print (score)