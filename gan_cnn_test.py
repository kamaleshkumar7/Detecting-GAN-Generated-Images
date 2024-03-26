import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
import cv2
import numpy as np
import os.path

# load the training data
train_images = []
for i in range(1, 1000):
    img = cv2.imread(f'train/{i:06d}.png', cv2.IMREAD_GRAYSCALE)
    train_images.append(img)
train_images = np.array(train_images)

# calculate the FFTs of the training images
train_ffts = []
for img in train_images:
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))
    train_ffts.append(magnitude_spectrum)
train_ffts = np.array(train_ffts)

# create the training labels
train_labels = np.zeros(len(train_images))
train_labels[500:] = 1

# load the test image
test_img = cv2.imread('real/1.jpg', cv2.IMREAD_GRAYSCALE)

# calculate the FFT of the test image
f = np.fft.fft2(test_img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))
test_fft = np.array([magnitude_spectrum])

# check if saved model exists, if yes, load it, otherwise train the model and save it
if os.path.isfile('gan_detector.h5'):
    model = tf.keras.models.load_model('gan_detector.h5')
else:
    # create the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(train_ffts.shape[1], train_ffts.shape[2], 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(train_ffts.reshape(train_ffts.shape[0], train_ffts.shape[1], train_ffts.shape[2], 1), train_labels, epochs=10, batch_size=32)

    # save the trained model
    model.save('gan_detector.h5')

# predict whether the test image is GAN or not
prediction = model.predict(test_fft.reshape(1, test_fft.shape[1], test_fft.shape[2], 1))
if prediction > 0.5:
    print("GAN image")
else:
    print("Real image")
