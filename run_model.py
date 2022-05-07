
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os


model = tf.keras.models.load_model('digits.model')
images = [item for item in os.listdir(path="drawn numbers")]


for image in images:
    img = cv.imread(f"drawn numbers/{image}")[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f"the result is probably: {np.argmax(prediction)}")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()