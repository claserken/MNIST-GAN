import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

SAMPLE_SIZE = 20000

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

p = np.random.permutation(train_images.shape[0])

train_images = train_images[p][:SAMPLE_SIZE]
train_labels = train_labels[p][:SAMPLE_SIZE]

seedifier = load_model("./seedifierAutoencoderBiggerNormal")

cmap = {
    0: "blue",
    1: "green",
    2: "red",
    3: "darkturquoise",
    4: "purple",
    5: "khaki",
    6: "black",
    7: "darkkhaki",
    8: "gray",
    9: "#747257",
}




latent_codes = seedifier(train_images, training=False).numpy()

X = latent_codes[:, 0]
Y = latent_codes[:, 1]
C = []
size = 9*np.ones(SAMPLE_SIZE)

for label in train_labels:
    C.append(cmap[label])

# for i, (image, color) in enumerate(latent_codes):
#     # print(latent_code)
#     X[i] = latent_code[0][0]
#     Y[i] = latent_code[0][1]
#     C.append(cmap[color.numpy()])

x = np.array([2,3,4])
y = np.array([2,3,4])

plt.scatter(X,Y, s=size, color=C)
print('DONE')
plt.show()