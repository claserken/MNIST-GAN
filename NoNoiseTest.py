import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

generator = load_model("./generator")
discriminator = load_model("./discriminator")

fig = plt.figure(figsize=(4,4))

for i in range(10):
	seed = [0]*60
	seed[i] = 1
	predictions = generator(tf.convert_to_tensor([seed]), training=False)
	plt.subplot(4,4, i+1)
	plt.axis('off')
	plt.imshow(predictions[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
	
plt.show()
