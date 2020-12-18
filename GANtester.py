import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Desired digit
digit = 2

# Add key value pairs where keys are from 10-59 and values are any reals for randomization (editing keys from 0-9 will result in encoding
# for multiple digits at once which also leads to some interesting results!)
edits = {10: 0, 30:1 }

generator = load_model("./DoubleGANV1/generator1")
#discriminator = load_model("./discriminator")

fig = plt.figure(figsize=(1,1))
seed = [0]*60


seed[digit] = 1

for key, value in edits.items():
	seed[key] = value

predictions = generator(tf.convert_to_tensor([seed]), training=False)

plt.subplot(1,1,1)
plt.imshow(predictions[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
plt.axis('off')
filename = "./digitcombos/digit" + str(digit) + str(edits) + ".jpg"
#plt.savefig(filename.replace(':', ''))
plt.show()
