import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]
BUFFER_SIZE = 60000
BATCH_SIZE = 256
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
# train_dataset = tf.data.Dataset.from_tensor_slices(train_images, train_labels)

# for i, batch in enumerate(train_dataset):
#     if i < 1:
#         for j in range(5):
#           print(batch[0][j])
#           print(batch[1][j])
# # print(train_dataset[0])
# exit()
print('starting')
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(100, input_shape=(60,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(7*7*256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256) # Note: None is the batch size
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    return model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(20))
    return model

def one_hot(n, fake=False):
    arr = np.zeros((len(n), 20))
    for i, a in enumerate(n):
        #print(type(i))
        #print(type(a))
        if fake:
            a += 10
        arr[i][a] = 1
    # arr = [0]*20
    # a = n
    # if fake:
    #     a += 10
    # arr[a] = 1
    return arr

# print(one_hot([0, 1, 2]))
# exit()

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def discriminator_loss(real_labels, real_output, fake_labels, fake_output):
    return cross_entropy(one_hot(real_labels), real_output) + cross_entropy(one_hot(fake_labels, fake=True), fake_output)
    # return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_loss(labels, fake_output):
    return cross_entropy(one_hot(labels), fake_output)  

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator = make_generator_model()
discriminator = make_discriminator_model()
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
EPOCHS = 50
noise_dim = 50
num_examples_to_generate = 10
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

extra = [0]*num_examples_to_generate
for i in range(num_examples_to_generate):
    extra[i] = i
additional = one_hot(extra)[:, :10]
seed = tf.concat([additional, seed], 1)
# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
#@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    arr = [0]*BATCH_SIZE
    
    for i in range(BATCH_SIZE):
        arr[i] = (i % 10)
    
    encoded = one_hot(arr)[:, :10]
    noise = tf.concat([encoded, noise], 1)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images[0], training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(labels=arr, fake_output=fake_output)
        disc_loss = discriminator_loss(real_labels=images[1].numpy(), real_output=real_output, fake_labels=arr, fake_output=fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
      plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))

train(train_dataset, EPOCHS)

generator.save("./generator")
discriminator.save("./discriminator")