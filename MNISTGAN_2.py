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

def discriminator_loss(real_labels, real_output, fake_labels, fake_output, fake_labels_cross, fake_output_cross):
    return cross_entropy(one_hot(real_labels), real_output) + 0.8*cross_entropy(one_hot(fake_labels, fake=True), fake_output) + 0.2*cross_entropy(one_hot(fake_labels_cross, fake=True), fake_output_cross)
    # return cross_entropy(tf.ones_like(fake_output), fake_output)

def generator_loss(labels, fake_output):
    return cross_entropy(one_hot(labels), fake_output)  

generator_optimizer1 = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer1 = tf.keras.optimizers.Adam(1e-4)
generator_optimizer2 = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer2 = tf.keras.optimizers.Adam(1e-4)
generator1 = make_generator_model()
discriminator1 = make_discriminator_model()
generator2 = make_generator_model()
discriminator2 = make_discriminator_model()

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
def train_step(batch1, batch2):
    noise1 = tf.random.normal([BATCH_SIZE, noise_dim])
    noise2 = tf.random.normal([BATCH_SIZE, noise_dim])

    arr = [0]*BATCH_SIZE
    
    for i in range(BATCH_SIZE):
        arr[i] = (i % 10)
    
    encoded = one_hot(arr)[:, :10]
    noise1 = tf.concat([encoded, noise1], 1)
    noise2 = tf.concat([encoded, noise2], 1)

    with tf.GradientTape() as gen1_tape, tf.GradientTape() as disc1_tape,
         tf.GradientTape() as gen2_tape, tf.GradientTape() as disc2_tape:

        generated_images1 = generator1(noise, training=True)
        generated_images2 = generator2(noise, training=True)

        real_output1 = discriminator1(batch1[0], training=True)
        fake_output1 = discriminator1(generated_images1, training=True)
        fake_output1_cross = discriminator1(generated_images2, training=True)

        real_output2 = discriminator2(batch2[0], training=True)
        fake_output2 = discriminator2(generated_images2, training=True)
        fake_output2_cross = discriminator2(generated_images1, training=True)
        
        gen1_loss = 0.8*generator_loss(labels=arr, fake_output=fake_output1) + 0.2*generator_loss(labels=arr, fake_output=fake_output2_cross)
        disc1_loss = discriminator_loss(real_labels=batch1[1].numpy(), real_output=real_output1, fake_labels=arr, fake_output=fake_output1, fake_labels_cross=arr, fake_output_cross=fake_output1_cross) 

        gen2_loss = 0.8*generator_loss(labels=arr, fake_output=fake_output2) + 0.2*generator_loss(labels=arr, fake_output=fake_output1_cross)
        disc2_loss = discriminator_loss(real_labels=batch2[1].numpy(), real_output=real_output2, fake_labels=arr, fake_output=fake_output2, fake_labels_cross=arr, fake_output_cross=fake_output2_cross)                    

        gradients_of_generator1 = gen1_tape.gradient(gen1_loss, generator1.trainable_variables)
        gradients_of_discriminator1 = disc1_tape.gradient(disc1_loss, discriminator1.trainable_variables)

        gradients_of_generator2 = gen2_tape.gradient(gen2_loss, generator1.trainable_variables)
        gradients_of_discriminator2 = disc2_tape.gradient(disc2_loss, discriminator1.trainable_variables)

        generator_optimizer1.apply_gradients(zip(gradients_of_generator1, generator1.trainable_variables))
        discriminator_optimizer1.apply_gradients(zip(gradients_of_discriminator1, discriminator1.trainable_variables))

        generator_optimizer2.apply_gradients(zip(gradients_of_generator2, generator2.trainable_variables))
        discriminator_optimizer2.apply_gradients(zip(gradients_of_discriminator2, discriminator2.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for i in len(dataset):
      train_step(dataset[i], dataset[-i-1])

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator1,
                             epoch + 1,
                             seed,
                             "gen1")
    generate_and_save_images(generator2,
                             epoch + 1,
                             seed,"gen2")
    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

def generate_and_save_images(model, epoch, test_input, name ):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
      plt.savefig(name + "_" + 'image_at_epoch_{:04d}.png'.format(epoch))

train(train_dataset, EPOCHS)

generator1.save("./DoubleGANV1/generator1")
discriminator1.save("./DoubleGANV1/discriminator1")
generator2.save("./DoubleGANV1/generator2")
discriminator2.save("./DoubleGANV1/discriminator2")