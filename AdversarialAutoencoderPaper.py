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
import math


# ### Load and prepare the dataset
# 
# You will use the MNIST dataset to train the generator and the discriminator. The generator will generate handwritten digits resembling the MNIST data.


(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()


train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]


BUFFER_SIZE = 60000
BATCH_SIZE = 256


# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# ## Create the models
# 
# Both the generator and discriminator are defined using the [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

# ### The Generator
# 
# The generator uses `tf.keras.layers.Conv2DTranspose` (upsampling) layers to produce an image from a seed (random noise). Start with a `Dense` layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the `tf.keras.layers.LeakyReLU` activation for each layer, except the output layer which uses tanh.

num_fixed = 0
noise_dim = 2
def make_seedifier_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=(28, 28, 1)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(noise_dim + num_fixed))
    return model

    
def make_generator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(100, input_shape=(noise_dim + num_fixed,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
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

    model.add(layers.Dense(30, input_shape=(noise_dim + num_fixed,)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(30))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


# Use the (as yet untrained) generator to create an image.

seedifier = make_seedifier_model()
generator = make_generator_model()
discriminator = make_discriminator_model()


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mae = tf.keras.losses.MeanAbsoluteError()
# mean_average = tf.keras.losses.MAE()

# ### Discriminator loss
# 
# This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.

# ### Generator loss
# The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, we will compare the discriminators decisions on the generated images to an array of 1s.


def generator_loss(fake_output):
    loss = cross_entropy(tf.ones_like(fake_output), fake_output)*120
    # loss += cross_entropy(image_input, image_output)/(28*28)
    # loss += autoencoder_loss(image_input, image_output)
    return loss

def autoencoder_loss(image_input, image_output):
    loss = mae(image_input, image_output)*len(image_input)
    # loss = cross_entropy(image_input, image_output)/(28*28)
    return loss

def discriminator_loss(gaussian_output, seedifier_out):
  disc_gaussian = discriminator(gaussian_output, training=True)
  disc_seedifier = discriminator(seedifier_out, training=True)

  loss = cross_entropy(disc_gaussian, tf.ones_like(disc_gaussian)) 
  loss += cross_entropy(disc_seedifier, tf.zeros_like(disc_seedifier))

  adv_loss = cross_entropy(disc_seedifier, tf.ones_like(disc_seedifier))
  return adv_loss, loss


# The discriminator and the generator optimizers are different since we will train two networks separately.


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
seedifier_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# ## Define the training loop
# 


EPOCHS = 15
num_examples_to_generate = 16

printed_autoencoder_loss = -1
# printed_generator_loss = -1
printed_discriminator_loss = -1
printed_adv_loss = -1

# We will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)

originalimagefig = plt.figure(figsize=(4,4))
initialimages = []

for i in range(num_examples_to_generate):
  for batch in train_dataset:
    digitimage = batch[i]
    initialimages.append(digitimage)
    plt.subplot(4, 4, i+1)
    plt.imshow(digitimage[:, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
    
    break

plt.show()  

def normpdf(x, mean, sd):
    newmean = mean*np.ones(shape=x.shape)
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = np.exp(-np.square(x-newmean)/(2*var))
    return num/denom

def get_gaussian_mixture(example_count):
  # x_coords = np.random.normal(size=(example_count, 1))
  # y_coords = np.random.normal(size=(example_count, 1), scale=0.5)
  # x_coords += 5

  # combine = np.concatenate((x_coords, y_coords), axis=1)

  # result = np.zeros_like(combine)

  # for i in range(example_count):
  #   rotateamount = np.random.randint(10)
  #   theta = rotateamount*np.pi/5
  #   c, s = np.cos(theta), np.sin(theta)
  #   R = np.array([[c, -s], [s, c]])
  #   multiply = (R @ combine[i].T).T
  #   result[i] = multiply
  # return tf.convert_to_tensor(result)
  return tf.random.normal([example_count, 2], stddev=5.0)


randomseed = get_gaussian_mixture(num_examples_to_generate)

# The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
# @tf.function

def train_step(images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as seed_tape:
      # print(images.shape)
      originalseeds = seedifier(images, training=True)

      # fixedparams = originalseeds.numpy()[:, :num_fixed].reshape((-1, num_fixed))
      noiseparams = originalseeds.numpy()[:, num_fixed:].reshape((-1, noise_dim))
      gaussian_seeds = get_gaussian_mixture(noiseparams.shape[0])

      

      # generated_images = generator(seedsrandomized, training=True)

      autoencoder_output = generator(originalseeds, training=True)

      adv_loss, disc_loss = discriminator_loss(gaussian_seeds, originalseeds)
      seed_loss = autoencoder_loss(images, autoencoder_output) + adv_loss
      # gen_loss = seed_loss + generator_loss(fake_output)
      


      global printed_autoencoder_loss
      # global printed_generator_loss 
      global printed_discriminator_loss 
      global printed_adv_loss

      printed_autoencoder_loss = (seed_loss - adv_loss).numpy()
      # printed_generator_loss = (gen_loss - seed_loss).numpy()
      printed_discriminator_loss = disc_loss.numpy()
      printed_adv_loss = adv_loss.numpy()

    gradients_of_generator = gen_tape.gradient(seed_loss, generator.trainable_variables)
    gradients_of_seedifier = seed_tape.gradient(seed_loss, seedifier.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    seedifier_optimizer.apply_gradients(zip(gradients_of_seedifier, seedifier.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def printLosses():
  print("Autoencoder Loss: " + str(printed_autoencoder_loss))
  # print("Adversarial Loss: " + str(printed_generator_loss))
  print("Discriminator Loss: " + str(printed_discriminator_loss))
  print("Adversarial Loss: " +  str(printed_adv_loss))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for image_batch in dataset:
      
      train_step(image_batch)

    # Produce images for the GIF as we go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    printLosses()

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs)



# **Generate and save images**
#
  


def generate_grid_from_gaussian(rotate_factor, epoch):
  x = np.linspace(-2, 2, 10)
  y = np.linspace(-1, 1, 10)
  imagenum = 0

  # Generate grid of digits based on seed

  fig = plt.figure(figsize=(10,10))

  for b in y:
    for a in x:
      imagenum += 1
      plt.subplot(10, 10, imagenum)
      
      theta = rotate_factor*np.pi/5
      c, s = np.cos(theta), np.sin(theta)
      R = np.array([[c, -s], [s, c]])

      rotated = (R @ [[a+5],[b]]).T

      image = generator(tf.convert_to_tensor(rotated), training=False)
      plt.imshow(image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    # plt.savefig('image_at_epoch_{:04d}_generated_layer_{:04d}.png'.format(epoch, layer))
    
  plt.savefig('image_at_epoch_{:02d}_grid_gaussian{:01d}.png'.format(epoch, rotate_factor))

def generate_and_save_images(model, epoch):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  # testfig = plt.figure(figsize=(4,4))

  # # Combo version
  # firstseedarray = np.zeros(shape=(num_examples_to_generate, num_fixed))

  # # Autoencode Version
  # firstseedarray = np.zeros(shape=(num_examples_to_generate, num_fixed + noise_dim))

  # for i in range(num_examples_to_generate):
    # for batch in train_dataset:
      # digitimage = initialimages[i]
      # plt.subplot(4, 4, i+1)
      # plt.imshow(digitimage[:, :, 0] * 127.5 + 127.5, cmap='gray')
      # plt.axis('off')
      # print(digitimage.shape)

      # # Combo version
      # firstseedarray[i] = seedifier(tf.convert_to_tensor([digitimage]), training=False).numpy()[:, :num_fixed]

      # # Autoencode version
      # firstseedarray[i] = seedifier(tf.convert_to_tensor([digitimage]), training=False)

      # break
  # plt.savefig('preimage_at_epoch_{:04d}.png'.format(epoch))

  # # Combo version
  # finalinput = tf.concat([firstseedarray, randomseed], 1)

  finalinput = get_gaussian_mixture(num_examples_to_generate)

  # #Autoencode version
  # finalinput = firstseedarray

  # 2-d visualization 
  if (epoch % 10) == 0:
    for rotate in range(10):
      generate_grid_from_gaussian(rotate, epoch)

  predictions = model(finalinput, training=False)

  # Output of net with random seeds

  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4, 4, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()


# ## Train the model
# Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).
# 
# At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.

train(train_dataset, EPOCHS)

generator.save("./generatorAutoencoderBiggerNormal")
seedifier.save("./seedifierAutoencoderBiggerNormal")
discriminator.save("./discriminatorAutoencoderBiggerNormal")

print("DONE")