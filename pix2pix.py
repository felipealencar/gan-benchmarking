import tensorflow as tf
from tensorflow.keras import layers, models

def build_generator_pix2pix():
    inputs = tf.keras.Input(shape=(75, 75, 2))

    # Encoder
    down1 = layers.Conv2D(64, (2, 2), strides=(3, 3), padding='same')(inputs)
    down2 = layers.Conv2D(128, (1, 1), strides=(3, 3), padding='same')(layers.LeakyReLU()(down1))
    
    # Decoder
    up1 = layers.Conv2DTranspose(18 * 18 * 256, (1, 1), strides=(2, 2), padding='valid')(layers.LeakyReLU()(down2))
    up2 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='valid')(layers.LeakyReLU()(up1))
    up3 = layers.Conv2DTranspose(2, (5, 5), strides=(2, 2), padding='valid')(layers.LeakyReLU()(up2))

    # Output
    outputs = layers.Activation('tanh')(up3)
    return models.Model(inputs=inputs, outputs=outputs)

# Discriminator model
def build_discriminator_pix2pix():
    inputs = tf.keras.Input(shape=(75, 75, 4))

    # Convolutional layers
    conv1 = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(inputs)
    conv2 = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(layers.LeakyReLU()(conv1))
    conv3 = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same')(layers.LeakyReLU()(conv2))
    
    # Classification layer
    outputs = layers.Conv2D(1, (3, 3), strides=(2, 2), padding='same')(layers.LeakyReLU()(conv3))

    return models.Model(inputs=inputs, outputs=outputs)

# Define the loss functions
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# Discriminator loss function
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Generator loss function
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers for generator and discriminator
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Initialize generator and discriminator models
generator = build_generator_pix2pix()
discriminator = build_discriminator_pix2pix()
# Training loop
@tf.function
def train_step(input_images, target_images):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate fake images
        input_images = tf.expand_dims(input_images, axis=0)
        input_images = tf.cast(input_images, tf.float32)  # Convert to float32
        generated_images = generator(input_images, training=True)

        # Discriminator loss
        target_images = tf.expand_dims(target_images, axis=0)
        target_images = tf.cast(target_images, tf.float32)  # Convert to float32
        real_output = discriminator(tf.concat([input_images, target_images], axis=-1), training=True)
        fake_output = discriminator(tf.concat([input_images, generated_images], axis=-1), training=True)
        disc_loss = discriminator_loss(real_output, fake_output)

        # Generator loss
        gen_loss = generator_loss(fake_output)
        

    # Calculate gradients and apply them
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

# Training loop
def train(dataset, epochs):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Initialize losses
        generator_loss = 0
        discriminator_loss = 0

        for input_images, target_images in dataset:
            batch_generator_loss, batch_discriminator_loss = train_step(input_images, target_images)

            # Update losses
            generator_loss += batch_generator_loss
            discriminator_loss += batch_discriminator_loss

        # Calculate average losses
        generator_loss /= len(dataset)
        discriminator_loss /= len(dataset)

        # Print losses for the epoch
        print(f"Generator Loss: {generator_loss:.4f}")
        print(f"Discriminator Loss: {discriminator_loss:.4f}")
        print()


# Define a function to split the paired images into input and target images
def split_images(real_image, target_image):
    return real_image, target_image

# Assuming X_train_array has shape (num_samples, 75, 75, 2)
num_samples = X_train_array.shape[0]
# Generate random noise as input images
input_images = np.random.normal(size=(num_samples, 75, 75, 2))

# Create dataset with input and target images
dataset = tf.data.Dataset.from_tensor_slices((input_images, X_train_array))
train(dataset, epochs=10)
