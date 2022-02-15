import numpy as np
import tensorflow as tf
import datetime

from keras.datasets import mnist

# Get dataset
(train_x, train_y), (val_x, val_y) = mnist.load_data()
train_x, val_x = train_x / 255, val_x / 255

# Define model
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile model into a tf Graph
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# Setup tensorboard logging
log_dir = "logs/mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train model
model.fit(x=train_x,
          y=train_y,
          batch_size=20,
          epochs=5, 
          validation_data=(val_x, val_y), 
          callbacks=[tensorboard_callback])

# Log test images to Tensorboard
file_writer = tf.summary.create_file_writer(log_dir)
with file_writer.as_default():
    for i in range(100):
        tf.summary.image(f"Predicted: {tf.argmax(model.predict(np.reshape(val_x[i], (1, 28, 28, 1)))[0])}", np.reshape(val_x[i], (1, 28, 28, 1)), step=i)

# Save the model
model.save("trained_model.h5")
