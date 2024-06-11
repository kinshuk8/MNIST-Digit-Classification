import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images / 255.0
test_images = test_images / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Predict on the test images
predictions = model.predict(test_images)

# Function to plot the image and prediction
def plot_image_and_prediction(index):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plt.imshow(test_images[index], cmap=plt.cm.binary)
    plt.title(f"True label: {test_labels[index]}")
    plt.subplot(1,2,2)
    plt.bar(range(10), predictions[index])
    plt.title(f"Predicted: {tf.argmax(predictions[index])}")
    plt.show()

# Plot a few predictions
for i in range(5):
    plot_image_and_prediction(i)
