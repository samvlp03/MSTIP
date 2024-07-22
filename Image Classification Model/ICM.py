import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Setup the environment with TensorFlow/Keras

# Step 2: Load a pre-trained model (e.g., VGG16) and modify the top layers for your classification task
input_shape = (32, 32, 3)
num_classes = 10

# Load the VGG16 model with pre-trained ImageNet weights, excluding the top fully connected layers
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))

# Add new top layers
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create a new model combining the base model and the new top layers
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers in the base model so they won't be trained
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 3: Prepare the dataset (e.g., CIFAR-10) and preprocess the images
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train, num_classes), to_categorical(y_test, num_classes)

# Data augmentation to improve model performance
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Step 4: Train the model and evaluate its performance
batch_size = 32
epochs = 5
history = model.fit(datagen.flow(x_train, y_train, batch_size=batch_size),
          steps_per_epoch=x_train.shape[0] // batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_accuracy}')
