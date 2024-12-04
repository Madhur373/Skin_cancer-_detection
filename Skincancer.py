import numpy as np
import os 
from PIL import Image
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_image_folder = r'D:\SATHYABAMA\IDP -Project\archive (2)\test'
train_image_folder = r'D:\SATHYABAMA\IDP -Project\archive (2)\train'
  
def load_and_preprocess_images(folder, filenames, target_size):
    images = []
    for filename in filenames:
       
        img_path = os.path.join(folder, filename)
       
        img = Image.open(img_path)
        img = img.resize(target_size)
        # Convert the image to numpy array and normalize pixel values
        img_array = np.array(img) / 255.0
        images.append(img_array)
    return np.array(images)

# Define target size for resizing images
target_size = (256, 256)

# Load and preprocess images from each class folder
num_classes = 2 
X_train = []
Y_train = []
X_test = []
Y_test = []
for i, class_name in enumerate(["benign","malignant"]):
    tr_folder = os.path.join(train_image_folder, class_name)
    tr_filenames = os.listdir(tr_folder)
    tr_images = load_and_preprocess_images(tr_folder, tr_filenames, target_size)
    tr_labels = np.full(len(tr_filenames), fill_value=i)
    X_train.append(tr_images)
    Y_train.append(tr_labels)

    te_folder = os.path.join(test_image_folder, class_name)
    te_filenames = os.listdir(te_folder)
    te_images = load_and_preprocess_images(te_folder, te_filenames, target_size)
    te_labels = np.full(len(te_filenames), fill_value=i)
    X_test.append(te_images)
    Y_test.append(te_labels)


# Concatenate images and labels
X_train = np.concatenate(X_train, axis=0)
Y_train = np.concatenate(Y_train, axis=0)
X_test = np.concatenate(X_test, axis=0)
Y_test = np.concatenate(Y_test, axis=0)

x_train,x_test,y_train,y_test = train_test_split(X_train, Y_train,test_size=0.2)

# Shuffle the data trains
shuffle_indices = np.random.permutation(len(x_train))
x_train_shuffled = x_train[shuffle_indices]
y_train_shuffled = y_train[shuffle_indices]

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Create augmented data generators
train_generator = train_datagen.flow(x_train_shuffled, y_train_shuffled, batch_size=32)
# Buliding CNN Model 
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # Add dropout for regularization
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,epochs=100,batch_size =32)
# Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)


model.save('skincancer.h5')