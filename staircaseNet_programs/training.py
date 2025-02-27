from keras.backend import binary_crossentropy
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from metrics import sensitivity, specificity, auc
from Staircasenet import staircase_net
# import tensorflow_addons as tfa
import matplotlib.pyplot as plt
H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def load_data(image_path, mask_path):
    image_files = sorted(glob(os.path.join(image_path, "*.tiff")))
    mask_files = sorted(glob(os.path.join(mask_path, "*.jpg")))

    if len(image_files) != len(mask_files):
        raise ValueError("Number of images and masks must be the same.")

    return image_files, mask_files

    
def shuffling(x, y):
    # Shuffle training images and masks
    x, y = shuffle(x, y, random_state=42)
    return x, y

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0  # Normalizing
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch_size=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset
    
# Setting
np.random.seed(42)
tf.random.set_seed(42)

# Directory to save files
create_dir("/content/drive/MyDrive/mini_project/files")

# Hyperparameters
batch_size = 2
lr = 1e-4
num_epochs = 60

model_path = os.path.join("/content/drive/MyDrive/mini_project/output_storage", "model.h5")
csv_path = os.path.join("/content/drive/MyDrive/mini_project/output_storage", "data.csv") 

# Dataset paths
train_x_images, train_x_masks = load_data("/content/drive/MyDrive/mini_project/a_data/preproccessing", "/content/drive/MyDrive/mini_project/a_data/train/mask")
valid_x_images, valid_x_masks = load_data("/content/drive/MyDrive/mini_project/a_data/test_preproccessing", "/content/drive/MyDrive/mini_project/a_data/test/mask")

# Shuffle training data
train_x_images, train_x_masks = shuffling(train_x_images, train_x_masks)

print(f"Train:{len(train_x_images)} - {len(train_x_masks)}")
print(f"Valid:{len(valid_x_images)} - {len(valid_x_masks)}")

train_dataset = tf_dataset(train_x_images, train_x_masks, batch_size=batch_size)
valid_dataset = tf_dataset(valid_x_images, valid_x_masks, batch_size=batch_size)

train_Step = len(train_x_images) // batch_size
valid_Step = len(valid_x_images) // batch_size

if (len(train_x_images) % batch_size) != 0:
    train_Step += 1
if (len(valid_x_images) % batch_size) != 0:
    valid_Step += 1    

# Model
model = staircase_net((H, W, 3))    
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', 'binary_accuracy', 'binary_crossentropy', sensitivity, specificity, auc])

# Define the callbacks
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6, verbose=1),
    CSVLogger(csv_path),
    TensorBoard(),
    EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_dataset,
    epochs=num_epochs,
    validation_data=valid_dataset,
    steps_per_epoch=train_Step,
    validation_steps=valid_Step,
    callbacks=callbacks
)

print("callbacks")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("/content/drive/MyDrive/mini_project/files/newloss.png")
