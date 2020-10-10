"""
train and evaluate flat breed classification model for predicting animal breed (1 out of 37 classes)

### Usage:
python breed_model_flat.py inception_v3
or: python breed_model_flat.py inception_resnet_v2
or: python breed_model_flat.py xception
"""

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import os

model_name = sys.argv[1] if len(sys.argv) > 1 else ""
TRAIN_BATCH_SIZE = 16 if model_name == 'xception' else 32

print(f'MODEL = {model_name}, BATCH_SIZE = {TRAIN_BATCH_SIZE}')
print('\n\nAugmentations:\nshear_range=0.2, zoom_range=0.2, horizontal_flip=True,'
      '\nwidth_shift_range=0.2, height_shift_range=0.2,'
      '\nrotation_range=20, brightness_range=[0.7, 1.1],')

# images will be resized to this shape, this is also the dims for layers
INPUT_SHAPE = [299, 299, 3]

"""LOAD DATAFRAMES"""
df = pd.read_csv(f"data_paths_and_classes_{'windows' if os.name == 'nt' else 'unix'}.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

train_df = df[df['train/test'] == 'train'][['path', 'cat/dog', 'breed']]
val_df = df[df['train/test'] == 'validation'][['path', 'cat/dog', 'breed']]
test_df = df[df['train/test'] == 'test'][['path', 'cat/dog', 'breed']]
num_of_classes = len(set(train_df['breed']))

"""CREATE IMAGE GENERATORS"""
"""TRAIN SET - This will create augmentations on-the-fly"""
if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    train_dataGen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       rotation_range=20, brightness_range=[0.7, 1.1],
                                       preprocessing_function=pre_process)
elif model_name == 'xception':
    pre_process = preprocess_input_xception
    train_dataGen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       rotation_range=20, brightness_range=[0.7, 1.1],
                                       preprocessing_function=pre_process)
else:  # inception_v3
    train_dataGen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                       width_shift_range=0.2, height_shift_range=0.2,
                                       rotation_range=20, brightness_range=[0.7, 1.1])

train_generator = train_dataGen.flow_from_dataframe(dataframe=train_df, x_col="path", y_col="breed",
                                                    class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                    batch_size=TRAIN_BATCH_SIZE)

"""VALIDATION SET"""
if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'xception':
    pre_process = preprocess_input_xception
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    val_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations

val_generator = val_data_gen.flow_from_dataframe(dataframe=val_df, x_col="path", y_col="breed",
                                                 class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                 batch_size=1, shuffle=False)

"""TEST SET"""
if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'xception':
    pre_process = preprocess_input_xception
    test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    test_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations

test_generator = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                   class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                   batch_size=1, shuffle=False)  # batch_size=1, shuffle=False for test!

"""PREPARE TENSORFLOW DATASETS FOR VALIDATION AND TEST"""
# we need the tf datasets because the generators yield infinite amount of images
# and while validating or testing we only wish 1 time every picture, without augmentations
val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32))
val_dataset = val_dataset.take(len(val_df))

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
test_dataset = test_dataset.take(len(test_df))  # Note: test_generator must have shuffle=False

"""DEFINE MODEL"""
if model_name == 'inception_v3':
    model = InceptionV3(weights=None, classes=num_of_classes)
elif model_name == 'inception_resnet_v2':
    model = InceptionResNetV2(weights=None, classes=num_of_classes)
elif model_name == 'xception':
    model = Xception(weights=None, classes=num_of_classes)
else:
    raise ValueError(f"not supported model name {model_name}")

"""MODEL PARAMS AND CALLBACKS"""
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
checkpoint = ModelCheckpoint(filepath=f'flat_weights_{model_name}.hdf5', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)

"""FIT MODEL"""
print('============ fit flat model ============')
model.fit(train_generator, epochs=100, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE),
          validation_data=val_dataset, callbacks=[checkpoint, early_stopping, reduce_lr])
# load the best (on validation) weights from .fit() phase
model.load_weights(filepath=f'flat_weights_{model_name}.hdf5')

print('============ predict flat model ============')
# we use the .predict() method and not .evaluate() so we can generate a confusion matrix (for example)
classes = train_generator.class_indices
inverted_classes = dict(map(reversed, classes.items()))
predictions = model.predict(test_dataset)
predictions = tf.argmax(predictions, axis=-1).numpy()
inverted_class_predictions = [inverted_classes[i] for i in predictions]

test_df['flat_prediction'] = inverted_class_predictions

accuracy = len(test_df[test_df['breed'] == test_df['flat_prediction']]) / len(test_df)
print(f'\n#RESULTS {model_name}# Flat Animal breed accuracy: {accuracy}. BATCH_SIZE: {TRAIN_BATCH_SIZE}')
