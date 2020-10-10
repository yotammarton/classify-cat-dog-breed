"""
This module uses adversarial regularization from NSL (neural structured learning)
to predict the breed (flat) of the animal (a multi-class classification problem from 37 different classes)
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import neural_structured_learning as nsl
import pandas as pd
import numpy as np
import os

IMAGE_INPUT_NAME = 'input_1'  # if experiencing problems with this change to value of 'model.layers[0].name'
LABEL_INPUT_NAME = 'label'

INPUT_SHAPE = [299, 299, 3]  # fits to inception_resnet_v2
model_name = 'inception_resnet_v2'  # you can change if you like, but change the preprocessing and/or input shape too
TRAIN_BATCH_SIZE = 16
multiplier, adv_step_size, adv_grad_norm = 0.2, 0.2, 'l2'  # parameters for the adversarial model

print(f'NEW RUN FOR NSL MODEL\n'
      f'MODEL = {model_name}, BATCH_SIZE = {TRAIN_BATCH_SIZE}\n'
      f'multiplier = {multiplier},  adv_step_size = {adv_step_size}, adv_grad_norm = {adv_grad_norm}')


def convert_to_dictionaries(image, label):
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


"""LOAD DATAFRAMES"""
df = pd.read_csv(f"data_paths_and_classes_{'windows' if os.name == 'nt' else 'unix'}.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

train_df = df[df['train/test'] == 'train'][['path', 'cat/dog', 'breed']]
val_df = df[df['train/test'] == 'validation'][['path', 'cat/dog', 'breed']]
test_df = df[df['train/test'] == 'test'][['path', 'cat/dog', 'breed']]
num_of_classes = len(set(train_df['breed']))

"""CREATE IMAGE GENERATORS"""
pre_process = preprocess_input_inception_resnet_v2
train_data_gen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                                    preprocessing_function=pre_process)

train_generator = train_data_gen.flow_from_dataframe(dataframe=train_df, x_col="path", y_col="breed",
                                                     class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                     batch_size=TRAIN_BATCH_SIZE)

val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
val_generator = val_data_gen.flow_from_dataframe(dataframe=val_df, x_col="path", y_col="breed",
                                                 class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                 batch_size=1, shuffle=False)

test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
test_generator = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                   class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                   batch_size=1, shuffle=False)

"""PREPARE TENSORFLOW DATASETS FOR TRAIN, VALIDATION AND TEST SETS"""
train_dataset = tf.data.Dataset.from_generator(
    lambda: train_generator,
    output_types=(tf.float32, tf.float32))
# convert the dataset to the desired format of NSL (dictionaries)
train_dataset = train_dataset.map(convert_to_dictionaries)

val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32))
val_dataset = val_dataset.map(convert_to_dictionaries)
val_dataset = val_dataset.take(len(val_df))

# same for test data
test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32))
test_dataset = test_dataset.map(convert_to_dictionaries)
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
test_dataset = test_dataset.take(len(test_df))  # Note: test_generator must have shuffle=False

"""DEFINE BASE MODEL"""
model = InceptionResNetV2(weights=None, classes=num_of_classes)

"""NSL"""
adversarial_config = nsl.configs.make_adv_reg_config(multiplier=multiplier,
                                                     adv_step_size=adv_step_size,
                                                     adv_grad_norm=adv_grad_norm)
adversarial_model = nsl.keras.AdversarialRegularization(model, label_keys=[LABEL_INPUT_NAME],
                                                        adv_config=adversarial_config)

checkpoint = ModelCheckpoint(filepath=f'nsl_weights_{model_name}_{multiplier}_{adv_step_size}_{adv_grad_norm}.hdf5',
                             save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)
adversarial_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print('============ fit adversarial model ============')
adversarial_model.fit(train_dataset, epochs=100, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE),
                      validation_data=val_dataset, callbacks=[checkpoint, early_stopping, reduce_lr])

adversarial_model.load_weights(filepath=f'nsl_weights_{model_name}_{multiplier}_{adv_step_size}_{adv_grad_norm}.hdf5')

print('================== inference ==================')
result = adversarial_model.evaluate(test_dataset)
print(f'#RESULTS# NSL model: \n{dict(zip(adversarial_model.metrics_names, result))}\n'
      f'model_name: {model_name}\n'
      f'multiplier: {multiplier}\n'
      f'adv_step_size: {adv_step_size}\n'
      f'adv_grad_norm: {adv_grad_norm}')
