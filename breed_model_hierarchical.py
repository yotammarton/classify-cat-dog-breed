"""
train and evaluate hierarchical classification models:
 first perform binary classification (cat/dog) and then
 breed classification according to the predicted animal in the first binary model
 (1 out of 12 for cats || 1 out of 25 for dogs)

### Usage:
python breed_model_hierarchical.py inception_v3
or: python breed_model_hierarchical.py inception_resnet_v2
or: python breed_model_hierarchical.py xception
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
TRAIN_BATCH_SIZE = 32 if model_name == 'inception_v3' else 16

print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('NEW RUN FOR HIERARCHICAL MODEL')
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
num_of_classes = len(set(train_df['cat/dog']))

dogs_train_df = train_df[train_df['cat/dog'] == 'dog']
dogs_val_df = val_df[val_df['cat/dog'] == 'dog']
dogs_test_df = test_df[test_df['cat/dog'] == 'dog']
dogs_num_of_classes = len(set(dogs_train_df['breed']))

cats_train_df = train_df[train_df['cat/dog'] == 'cat']
cats_val_df = val_df[val_df['cat/dog'] == 'cat']
cats_test_df = test_df[test_df['cat/dog'] == 'cat']
cats_num_of_classes = len(set(cats_train_df['breed']))

"""CREATE IMAGE GENERATORS"""
"""TRAIN SET - This will create augmentations on-the-fly"""
image_datagen_params = dict(shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                            width_shift_range=0.2, height_shift_range=0.2,
                            rotation_range=20, brightness_range=[0.7, 1.1])  # params for ImageDataGenerator

if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    image_datagen_params['preprocessing_function'] = pre_process

    # binary model #
    train_dataGen = ImageDataGenerator(**image_datagen_params)
    # dogs model #
    dogs_train_dataGen = ImageDataGenerator(**image_datagen_params)

    # cats model #
    cats_train_dataGen = ImageDataGenerator(**image_datagen_params)

elif model_name == 'xception':
    pre_process = preprocess_input_xception
    image_datagen_params['preprocessing_function'] = pre_process

    # binary model #
    train_dataGen = ImageDataGenerator(**image_datagen_params)
    # dogs model #
    dogs_train_dataGen = ImageDataGenerator(**image_datagen_params)

    # cats model #
    cats_train_dataGen = ImageDataGenerator(**image_datagen_params)

else:  # inception_v3
    # binary model #
    train_dataGen = ImageDataGenerator(**image_datagen_params)
    # dogs model #
    dogs_train_dataGen = ImageDataGenerator(**image_datagen_params)
    # cats model #
    cats_train_dataGen = ImageDataGenerator(**image_datagen_params)

# generators
# binary model #
train_generator = train_dataGen.flow_from_dataframe(dataframe=train_df, x_col="path", y_col="cat/dog",
                                                    class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                    batch_size=TRAIN_BATCH_SIZE)
# dogs model #
dogs_train_generator = dogs_train_dataGen.flow_from_dataframe(dataframe=dogs_train_df, x_col="path", y_col="breed",
                                                              class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                              batch_size=TRAIN_BATCH_SIZE)
# cats model #
cats_train_generator = cats_train_dataGen.flow_from_dataframe(dataframe=cats_train_df, x_col="path", y_col="breed",
                                                              class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                              batch_size=TRAIN_BATCH_SIZE)

"""VALIDATION SET"""
if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    # binary model #
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # dogs model #
    dogs_val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # cats model #
    cats_val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'xception':
    pre_process = preprocess_input_xception
    # binary model #
    val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # dogs model #
    dogs_val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # cats model #
    cats_val_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    # binary model #
    val_data_gen = ImageDataGenerator(rescale=1. / 255)

    # dogs model #
    dogs_val_data_gen = ImageDataGenerator(rescale=1. / 255)

    # cats model #
    cats_val_data_gen = ImageDataGenerator(rescale=1. / 255)

# generators
# binary model #
val_generator = val_data_gen.flow_from_dataframe(dataframe=val_df, x_col="path", y_col="cat/dog",
                                                 class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                 batch_size=1, shuffle=False)
# dogs model #
dogs_val_generator = dogs_val_data_gen.flow_from_dataframe(dataframe=dogs_val_df, x_col="path", y_col="breed",
                                                           class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                           batch_size=1, shuffle=False)
# cats model #
cats_val_generator = cats_val_data_gen.flow_from_dataframe(dataframe=cats_val_df, x_col="path", y_col="breed",
                                                           class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                           batch_size=1, shuffle=False)

"""TEST SET"""
if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    # binary model #
    test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # dogs model #
    dogs_test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # cats model #
    cats_test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'xception':
    pre_process = preprocess_input_xception
    # binary model #
    test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # dogs model #
    dogs_test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)

    # cats model #
    cats_test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    # binary model #
    test_data_gen = ImageDataGenerator(rescale=1. / 255)

    # dogs model #
    dogs_test_data_gen = ImageDataGenerator(rescale=1. / 255)

    # cats model #
    cats_test_data_gen = ImageDataGenerator(rescale=1. / 255)

# generators
# binary model #
test_generator = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="cat/dog",
                                                   class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                   batch_size=1, shuffle=False)  # batch_size=1, shuffle=False for test!
# dogs model #
dogs_test_generator = dogs_test_data_gen.flow_from_dataframe(dataframe=dogs_test_df, x_col="path", y_col="breed",
                                                             class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                             batch_size=1, shuffle=False)
# cats model #
cats_test_generator = cats_test_data_gen.flow_from_dataframe(dataframe=cats_test_df, x_col="path", y_col="breed",
                                                             class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                             batch_size=1, shuffle=False)

"""PREPARE TENSORFLOW DATASETS FOR VALIDATION AND TEST"""
# we need the tf datasets because the generators yield infinite amount of images
# and while validating or testing we only wish 1 time every picture, without augmentations

# binary model #
val_dataset = tf.data.Dataset.from_generator(
    lambda: val_generator,
    output_types=(tf.float32, tf.float32))
val_dataset = val_dataset.take(len(val_df))

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
test_dataset = test_dataset.take(len(test_df))  # Note: test_generator must have shuffle=False

# dogs model #
dogs_val_dataset = tf.data.Dataset.from_generator(
    lambda: dogs_val_generator,
    output_types=(tf.float32, tf.float32))
dogs_val_dataset = dogs_val_dataset.take(len(dogs_val_df))

dogs_test_dataset = tf.data.Dataset.from_generator(
    lambda: dogs_test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
dogs_test_dataset = dogs_test_dataset.take(len(dogs_test_df))  # Note: test_generator must have shuffle=False

# cats model #
cats_val_dataset = tf.data.Dataset.from_generator(
    lambda: cats_val_generator,
    output_types=(tf.float32, tf.float32))
cats_val_dataset = cats_val_dataset.take(len(cats_val_df))

cats_test_dataset = tf.data.Dataset.from_generator(
    lambda: cats_test_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
cats_test_dataset = cats_test_dataset.take(len(cats_test_df))  # Note: test_generator must have shuffle=False

"""DEFINE MODEL"""
if model_name == 'inception_v3':
    binary_model = InceptionV3(weights=None, classes=num_of_classes)
    dogs_model = InceptionV3(weights=None, classes=dogs_num_of_classes)
    cats_model = InceptionV3(weights=None, classes=cats_num_of_classes)
elif model_name == 'inception_resnet_v2':
    binary_model = InceptionResNetV2(weights=None, classes=num_of_classes)
    dogs_model = InceptionResNetV2(weights=None, classes=dogs_num_of_classes)
    cats_model = InceptionResNetV2(weights=None, classes=cats_num_of_classes)
elif model_name == 'xception':
    binary_model = Xception(weights=None, classes=num_of_classes)
    dogs_model = Xception(weights=None, classes=dogs_num_of_classes)
    cats_model = Xception(weights=None, classes=cats_num_of_classes)
else:
    raise ValueError(f"not supported model name {model_name}")

"""MODEL PARAMS AND CALLBACKS"""
# binary model #
binary_model.compile(optimizer='adam', loss='BinaryCrossentropy',
                     metrics=['accuracy'])
binary_checkpoint = ModelCheckpoint(filepath=f'advanced_weights_binary_{model_name}.hdf5', save_best_only=True,
                                    verbose=1)
binary_early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
binary_reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)

# dogs model #
dogs_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
dogs_checkpoint = ModelCheckpoint(filepath=f'advanced_weights_dogs_{model_name}.hdf5', save_best_only=True, verbose=1)
dogs_early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
dogs_reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)

# cats model #
cats_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cats_checkpoint = ModelCheckpoint(filepath=f'advanced_weights_cats_{model_name}.hdf5', save_best_only=True, verbose=1)
cats_early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)
cats_reduce_lr = ReduceLROnPlateau(patience=5, verbose=1)

"""FIT MODEL"""
print('============ binary model fit ============')
binary_model.fit(train_generator, epochs=100, steps_per_epoch=np.ceil(len(train_df) / TRAIN_BATCH_SIZE),
                 validation_data=val_dataset, callbacks=[binary_checkpoint, binary_early_stopping, binary_reduce_lr])
# load the best (on validation) weights from .fit() phase
binary_model.load_weights(filepath=f'advanced_weights_binary_{model_name}.hdf5')

print('============ dogs model fit ============')
dogs_model.fit(dogs_train_generator, epochs=100, steps_per_epoch=np.ceil(len(dogs_train_df) / TRAIN_BATCH_SIZE),
               validation_data=dogs_val_dataset, callbacks=[dogs_checkpoint, dogs_early_stopping, dogs_reduce_lr])
# load the best (on validation) weights from .fit() phase
dogs_model.load_weights(filepath=f'advanced_weights_dogs_{model_name}.hdf5')

print('============ cats model fit ============')
cats_model.fit(cats_train_generator, epochs=100, steps_per_epoch=np.ceil(len(cats_train_df) / TRAIN_BATCH_SIZE),
               validation_data=cats_val_dataset, callbacks=[cats_checkpoint, cats_early_stopping, cats_reduce_lr])
# load the best (on validation) weights from .fit() phase
cats_model.load_weights(filepath=f'advanced_weights_cats_{model_name}.hdf5')

"""EVALUATE MODELS"""
print('============ binary model evaluate ============')
binary_result = binary_model.evaluate(test_dataset)
print(dict(zip(binary_model.metrics_names, binary_result)))

# we want to know how good the dogs model on all the real dogs, and not only on the ones who predicted as dogs
print('============ dogs model evaluate ============')
dogs_result = dogs_model.evaluate(dogs_test_dataset)
print(dict(zip(dogs_model.metrics_names, dogs_result)))

# we want to know how good the cats model on all the real cats, and not only on the ones who predicted as cats
print('============ cats model evaluate ============')
cats_result = cats_model.evaluate(cats_test_dataset)
print(dict(zip(cats_model.metrics_names, cats_result)))

"""PREDICT MODELS"""
# binary prediction #
classes = train_generator.class_indices
inverted_classes = dict(map(reversed, classes.items()))
binary_predictions = binary_model.predict(test_dataset)
binary_predictions = tf.argmax(binary_predictions, axis=-1).numpy()
binary_predictions = [inverted_classes[i] for i in binary_predictions]

test_df['binary_prediction'] = binary_predictions

"""BREED MODELS"""

# dog breed prediction #
predicted_as_dogs_df = test_df[test_df['binary_prediction'] == 'dog']

if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    predicted_as_dogs_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'xception':
    pre_process = preprocess_input_xception
    predicted_as_dogs_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    predicted_as_dogs_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations

predicted_as_dogs_generator = predicted_as_dogs_data_gen.flow_from_dataframe(dataframe=predicted_as_dogs_df,
                                                                             x_col="path",
                                                                             y_col="breed",
                                                                             class_mode="categorical",
                                                                             target_size=INPUT_SHAPE[:2],
                                                                             batch_size=1,
                                                                             shuffle=False)
dogs_classes = dogs_train_generator.class_indices  # super important to take the classes from train!
dogs_inverted_classes = dict(map(reversed, dogs_classes.items()))

predicted_as_dogs_dataset = tf.data.Dataset.from_generator(
    lambda: predicted_as_dogs_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
predicted_as_dogs_dataset = predicted_as_dogs_dataset.take(len(predicted_as_dogs_df))

dogs_predictions = dogs_model.predict(predicted_as_dogs_dataset)
dogs_predictions = tf.argmax(dogs_predictions, axis=-1).numpy()
dogs_predictions = [dogs_inverted_classes[i] for i in dogs_predictions]

predicted_as_dogs_df['breed_prediction'] = dogs_predictions

# cat breed prediction #
predicted_as_cats_df = test_df[test_df['binary_prediction'] == 'cat']

if model_name == 'inception_resnet_v2':
    pre_process = preprocess_input_inception_resnet_v2
    predicted_as_cats_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
elif model_name == 'xception':
    pre_process = preprocess_input_xception
    predicted_as_cats_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
else:
    predicted_as_cats_data_gen = ImageDataGenerator(rescale=1. / 255)  # without augmentations

predicted_as_cats_generator = predicted_as_cats_data_gen.flow_from_dataframe(dataframe=predicted_as_cats_df,
                                                                             x_col="path",
                                                                             y_col="breed",
                                                                             class_mode="categorical",
                                                                             target_size=INPUT_SHAPE[:2],
                                                                             batch_size=1,
                                                                             shuffle=False)

cats_classes = cats_train_generator.class_indices  # super important to take the classes from train!
cats_inverted_classes = dict(map(reversed, cats_classes.items()))

predicted_as_cats_dataset = tf.data.Dataset.from_generator(
    lambda: predicted_as_cats_generator,
    output_types=(tf.float32, tf.float32))
# for test data we dont want to generate infinite data, we just want the amount of data in the test (that's why take())
predicted_as_cats_dataset = predicted_as_cats_dataset.take(len(predicted_as_cats_df))

cats_predictions = cats_model.predict(predicted_as_cats_dataset)
cats_predictions = tf.argmax(cats_predictions, axis=-1).numpy()
cats_predictions = [cats_inverted_classes[i] for i in cats_predictions]

predicted_as_cats_df['breed_prediction'] = cats_predictions

"""RESULTS CALCULATION"""

concat_df = pd.concat([predicted_as_dogs_df, predicted_as_cats_df])

binary_accuracy = len(test_df[test_df['cat/dog'] == test_df['binary_prediction']]) / len(test_df)
print(f'#RESULTS {model_name}# Hierarchical animal binary accuracy: {binary_accuracy}. Batchsize: {TRAIN_BATCH_SIZE}')

final_accuracy = len(concat_df[concat_df['breed'] == concat_df['breed_prediction']]) / len(concat_df)
print(
    f'#RESULTS {model_name}# Hierarchical breed accuracy (out of 37): {final_accuracy}. Batchsize: {TRAIN_BATCH_SIZE}')

dogs_breed_accuracy = \
    len(predicted_as_dogs_df[predicted_as_dogs_df['breed'] == predicted_as_dogs_df['breed_prediction']]) / \
    len(dogs_test_df)
print(f'#RESULTS {model_name}# Hierarchical dogs breed accuracy (out of 25):'
      f' {dogs_breed_accuracy}. Batchsize: {TRAIN_BATCH_SIZE}')

cats_breed_accuracy = \
    len(predicted_as_cats_df[predicted_as_cats_df['breed'] == predicted_as_cats_df['breed_prediction']]) / \
    len(cats_test_df)
print(f'#RESULTS {model_name}# Hierarchical cats breed accuracy (out of 12):'
      f' {cats_breed_accuracy}. Batchsize: {TRAIN_BATCH_SIZE}')
