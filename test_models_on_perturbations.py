"""
This code uses 2 models in order to show the power of adversarial regularization model
trained as part of the NSL framework
We will use InceptionResNetV2 architecture for both:
1. 'base' model - normally trained model to predict breed classification
2. 'adversarial' model - a model trained using the NSL framework with adversarial examples

We load the weights we trained for both models (flat),
And perturb examples to show the power of the adversarial model
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_inception_resnet_v2
import neural_structured_learning as nsl
import pandas as pd
import keras
import os

IMAGE_INPUT_NAME = 'input_1'  # if experiencing problems with this change to value of 'model.layers[0].name'
LABEL_INPUT_NAME = 'label'

INPUT_SHAPE = [299, 299, 3]
multiplier, adv_step_size, adv_grad_norm = 0.2, 0.2, 'l2'  # configuration for adversarial model


def convert_to_dictionaries(image, label):
    return {IMAGE_INPUT_NAME: image, LABEL_INPUT_NAME: label}


"""LOAD DATAFRAMES"""
df = pd.read_csv(f"data_paths_and_classes_{'windows' if os.name == 'nt' else 'unix'}.csv")
df['cat/dog'] = df['cat/dog'].astype(str)
df['breed'] = df['breed'].astype(str)

test_df = df[df['train/test'] == 'test'][['path', 'cat/dog', 'breed']]
num_of_classes = len(set(test_df['breed']))

pre_process = preprocess_input_inception_resnet_v2
test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
test_generator_1 = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                     class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                     batch_size=1, shuffle=False)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator_1,
    output_types=(tf.float32, tf.float32))
test_dataset = test_dataset.map(convert_to_dictionaries)
test_dataset = test_dataset.take(len(test_df))

"""BASE MODEL"""
base_model = InceptionResNetV2(weights=None, classes=num_of_classes)
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

"""NSL MODEL"""
adv_config = nsl.configs.make_adv_reg_config(multiplier=multiplier,
                                             adv_step_size=adv_step_size,
                                             adv_grad_norm=adv_grad_norm)
adv_model = nsl.keras.AdversarialRegularization(keras.models.clone_model(base_model),  # cloned the base model
                                                label_keys=[LABEL_INPUT_NAME],
                                                adv_config=adv_config)
adv_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# eval on 1 batch just to 'wakeup' the adversarial model (in order to load random weights)
adv_model.evaluate(test_dataset.take(1), verbose=0)

"""LOAD DATASET AGAIN BECAUSE OF EVALUATE"""
pre_process = preprocess_input_inception_resnet_v2
test_data_gen = ImageDataGenerator(preprocessing_function=pre_process)
test_generator_2 = test_data_gen.flow_from_dataframe(dataframe=test_df, x_col="path", y_col="breed",
                                                     class_mode="categorical", target_size=INPUT_SHAPE[:2],
                                                     batch_size=1, shuffle=False)

test_dataset = tf.data.Dataset.from_generator(
    lambda: test_generator_2,
    output_types=(tf.float32, tf.float32))
test_dataset = test_dataset.map(convert_to_dictionaries)
test_dataset = test_dataset.take(len(test_df))

"""LOAD WEIGHTS WE TRAINED"""
base_model.load_weights(r'flat_weights_inception_resnet_v2.hdf5')  # 81.14% acc on original test data
adv_model.load_weights(r'nsl_weights_inception_resnet_v2_0.2_0.2_l2.hdf5')  # 77.66% acc on original test data

"""DEFINE MODELS TO EVALUATE ON ADVERSARIAL EXAMPLES"""
models_to_eval = {
    'base': base_model,
    'adv-regularized': adv_model.base_model
}

metrics = {
    name: tf.keras.metrics.CategoricalAccuracy()
    for name in models_to_eval.keys()
}

# this model will generate the adversarial examples based on the trained base model (to see where his weakness is)
reference_model = nsl.keras.AdversarialRegularization(
    base_model,
    label_keys=[LABEL_INPUT_NAME],
    adv_config=adv_config)

reference_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc'])

perturbed_images, labels, predictions, probabilities = [], [], [], []

# perturb the images and get new perturbed batch for every original batch
# in the perturbed batch the images look very similar to the original ones (to the human eye)
# but the values are changed just a little so it will make the base model to get wrong classification

for k, batch in enumerate(test_dataset):
    # we perturbed the image, adding a small noise
    perturbed_batch = reference_model.perturb_on_batch(batch)

    # Clipping makes perturbed examples have the same range as regular ones.
    # (!!) super important to clip to the same original values that the original features had
    perturbed_batch[IMAGE_INPUT_NAME] = tf.clip_by_value(
        perturbed_batch[IMAGE_INPUT_NAME], -1.0, 1.0)
    y_true = perturbed_batch.pop(LABEL_INPUT_NAME)
    perturbed_images.append(perturbed_batch[IMAGE_INPUT_NAME].numpy())
    labels.append(y_true.numpy())
    predictions.append({})
    probabilities.append({})
    for name, model in models_to_eval.items():
        y_pred = model(perturbed_batch)
        metrics[name](y_true, y_pred)
        probabilities[-1][name] = y_pred
        predictions[-1][name] = tf.argmax(y_pred, axis=-1).numpy()

# evaluate the models on how they did on the perturbed data
for name, metric in metrics.items():
    print('%s model accuracy on perturbed data: %f' % (name, metric.result().numpy()))
