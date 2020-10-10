# Pet breed classification

This repository contains scripts for classification of pet images into their breeds using CNNs.<br>We use the Oxford-IIIT Pet Dataset (25 dog breeds, 12 cat breeds)  

[<img src="https://i.imgur.com/2YfHYVG.png">]()

## Abstract

Determine the breed of animal from an image is a multi-class classification problem that
can be daunting for manual human identification. This
problem is also very challenging for computers
because these animals, particularly cats, are very
deformable and there can be quite subtle differences
between the breeds.

Beyond the technical interest of fine-grained categorization,
extracting information from images of pets has a practical side too:
People devote a lot of attention to their domestic animals.
It is not unusual for owners to believe the incorrect breed for their pet,
so having a method of automated classification could provide a
gentle way of alerting them to such errors.

[<img src="https://i.imgur.com/Ncglgog.png">]()

## What is that?
This is my final project in the Technion course "Advanced Machine Learning and Optimization" (097209).<br>
A joint work of myself and [Gal Goldstein](https://www.linkedin.com/in/gal-goldstein-8776b0168/).

[Project report](https://drive.google.com/file/d/1P7VmEGp_8rhoUefoaoSLoUsyD8lIfiVD/view?usp=sharing)
<br>
[Project poster](https://drive.google.com/file/d/1buUUdys3v0eah8A6Mu6EI4ZnNcQuhyTk/view?usp=sharing)

## 2 classification methodologies
**Flat approach:**<br>
The breed is determined directly (37-class problem).

**Hierarchical approach:**<br>
The pet’s family is assigned first (cat / dog), and then the breed is
determined by conditioning on the family.
[<img src="https://i.imgur.com/RvAxbgf.png">]()

## Neural Structured Learning
Many Neural Networks are very sensitive to adversarial
perturbations, this is true also for our models - even
small changes in the input data (indistinguishable to the
human eye) can make the model predict the wrong class with high confidence level.
We use the Neural Structured Learning (NSL) framework
(by [TensorFlow](https://github.com/tensorflow/neural-structured-learning)) to improve the robustness of our models to adversarial
perturbations. By doing so we achieve higher stability
without losing accuracy.


## Prerequisites
Download [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) and place 'annotations' and 'images' folders in code directory. 

## Usage example

`python create_oxford_image_paths_file.py` - creating data paths file<br>
`python breed_model_flat.py inception_resnet_v2` - train flat model with Inception ResNet V2 architecture<br>
`python breed_model_hierarchical.py inception_resnet_v2` - train hierarchical model with Inception ResNet V2 architecture<br>
`python neural_structured_learning_model.py` - train NSL model with adversarial examples<br>
`python test_models_on_perturbations.py` - test both NSL model trained with adversarial examples and a basic flat model on perturbed images

## Results
Training the models from scratch with only ~200 images per class,<br>
we achieve accuracy of 81% on the test set using Inception ResNet V2 flat model (basic model).

The NSL Inception ResNet V2 achieves 77.5% accuracy on the test set (adversarial model).

When we use perturbed images the **accuracy of the basic model drops from 81% to 69%** 👎👎👎<br> 
but for the NSL adversarial model we experience only 1% drop in accuracy from 77.5% to 76.5% 🤩💪



