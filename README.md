# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet50 backbone.

# Getting Started

* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask R CNN implementation. 

* ([maize.py](samples/maize/maize.py)): Main Script for training, testing, and evaluating the model.

* ([image_prep.py](samples/maize/image_prep.py)): Script for downloading and preparing images from Segments.ai

* ([feature_extraction.py](samples/maize/feature_extraction.py)): Code for feature extraction


# Requirements

Install the following packages and their respective versions before running any code mentioned above. 

- TensorFlow: version 1.15.0 or 1.14.0
- TensorFlow GPU: version 1.15.0 or 1.14.0
- Keras: version 2.0.8
- Scikit-image: version 0.16.2
- h5py: version 2.10.0
- segments-ai
