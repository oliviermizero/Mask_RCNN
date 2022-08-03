# Mask R-CNN for Object Detection and Segmentation

This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet101 backbone.

# Getting Started
* ([model.py](mrcnn/model.py), [utils.py](mrcnn/utils.py), [config.py](mrcnn/config.py)): These files contain the main Mask RCNN implementation. 

* ([maize.py](samples/maize/maize.py)): Main Script for training, testing, and evaluating the model.

* ([image_prep.py](samples/maize/image_prep.py)): Script for downloading and preparing images from Segments.ai

* ([feature_extraction.py](samples/maize/feature_extraction.py)): Tentative code for feature extraction
