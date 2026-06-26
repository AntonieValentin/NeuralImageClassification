# Neural Image Classification

A PyTorch project focused on training and comparing neural networks for image classification. The project covers two computer vision tasks: facial expression recognition and skin lesion classification.

The main goal is to build a complete training pipeline, starting from data loading and augmentation, then moving through MLP and CNN models, fine-tuning pretrained networks, and comparing the results through controlled experiments.

## Overview

The project uses two image datasets with different challenges. The facial expression dataset contains RGB face images grouped into emotion classes such as happiness, sadness, anger, fear, surprise, disgust, and neutral. The skin lesion dataset contains dermatoscopic images grouped by diagnostic class.

Both datasets are relatively small and imbalanced, so the implementation focuses not only on model architecture, but also on preprocessing, augmentation, and class imbalance handling.

## Training Pipeline

The project follows a standard PyTorch workflow:

* custom dataset and dataloader setup;
* image resizing and normalization;
* train and validation loops;
* loss and metric tracking;
* TensorBoard logging;
* model comparison across multiple experiments.

The same general pipeline is reused for both datasets, with dataset-specific transforms and output classes.

## Models

Three types of models are used and compared:

### MLP

The MLP baseline flattens the input image and passes it through fully connected layers. It is useful as a simple reference point, even if it does not capture spatial structure as well as convolutional models.

### CNN

The CNN model extracts visual features using convolutional layers, activation functions, pooling, and optional BatchNorm. It is designed to stay compact while still being strong enough for low-resolution image classification.

### Fine-tuned Model

The fine-tuning stage uses a pretrained vision model, such as ResNet or MobileNet, and adapts its classification head to the target dataset. This helps reuse features learned from larger image collections and usually improves performance on smaller datasets.

## Experiments

The project includes several ablation-style experiments:

* MLP with and without dropout;
* CNN with and without BatchNorm;
* CNN trained with different augmentation strategies;
* training with and without class imbalance handling;
* fine-tuning with and without learning rate warmup.

Each experiment is tracked with training and validation curves, so the behavior of the models can be compared clearly.

## Data Augmentation

Augmentations are applied only during training. The choices depend on the dataset:

* geometric transforms such as flips, rotations, crops, and resizing;
* color transforms such as brightness, contrast, and saturation changes;
* more conservative color changes for medical images, where color can be important for the diagnosis.

The purpose is to improve generalization without changing the meaning of the image.

## Evaluation

Models are evaluated using both global and class-aware metrics:

* accuracy;
* macro and micro F1-score;
* confusion matrix;
* train and validation loss curves.

This is especially important because both tasks involve class imbalance, where accuracy alone can hide weak performance on minority classes.

## Notes

The project is mainly about understanding how neural models behave under different training choices. The final results are less important than the comparison between architectures, augmentation setups, imbalance handling, and fine-tuning strategies.

The result is a compact image classification workflow that connects practical PyTorch training with real computer vision problems.
