## Convolutional Feature-interacted Factorization Machines for Sparse Prediction in Recommender Systems
This is our Tensorflow implementation for the paper:  CFFM

## Introduction
Multi-modal Graph Convolution Network is a novel multi-modal recommendation framework based on graph convolutional networks, explicitly modeling modal-specific user preferences to enhance micro-video recommendation.

## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:<br>
- Tensorflow == 1.14.0
- Tensorflow-tensorboard == 1.5.1
- Numpy == 1.17.4
- Pandas == 0.23.4
- Scikit-learn == 0.20.1
- Scipy == 1.1.0

## Example to Run the Codes
The instruction of commands has been clearly stated in the codes.

## Dataset
We provide three processed datasets: Book-Crossing, MovieLens, and Frappe.

|Dataset|Book-Crossing|MovieLens|Frappe|
|:-|:-|:-|:-|
|Features|226,336|22,611|90,445|5,382|
|Fields|6|3|10|
|Records|1,213,367|2,006,856|288,606|
|Sparsity|99.97\%|99.99\%|99.81\%|

Each dataset is divided into a training set: XXX.train.libfm (70%), a validation set: XXX.validation.libfm (20%),and a test set: test.libfm (10%)
  
