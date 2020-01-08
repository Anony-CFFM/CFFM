## Convolutional Feature-interacted Factorization Machines for Sparse Prediction in Recommender Systems
This is our Tensorflow implementation for the paper: 
> CFFM

### Introduction
Multi-modal Graph Convolution Network is a novel multi-modal recommendation framework based on graph convolutional networks, explicitly modeling modal-specific user preferences to enhance micro-video recommendation.

### Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:<br>
- Tensorflow == 1.14.0
- Tensorflow-tensorboard == 1.5.1
- Numpy == 1.17.4
- Pandas == 0.23.4
- Scikit-learn == 0.20.1
- Scipy == 1.1.0

### Example to Run the Codes
The instruction of commands has been clearly stated in the codes.

### Dataset
We provide three processed datasets: Book-Crossing, Frappe, and Movielens.

||#Interactions|#Users|#Items|Visual|Acoustic|Textual|
|:-|:-|:-|:-|:-|:-|:-|
|Kwai|1,664,305|22,611|329,510|2,048|-|100|
|Tiktok|726,065|36,656|76,085|128|128|128|
|Movielens|1,239,508|55,485|5,986|2,048|128|100|

-`train.npy`
   Train file. Each line is a user with her/his positive interactions with items: (userID and micro-video ID)  
-`val.npy`
   Validation file. Each line is a user with her/his 1,000 negative and several positive interactions with items: (userID and micro-video ID)  
-`test.npy`
   Test file. Each line is a user with her/his 1,000 negative and several positive interactions with items: (userID and micro-video ID)  
