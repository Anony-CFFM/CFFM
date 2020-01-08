## Convolutional Feature-interacted Factorization Machines for Sparse Prediction in Recommender Systems 
This is our Tensorflow implementation for the paper. 

## Introduction
Convolutional Feature-interacted Factorization Machine leverages outer product and inner product to encode pairwise cor-relations between feature-interacted dimensions,  and utilize CNN to extract signals from them. Besides, we enhance the linear regression in FM by introducing a linear attention network, which improves the representation ability with fewer paramaters.

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
- Book-Crossing dataset<br>
```python CFFM.py --dataset book-crossing --epoch 50 --batch_size 512 --inner_dims 32 --outer_dims 32 --lamda 0 --lr 0.05 --loss_type square_loss --num_field 6 --linear_att 1 --inner_conv 1 --outer_conv 1 --activation relu```
- MovieLens dataset<br>
```python CFFM.py --dataset ml-tag --epoch 50 --batch_size 1024 --inner_dims 32 --outer_dims 32 --lamda 0 --lr 0.05 --loss_type square_loss --num_field 3 --linear_att 1 --inner_conv 1 --outer_conv 1 --activation elu```
- Frappe dataset<br>
```python CFFM.py --dataset frappe --epoch 50 --batch_size 256 --inner_dims 32 --outer_dims 32 --lamda 0 --lr 0.05 --loss_type square_loss --num_field 10 --linear_att 1 --inner_conv 1 --outer_conv 1 --activation selu```



## Dataset
We provide three processed datasets: [Book-Crossing](http://www.informatik.uni-freiburg.de/~cziegler/BX/), [MovieLens](https://grouplens.org/datasets/movielens/latest/), and [Frappe](http://baltrunas.info/research-menu/frappe).

|Dataset|Book-Crossing|MovieLens|Frappe|
|:-|:-|:-|:-|
|Features|226,336|22,611|90,445|5,382|
|Fields|6|3|10|
|Records|1,213,367|2,006,856|288,606|
|Sparsity|99.97\%|99.99\%|99.81\%|

Each dataset is divided into:
 - The training set: **XXX.train.libfm** (70%)
 - The validation set: **XXX.validation.libfm** (20%)
 - The test set: **test.libfm** (10%)
 
 
 
 
  
