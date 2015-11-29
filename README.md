# Custom Torch MSE Criterion using AutoGrad

A custom Torch criterion based on the [MSECriterion](https://github.com/torch/nn/blob/master/doc/criterion.md#nn.MSECriterion) to train a nueral network that will learn to either over or under predict. This may be useful in certain situations i.e when over predicting is financially more expensive than under predicting.  

The model uses twitter's [autograd](https://github.com/twitter/torch-autograd) to automatically differentiate the criterion.

The example trains a model using the [Boston Housing Data](http://lib.stat.cmu.edu/datasets/boston).

### Under Predicting

``` 
 th -i main.lua --under
```

### Over Prediciting

``` 
 th -i main.lua --over
```
