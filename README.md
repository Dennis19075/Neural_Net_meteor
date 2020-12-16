## Index
- [Index](#index)
- [Project](#Project)
- [Libraries](#libraries)
- [Input](#input)
- [Model](#model)
  - [Input layer](#input-layer)
  - [Hidden layers](#hidden-layers)
  - [Output layer](#output-layer)
  - [Compile](#compile)
- [Training (Fit)](#training)
- [Validate](#validate)
  - [Real-time application (soon)](#real-time-application)
- [References](#references)

---
## Project

This indications are for **app/train2.py.** It gives better accuracy than app/training.py

The model I trained I will send through email. You just have to save inside **model** folder.

For use **app/train2.py** you need add **data** folder to this project with all of the dataset. Inside data create 2 folders (meteorite and no_meteorite)

For use **app/predict.py** you need add some example to **example_clips** folder.

---
## Libraries
You need the next libraries to train, save and run the model. Using pip:

**Tensorflow 2.0**

TensorFlow is an open source machine learning framework for everyone.
```python
pip install tensorflow==2.0
```
**Scikit** 

A set of python modules for machine learning and data mining

```python
pip install -U scikit-learn
```

**Imutils**

A series of convenience functions to make basic image processing functions.
```
pip install imutils
```

**Matplotlib**


Python plotting package. To plotting validate and error value in each epoch in the training.
```python
pip install matplotlib
```
**Numpy**

NumPy is the fundamental package for array computing with Python.
```python
pip install numpy
```
**Argparse**

Python command-line parsing library
```python
pip install argparse
```
**Pickle**

Create portable serialized representations of Python objects.
```python
pip install pickle4
```
**Open CV**

Wrapper package for OpenCV python bindings.
```python
pip install opencv-python
```
**SciPy**

SciPy is open-source software for mathematics, science, and engineering.
```python
pip install scipy
```
---
## Input

First of all, you need a folder (called **data** in my case). Where it has each folder with each type of value the neural network will predict.

This first version is using 1017 pictures (meteorite) and 822 pictures (no meteorite), 50 epochs to train and the accuracy was **76%**.

To execute train2.py just need:
```python
python app/train2.py
```


The next execution is for **app/training.py** a worse model.
```python
python app/training.py --dataset data --model model/activity.model \
	--label-bin model/lb.pickle --epochs 10
```

Now, resize the dataset to 180 x 180 (standard size) and set the batch_size. The batch size is the number of input data values that you are introducing at once in the model. It is recommended set with 32, 64, 128, it depends of something but I don't know what.

Separate the part of the dataset for train and validation. 80% to train and 20% to validate.

---
## Model

Make a function called **make_model**. It needs an input_shape and num_classes

### Input layer

It will rescaling the input data first.Then add Conv2D with 32 neurons and a window 3x3, BatchNormalization layers with **relu** activation function. Repeat the same 3 functions but now Conv2D with 64 neurons.

### Hidden layers

#### Layer list
- Activation function **relu**
- SeparableConv2D
- BatchNormalization
- Activation function **relu**
- SeparableConv2D
- BatchNormalization
- MaxPooling2D
- Conv2D (for residual. I don't understand this yet)
- SeparableConv2D
- BatchNormalization
- Activation function **relu**
- GlobalAveragePooling2D

### Output layer 

If the number of classes is equals to 2 the activation function for the output layer is  **sigmoid**. If the number of classes is equals to more than 2 use softmax. Softmax is a good activation function for classify many types. (classify clouds)

 **Sigmoid** works with 2 values between 0 and 1.

Finally add Dropout helps to prevent overfitting. Dropout uses the activation function sigmoid.

### Compile

Before starting to train, you need to define some parameters to save and compile the model.

First the name for the saved model.

1)	**Optimizer**: using **Adam** function that uses gradient descent to optimize. This optimizer is better than RMSprop, function that the tutorial was using.
2)	**Loss function**: binary-crossentropy as the loss function due the convolutional neural network just needs to recognize 2 types of categories.

3)	**Metrics**: to check results like the accuracy while the model is training in each epoch.

```python
model.compile(
  loss='binary_crossentropy',
  optimizer=keras.optimizers.Adam(1e-3),
    metrics=['accuracy'])
```
---

## Training (fit)

```python
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)
```

You can use the model.fit return value to plot the model's training. I add this for the next version, it is a good feedback to check how the neural network learn.

## Validate

### Run examples

Now, you have the model trained so you can test it with some examples. In the next instruction, you run the prediction of a clip.
```python
python app/predict.py --model model/save_at_1.h5 \
	--label-bin model/lb.pickle \
	--output output/meteorite_1frame.avi \
	--size 1 \
	--input /example_clips/test_0.mp4 
```

The predicted clip will save a copy with the results in **output** folder.

### Real-time application

  *Soon*

---
## References
- *https://www.pyimagesearch.com/2019/07/15/video-classification-with-keras-and-deep-learning/*
- *https://neurohive.io/en/popular-networks/resnet/*
- *https://pypi.org/*
- *https://keras.io/examples/vision/image_classification_from_scratch/*