# **Behavioral Cloning** 
by Hanbyul Yang, Aug 21, 2017

## Overview

This is a project of Self-Driving Car Nanodegree Program of Udacity.

The goals of this project is making simulation car drives around track without leaving the road. All of process are included such as collecting data, building ConvNet with Keras, training, validating and testing.
For the network model codes, Check [`model.py`](./model.py).

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"


### Files Submitted 
My project includes the following files, required and helper:
Required
- `model.py` containing the script to create and train the model
- `drive.py` for driving the car in autonomous mode
- `model.h5` containing a trained convolution neural network 
- `writeup.md` summarizing the results. This file. 
- `video.mp4` a video recording of vehicle driving autonomously around the track. (two full laps)

Helper files
- `video.py` script to make video file
- `figure.png` training history plot file. contains training and validation loss.
- `merge_data.py` helper file for merging training data.
- `README.md` containing simple descriptions of files.

### Model Architecture and Training Strategy

#### Model architecture 

I used [NVIDIA's architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and it worked very well. It contains 5 convolutional layers and 3 fully connected layers. relu activation is used after Each convolutional layer. First three conv layers has 5x5 kernel and stride 2, last two conv layers has 3x3 kernel with stride 1. All conv layer are padded 'valid'.

#### Training strategy

I followed general guide line of project description.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an ADAM optimizer, so the learning rate was not tuned manually. Mean squared error (MSE) is used for loss function.

#### Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, turning left and right driving. 

Especially, I made lots of data for turning the corner left and right.

For details about how I created the training data, see the next section. 

#### Design Approach

The overall strategy for deriving a model architecture was following successful model that already proven. At the first I tried LeNet but it couldn't drive well. So I applied NVIDIA's architecture.

By splitting my data into training(80%) and validation(20%) data, I checked both loss for not to falling overfitting.

To combat the overfitting, I got lots of data other than just center line driving.
There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected more data from those spots of the track.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving normal and reverse directions. Here is an example image of center lane driving:

![alt text][image2]

Then I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover center line. 

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I recorded left turns and right turns. Especially, I recorded sharp turns a lot.

![alt text][image6]
![alt text][image7]

After the collection process, I had X number of data points. I then preprocessed this data by reordering color channel from BGR to RGB since drive.py use RGB format and cv2.imread() read image with BGR format. Top 65 pixels and bottom 25 pixels are cropped. Data Normalization are applied for the fast and stable training.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
5 epochs are used for training.
