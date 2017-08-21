# **Behavioral Cloning** 
by Hanbyul Yang, Aug 21, 2017

## Overview

This is a project of Self-Driving Car Nanodegree Program of Udacity.

The goals of this project is making simulation car drives around track without leaving the road. All of process are included such as collecting data, building ConvNet with Keras, training, validating and testing.
For the network model codes, Check [`model.py`](./model.py).

[//]: # (Image References)

[normal_lap]: ./writeup_images/normal_lap_center_2017_08_20_12_25_15_918.jpg "Normal lap"
[normal_lap_reverse]: ./writeup_images/normal_lap_reverse_center_2017_08_20_13_14_29_145.jpg "Reverse direction lap"
[recovery_from_left]: ./writeup_images/recovery_from_left_center_2017_08_20_13_19_22_136.jpg "Recovery from left"
[recovery_from_right]: ./writeup_images/recovery_from_right_center_2017_08_20_13_22_03_742.jpg "Recovery from right"
[turn_left]: ./writeup_images/turn_left_center_2017_08_20_17_42_33_607.jpg "Turn left"
[turn_right]: ./writeup_images/turn_right_center_2017_08_20_17_58_09_395.jpg "Turn right"
[cropped]: ./writeup_images/cropped_center_2017_08_20_12_25_15_918.png "cropped normal lap"
[plot_loss]: ./writeup_images/figure.png "Loss plot "


### Files Submitted 
My project includes the following files, required and helper:
Required
- `drive.py` for driving the car in autonomous mode
- `model.h5` containing a trained convolution neural network 
- `model.py` containing the script to create and train the model
- `video.mp4` a video recording of vehicle driving autonomously around the track. (two full laps)
- `writeup.md` summarizing the results. This file. 

Other files
- `README.md` containing simple descriptions of files.
- `video.py` script to make video file
- `helper/`  helper python script files for this project
- `writeup_images/` images for writeup

### Model Architecture and Training Strategy

#### Model architecture 

Final model that I used is the [NVIDIA's architecture](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and it worked very well. It contains 5 convolutional layers and 3 fully connected layers. relu activation is used after each convolutional layer. First three conv layers has 5x5 kernel and stride 2, last two conv layers has 3x3 kernel with stride 1. All conv layer are padded 'valid'.

#### Training strategy

I followed the general guide line of project description.
- three laps of center lane driving and three reversed lap.
- some data (about one lap) recovery driving from the each side.
- Many data focusing on driving smoothly around curves, especially sharp corner.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used an ADAM optimizer. Mean squared error (MSE) is used for loss function.

#### Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, turning left and right driving. Especially, I made lots of data for turning the corner left and right.

For details about how I created the training data, see the next section. 

#### Design Approach

The overall strategy for deriving a model architecture was following successful model that already proven. So I used NVIDIA's architecture.

By splitting data into training(80%) and validation(20%), I checked both loss for not to falling overfitting.

To combat the overfitting, I got lots of data other than just center line driving.
There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I collected more data from those spots of the track.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving normal and reverse directions. Here are example images of center lane driving, left is normal lap and right is reverse direction lap.

![alt text][normal_lap]
![alt text][normal_lap_reverse]

Then I recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover center line.

![alt text][recovery_from_left]
![alt text][recovery_from_right]

I then collected left turns and right turns. Especially, I recorded sharp turns a lot.

![alt text][turn_left]
![alt text][turn_right]

After the collection process, I had 23310 number of data points. Below table shows the number of each training data that I split into 6 categories and count.

| Data recorded      | Count |
|:-------------------|------:|
| Normal lap | 5176 |
| Reversed normal lap | 5447 |
| Recovery from left | 1104 |
| Recovery from right | 1343 |
| Turn left | 5312 |
| Turn right | 4928 |


I preprocessed these data by reordering color channel from BGR to RGB since `drive.py` use RGB format and `cv2.imread()` read image with BGR format. Also, top 65 pixels and bottom 25 pixels are cropped. Then, data Normalization are applied for the fast and stable training.

For example, Belows are cropped image and its original.

![alt text][cropped]
![alt text][normal_lap]

I finally randomly shuffled the dataset and put 20% of the data into a validation set.
5 epochs are used for training. The plot below shows that both training and validation loss were decreased.

![alt text][plot_loss]

## Reflections
I used suggested or well-known architecture (NVIDIA's) and focused to handle the dataset. Low loss didn't guarantee successful autonomous driving but I referred as an index for overfitting. The keys of this project were categorizing dataset and getting enough data from driving corner.
